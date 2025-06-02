"""MongoDB-based system state management."""

import copy
import pickle
import traceback
from loguru import logger
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import settings

from storage.weight_storage import WeightStore

from orchestrator.serializers import LossReport
from orchestrator.miner_registry import MinerRegistry
from orchestrator import ORCHESTRATOR_NON_SERIALIZABLE_FIELDS

from storage.activation_storage import ActivationStore, Activation
from pymongo.server_api import ServerApi
from motor.motor_asyncio import AsyncIOMotorClient
from motor.motor_asyncio import AsyncIOMotorCollection

from utils.partitions import PartitionManager
from utils.shared_states import MergingPhase


class SystemState(BaseModel):
    """Holds the complete system state for recovery."""

    # TODO: This should really agnostically import the data from the
    # Orchestrator class so we don't need to manually add all the fields
    identifier: Optional[str] = None
    miner_registry: MinerRegistry
    activation_store: ActivationStore
    weight_store: WeightStore
    merging_phases: list[MergingPhase]
    miners_with_submitted_scores: Dict[str, str]
    global_miner_weights: Dict[int, List[float]]
    tracked_activations: Dict[str, List[Dict[str, Any]]]
    miner_scores: Dict[int, List[float]]
    losses: Dict[str, List[LossReport]]
    TIMEOUT: int
    total_forwards: int
    total_backwards: int
    total_completed: int
    N_LAYERS: int
    netuid: Optional[int] = None
    partition_manager: PartitionManager

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_mongo(cls, data: Dict[str, Any]) -> "SystemState":
        """Create a SystemState instance from MongoDB data.

        Args:
            data: Dictionary containing the MongoDB document data

        Returns:
            SystemState: A new SystemState instance
        """

        state_data = copy.deepcopy(data)

        identifier = str(state_data["_id"])
        state_data["identifier"] = identifier
        logger.info(f"Loaded system state from MongoDB (id: {identifier})")

        classes = cls.reconstruct_classes(data=state_data)

        for class_name, class_data in classes.items():
            state_data[class_name] = copy.deepcopy(class_data)

        state = cls(**state_data)
        return state

    @staticmethod
    def reconstruct_classes(
        data: Dict[str, Any],
    ) -> dict[str, ActivationStore | WeightStore | MinerRegistry | Dict[str, List[LossReport]] | PartitionManager]:
        """Reconstruct the activation and weight stores from stored data.

        Returns:
            tuple[ActivationStore, WeightStore]: Reconstructed store instances
        """
        # Reconstruct activation store
        activation_store = ActivationStore(N_LAYERS=data["N_LAYERS"])

        activations = {k: Activation(**v) for k, v in data["activation_store"]["activations"].items()}
        activation_store.activations = activations

        activations = {k: Activation(**v) for k, v in data["activation_store"]["initial_activations"].items()}
        activation_store.initial_activations = activations

        # Reconstruct weight store
        weight_store = WeightStore(
            weights=data["weight_store"]["weights"],
            layer_weights=data["weight_store"]["layer_weights"],
            store_uid=data["weight_store"]["store_uid"],
            timestamp=data["weight_store"]["timestamp"],
        )
        partition_manager = PartitionManager(
            partitions=data["partition_manager"]["partitions"],
            original_weight_paths=data["partition_manager"]["original_weight_paths"],
        )
        losses = {k: [LossReport(**v) for v in data["losses"][k]] for k in data["losses"].keys()}

        # Reconstruct miner registry
        miner_registry = MinerRegistry(miner_hotkeys=[])
        for hotkey, miner_data in data["miner_registry"]["registry"].items():
            miner_registry.add_miner_to_registry(hotkey)
            for attr, value in miner_data.items():
                miner_registry.set_miner_attribute(hotkey, attr, value)

        return {
            "activation_store": activation_store,
            "weight_store": weight_store,
            "miner_registry": miner_registry,
            "losses": losses,
            "partition_manager": partition_manager,
        }


class MongoStateManager:
    """Manages system state persistence and recovery using MongoDB."""

    def __init__(
        self,
        connection_string: str = settings.MONGODB_CONNECTION_STRING,
        db_name: str = settings.MONGODB_DB_NAME,
    ):
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
        self.collection = None

    @classmethod
    async def create(
        cls,
        connection_string: str = settings.MONGODB_CONNECTION_STRING,
        db_name: str = settings.MONGODB_DB_NAME,
    ) -> "MongoStateManager":
        """Create and initialize a new MongoStateManager instance.

        Args:
            connection_string: MongoDB connection string
            db_name: Name of the database to use

        Returns:
            Initialized MongoStateManager instance
        """
        instance = cls(connection_string=connection_string, db_name=db_name)
        await instance.initialize()
        return instance

    async def initialize(self):
        """Initialize the MongoDB connection and verify it works."""
        # Create a new client with robust connection settings
        logger.info(f"Initializing MongoDB connection with {self.connection_string}")
        self.client = AsyncIOMotorClient(
            self.connection_string,
            server_api=ServerApi("1"),
            tls=True,
            tlsAllowInvalidCertificates=True,  # Allow invalid certificates
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
        )
        self.db = self.client[self.db_name]
        self.collection: AsyncIOMotorCollection = self.db.system_states

        # Test connection
        await self._test_connection()

    async def _test_connection(self):
        """Test the MongoDB connection."""
        try:
            await self.client.admin.command("ping")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

    async def save_state(self, orchestrator: Any, on_weights_merged: bool = False) -> str:
        """Save the current system state to MongoDB using pickle serialization."""
        logger.info("Attempting to save current orchestrator state to MongoDB")
        try:
            # Get all field names from the Pydantic model
            all_fields = orchestrator.__class__.model_fields.keys()

            # Create state dict with all serializable fields
            state_dict = {}
            for field in all_fields:
                if field not in ORCHESTRATOR_NON_SERIALIZABLE_FIELDS and hasattr(orchestrator, field):
                    value = getattr(orchestrator, field)
                    # Clear any processing_task in Activation objects before copying
                    if field == "activation_store":
                        for activation in value.activations.values():
                            activation.processing_task = None
                    # Now safe to deep copy
                    value = copy.deepcopy(value)
                    state_dict[field] = value

            # Pickle the state dict
            try:
                pickled_state = pickle.dumps(state_dict)
            except TypeError as e:
                logger.error(f"Pickling error: {e}")
                logger.error("Attempting to identify problematic fields...")
                for field, value in state_dict.items():
                    try:
                        pickle.dumps(value)
                    except Exception as e:
                        logger.error(f"Cannot pickle field {field}: {e}")
                raise

            # Create document for MongoDB
            state_doc = {
                "timestamp": datetime.now(timezone.utc),
                "pickled_state": pickled_state,
                "version": "1.0",  # Add version for future compatibility
                "on_weights_merged": on_weights_merged,
            }

            # Verify MongoDB connection before insert
            logger.debug("Verifying MongoDB connection...")
            await self._test_connection()

            try:
                result = await self.collection.insert_one(state_doc)
                logger.info(f"System state saved to MongoDB with ID: {result.inserted_id}")
                logger.debug(f"Saved fields: {list(state_dict.keys())}")
                return str(result.inserted_id)
            except Exception as e:
                logger.error(f"Error inserting state into MongoDB: {e}")
                raise

        except Exception as e:
            logger.error(f"Error saving system state to MongoDB: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            raise

    async def load_system_state(self) -> Dict[str, Any]:
        """Load the most recent system state.

        Returns:
            Dict[str, Any]: The loaded orchestrator state from MongoDB
        """
        logger.info("Attempting to load most recently saved orchestrator state...")
        state: Dict[str, Any] = await self.load_latest_state()
        return state

    async def load_latest_state(self) -> Optional[Dict[str, Any]]:
        """Load the most recent system state from MongoDB using pickle deserialization.

        Returns:
            Optional[Dict[str, Any]]: The loaded state dictionary or None if no states exist
        """
        logger.info("Attempting to load latest system state from MongoDB")
        try:
            # Find most recent state
            cursor = self.collection.find().sort("timestamp", -1).limit(1)
            state_list = await cursor.to_list(length=1)

            if not state_list:
                logger.warning("No system states found in MongoDB, starting from scratch")
                return None

            state_doc = state_list[0]

            # Check if this is a pickled state
            if "pickled_state" not in state_doc:
                logger.warning("Found old format state document, skipping")
                return None

            # Unpickle the state
            try:
                state_dict = pickle.loads(state_doc["pickled_state"])
                logger.info(f"Successfully loaded state from MongoDB (id: {state_doc['_id']})")
                return state_dict

            except Exception as e:
                logger.error(f"Error unpickling state: {e}")
                logger.error(f"Detailed error: {traceback.format_exc()}")
                return None

        except Exception as e:
            logger.error(f"Error loading system state from MongoDB: {e}")
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return None

    async def _cleanup_old_states(self, keep_last: int = 5):
        """Remove old states from MongoDB, keeping only the most recent ones.

        Args:
            keep_last: Number of most recent states to keep
        """
        try:
            # Get count of states
            count = await self.collection.count_documents({})
            if count <= keep_last:
                return

            # Find timestamp of the nth most recent state
            cursor = self.collection.find().sort("timestamp", -1).skip(keep_last).limit(1)
            old_states = await cursor.to_list(length=1)

            if old_states:
                # Delete all states older than this one
                await self.collection.delete_many({"timestamp": {"$lt": old_states[0]["timestamp"]}})
                logger.debug(f"Cleaned up old states from MongoDB, keeping {keep_last} most recent")

        except Exception as e:
            logger.error(f"Error cleaning up old states from MongoDB: {e}")

    async def close(self):
        """Close the MongoDB connection."""
        self.client.close()

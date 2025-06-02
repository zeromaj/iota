# Fields that should never be serialized
ORCHESTRATOR_NON_SERIALIZABLE_FIELDS = {
    "lock",
    "validator_init_lock",
    "metagraph_syncer",
    "metagraph",
    "config",
    "subtensor",
    "dashboard_reporter",
    "validator_pool",
    "wallet_name",
    "wallet_hotkey",
}

# IOTA

In August 2024, Bittensor’s Subnet 9 (SN9) demonstrated that a distributed network of incentivized, permissionless actors could each pretrain large language models (LLMs) ranging from 700 million to 14 billion parameters, while surpassing established baselines. While that work validated blockchain-based decentralized pretraining as viable, it contained core issues: 

1. Every miner had to fit an entire model locally, and
2. “winner-takes-all” rewards encouraged model hoarding.

Here we introduce **IOTA (Incentivized Orchestrated Training Architecture)**, an architecture that addresses these limitations by transforming SN9’s previously isolated competitors into a single cooperating unit that can scale arbitrarily while still rewarding each contributor fairly. IOTA is a data- and pipeline-parallel training algorithm designed to operate on a network of heterogeneous, unreliable devices in adversarial and trustless environments. The result is a permissionless system that:

1. is capable of pretraining frontier-scale models without per-node GPU bloat,  
2. tolerates unreliable devices and, 
3. aligns participants through transparent token economics.

### Install
1. First install uv (https://docs.astral.sh/uv/)
2. Install packages using `uv sync`
3. then activate the environment using `source .venv/bin/activate`

### Miner Documentation
Running the miner is as easy as `python launch_miner.py`. For more information, reference [the official miner docs](https://docs.macrocosmos.ai/subnets/subnet-9-pre-training/subnet-9-iota-mining-setup-guide)

You can also run many miners on the same machine by running `python launch_miner.py --num_miners NUMBER_OF_MINERS`

### Validation Documentation
Running the validator `python launch_validator.py`. For more information, reference [the official validator docs](https://docs.macrocosmos.ai/subnets/subnet-9-pre-training/subnet-9-validating)

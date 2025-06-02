# IOTA

**I**ncentivized **O**rchestrated **T**raining **A**rchitecture

### Install
1. First install uv (https://docs.astral.sh/uv/)
2. Install packages using `uv sync`
3. then activate the environment using `source .venv/bin/activate`

### Miner Documentation
Running the miner is as easy as `python launch_miner.py`. For more information, reference [the official miner docs](https://docs.macrocosmos.ai/subnets/subnet-9-pre-training/subnet-9-iota-mining-setup-guide)

You can also run many miners on the same machine by running `python launch_miner.py --num_miners NUMBER_OF_MINERS`

### Validation Documentation
Running the validator `python launch_validator.py`. For more information, reference [the official validator docs](https://docs.macrocosmos.ai/subnets/subnet-9-pre-training/subnet-9-validating)

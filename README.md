<div align="center">

# IOTA

</div>

**I**ncentivized **O**rchestrated **T**raining **A**rchitecture (IOTA) is a framework for pretraining large language models across a network of heterogeneous, unreliable, permissionless and token incentivized machines. IOTA employs a data- and pipeline-parallel architecture to accelerate training and reduce hardware requirements for participants.

<div align="center">

<a href="https://iota.macrocosmos.ai">
  <img src="./assets/iota-page.png" alt="iota" width="600"/>
</a>

*explore above*

</div>

## **Overview**:
- Orchestrator distributes model layers across heterogeneous miners and streams activations between them.
- All network communication is mediated via the orchestrator, and a shared S3 bucket is used to store activations and layer weights.
- Miners compete to process as many activations as possible in the training stage.
- Miners periodically upload their local weights and merge their activations using a variant of Butterfly All-Reduce.
- Validators spot-check miners to ensure that work was performed as required.

For a more comprehensive overview, please refer to our technical paper [here](https://www.macrocosmos.ai/research/iota_primer.pdf).

<div align="center">
    <a href="https://www.macrocosmos.ai/research/iota_primer.pdf">
    <img src="./assets/iota-paper-page.png" alt="iota" width="600"/>
    </a>
</div>


## Current Run Information 📉
1. **15B parameter** Llama-inspired architecture with uninterrupted residual flow (see paper for details)
2. **5 layers**, breaking the model into 5 distinct training sections (1 head, 1 tail, 3 body)

## Comprehensive Dashboard
Visualizing the state of the network, the number of miners the number of layers, and general metrics is paramount to understanding the training process. We provide a comprehensive dashboard [here](https://iota.macrocosmos.ai/dashboard/mainnet)

<div align="center">
    <a href="https://iota.macrocosmos.ai/dashboard/mainnet">
    <img src="./assets/iota-dashboard.png" alt="iota" width="600"/>
    </a>
</div>



## Installation
1. First install uv (https://docs.astral.sh/uv/)
2. Install packages using `uv sync`
3. then activate the environment using `source .venv/bin/activate`

## Addtional Miner Documentation
Running the miner is as easy as `python launch_miner.py`. For more information, reference [the official miner docs](https://docs.macrocosmos.ai/subnets/subnet-9-pre-training/subnet-9-iota-mining-setup-guide)

You can also run many miners on the same machine by running `python launch_miner.py --num_miners NUMBER_OF_MINERS`

## Additional Validation Documentation
Running the validator `python launch_validator.py`. For more information, reference [the official validator docs](https://docs.macrocosmos.ai/subnets/subnet-9-pre-training/subnet-9-validating)

## Compute Requirements
We recommend: 
1. A100 80GB, or larger GPU 
2. Ubuntu
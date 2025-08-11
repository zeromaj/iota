# Gradient Validator Service

The Gradient Validator Service is responsible for validating gradients and weights in the distributed training system. It can run as a single instance or multiple instances for improved scalability and fault tolerance.

## Features

- Support for single or multiple validator instances
- Configurable host and port settings
- Automatic port allocation for multiple instances
- Comprehensive logging
- Prometheus metrics support (optional)

## Running the Validator Service

### Using the Shell Script

The easiest way to start validator instances is using the provided shell script:

```bash
# Start a single validator with default settings
./start_validators.sh

# Start multiple validators with custom settings
./start_validators.sh --host 0.0.0.0 --port 8081 --count 3

# Show help message
./start_validators.sh --help
```

### Using Python Directly

You can also run the validator service directly using Python:

```bash
# Start a single validator
python launch_validator.py

# Start multiple validators
python launch_validator.py --host 0.0.0.0 --port 8081 --count 3
```

## Configuration

The validator service can be configured through command line arguments or environment variables:

| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `--host`  | `VALIDATOR_HOST`    | `0.0.0.0` | Host to run the validator(s) on |
| `--port`  | `VALIDATOR_PORT`    | `8081`    | Starting port number |
| `--count` | `VALIDATOR_COUNT`   | `1`       | Number of validator instances |

Additional settings can be configured through environment variables:
- `PROMETHEUS`: Set to any value to enable Prometheus metrics

## Logging

Logs are written to both stderr and rotating log files:
- Console output: INFO level and above
- File logs: `validator_{timestamp}.log` with 100MB rotation

## Architecture

When running multiple validators:
- Each validator runs on a separate port (incrementing from the start port)
- Each validator has its own activation and weight storage
- The orchestrator automatically distributes validation work across available validators
- Validators can operate independently, providing fault tolerance

## Health Checks

Each validator instance exposes the following endpoints:
- `/gradient-validator/status`: Get validator status and tracked miner
- Health metrics at `/metrics` (if Prometheus is enabled)

## Error Handling

The service includes comprehensive error handling:
- Graceful shutdown on Ctrl+C
- Automatic retry logic for failed operations
- Detailed error logging
- Process exit codes for system integration

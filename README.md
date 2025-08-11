# IOTA

**I**ncentivized **O**rchestrated **T**raining **A**rchitecture

# Development

## Prerequisites

1. **Environment Setup**: First, create your environment file:

```sh
cp dotenv.example .env
# Edit .env with your actual values:
# HF_TOKEN="your_hugging_face_token"
# AWS_ACCESS_KEY_ID="your_aws_access_key"
# AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
```

2. **Dependencies**: Install UV package manager and sync dependencies:

```sh
# Run the dev environment setup script
./devsetup.sh
```

## Development Workflow

### Option 1: Using Task Runner (recommended)

The project includes a `Taskfile.yml` for common operations:

```sh
# Install Task runner if needed
# if you haven't run the devsetup.sh you can also use go to install:
# go install github.com/go-task/task/v3/cmd/task@latest

# Build and run Docker backend services
task up

# Start 3 miners
task start-miners

# To get a list of all commands:
task
```


### Option 2: Use Docker Compose

Start the full stack:

```sh
# Start all services
docker compose -f compose.dev.yaml up

# Or run in background
docker compose -f compose.dev.yaml up -d

# Build a compose.miners.yaml file
task generate-miners MINERS=0,1,2

# Start the miners
docker compose -f compose.miners.yaml up
```

This gives you:
- **PostgreSQL**: localhost:5432 (user: postgres, password: postgres, db: iota_orch_state)
- **Orchestrator**: localhost:8000 (API endpoints)
- **Scheduler**: scheduler stand-alone service

Useful commands:

```sh
# View logs
docker compose -f compose.dev.yaml logs -f

# Stop and remove services
docker compose -f compose.dev.yaml down

# Rebuild specific service
docker compose -f compose.dev.yaml build orchestrator
docker compose -f compose.dev.yaml up orchestrator --force-recreate
```

### Option 3: Hybrid Development (Database in Docker, Services Local)

For faster development with hot reloading:
Start only PostgreSQL:

```sh
docker compose up postgres -d
```

Run services locally:

```sh
# Terminal 1 - Orchestrator
uv run src/orchestrator/main.py

# Terminal 2 - Scheduler
uv run src/scheduler/main.py

# Terminal 3 - Miners
uv run src/miner/scripts/launch_multiple_miners.py --num-miners 3
```

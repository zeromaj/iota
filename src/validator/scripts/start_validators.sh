#!/bin/bash

set -e  # Exit on error

# Default values
DEFAULT_HOST="0.0.0.0"
DEFAULT_START_PORT=8083
DEFAULT_COUNT=1

# Function to validate numeric input
validate_number() {
    local value=$1
    local name=$2
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "Error: $name must be a positive number"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      START_PORT="$2"
      validate_number "$START_PORT" "port"
      shift 2
      ;;
    --count)
      COUNT="$2"
      validate_number "$COUNT" "count"
      if [ "$COUNT" -lt 1 ]; then
        echo "Error: count must be at least 1"
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--host HOST] [--port START_PORT] [--count VALIDATOR_COUNT]"
      echo
      echo "Options:"
      echo "  --host HOST          Host to run validators on (default: $DEFAULT_HOST)"
      echo "  --port START_PORT    Starting port number (default: $DEFAULT_START_PORT)"
      echo "  --count COUNT        Number of validator instances (default: $DEFAULT_COUNT)"
      echo "  -h, --help          Show this help message"
      exit 0
      ;;
    *)
      echo "Error: Unknown option: $1"
      echo "Use -h or --help for usage information"
      exit 1
      ;;
  esac
done

# Set defaults if not provided
HOST=${HOST:-$DEFAULT_HOST}
START_PORT=${START_PORT:-$DEFAULT_START_PORT}
COUNT=${COUNT:-$DEFAULT_COUNT}

# Validate port range
END_PORT=$((START_PORT + COUNT - 1))
if [ "$END_PORT" -gt 65535 ]; then
    echo "Error: Port range $START_PORT-$END_PORT exceeds maximum port number 65535"
    exit 1
fi

# Export environment variables
export VALIDATOR_HOST=$HOST
export VALIDATOR_PORT=$START_PORT
export VALIDATOR_COUNT=$COUNT

# Launch validator(s)
echo "Launching $COUNT validator instance(s) on $HOST starting at port $START_PORT"

# Run the validator with proper error handling
if ! python launch_validator.py --host "$HOST" --port "$START_PORT" --count "$COUNT"; then
    echo "Error: Failed to start validator service"
    exit 1
fi

# The script will exit here when the validators are stopped
echo "All validators have been stopped."

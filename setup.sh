#!/bin/bash

echo "IOTA Setup Script"
echo "=================="
echo ""

# Function to backup existing .env file
backup_env_file() {
    if [ -f ".env" ]; then
        timestamp=$(date +"%Y%m%d_%H%M%S")
        backup_file=".env.backup_${timestamp}"
        cp .env "$backup_file"
        echo "✓ Backed up existing .env to $backup_file"
    fi
}

# Ask user what they want to set up
echo "What would you like to set up?"
echo "1) Miner"
echo "2) Validator"
echo ""
read -p "Please enter your choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "Setting up Miner..."
        backup_env_file
        cp src/miner/miner-example.env .env
        echo "✓ Copied miner environment file to .env"
        echo ""
        echo "📝 NEXT STEPS:"
        echo "1. Edit .env and configure your settings"
        echo "2. Run: ./start_miner.sh"
        ;;
    2)
        echo ""
        echo "Setting up Validator..."
        backup_env_file
        cp src/validator/validator-example.env .env
        echo "✓ Copied validator environment file to .env"
        echo ""
        echo "📝 NEXT STEPS:"
        echo "1. Edit .env and configure your settings"
        echo "2. Run: ./start_validators.sh"
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again and choose 1 or 2."
        exit 1
        ;;
esac

echo ""
echo "Setup complete! 🎉"

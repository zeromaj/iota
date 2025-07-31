#!/bin/bash

echo "IOTA Setup Script"
echo "=================="
echo ""

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
        cp src/miner/miner-example.env src/miner/.env
        echo "✓ Copied miner environment file to src/miner/.env"
        echo ""
        echo "📝 NEXT STEPS:"
        echo "1. Edit src/miner/.env and configure your settings"
        echo "2. Run: ./start_miner.sh"
        ;;
    2)
        echo ""
        echo "Setting up Validator..."
        cp src/validator/validator-example.env src/validator/.env
        echo "✓ Copied validator environment file to src/validator/.env"
        echo ""
        echo "📝 NEXT STEPS:"
        echo "1. Edit src/validator/.env and configure your settings"
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
#!/bin/bash

# Quick Start Script for AI Learning Repo
# Run this after cloning to get set up quickly

set -e  # Exit on error

echo "=================================="
echo "LangChain RAG Toolkit - Setup"
echo "=================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✅ Found: $PYTHON_VERSION"
else
    echo "❌ Python 3 not found. Please install Python 3.9 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✅ pip upgraded"

# Install requirements
echo ""
echo "📚 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt > /dev/null 2>&1
echo "✅ Dependencies installed"

# Create .env file if it doesn't exist
echo ""
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file and add your OpenAI API key!"
    echo "   Get your key from: https://platform.openai.com/api-keys"
else
    echo "✅ .env file already exists"
fi

# Summary
echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your OPENAI_API_KEY"
echo "2. Activate the environment: source venv/bin/activate"
echo "3. Run your first script: python 01_hello_langchain.py"
echo ""
echo "📚 Recommended reading order:"
echo "   - SETUP_GUIDE.md"
echo "   - RECOMMENDATIONS.md"
echo "   - CHEATSHEET.md (quick reference)"
echo ""
echo "Need help? Check SETUP_GUIDE.md for troubleshooting."
echo ""
echo "Happy learning! 🚀"

#!/bin/bash

# Quick Start Script for AI Agent Research Assistant
# This script starts the Streamlit frontend

echo "🚀 Starting AI Agent Research Assistant..."
echo "================================================"
echo ""

# Check if in project directory
if [ ! -f "frontend/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "Please run: python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "Creating .env file template..."
    cat > .env << EOF
# API Keys
OPENROUTER_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here

# Model Configuration
MODEL=google/gemini-2.0-flash-exp:free
EOF
    echo "✅ Created .env file. Please add your API keys."
fi

echo ""
echo "✨ Starting Streamlit..."
echo "================================================"
echo "📱 App will open in your browser"
echo "🌐 URL: http://localhost:8501"
echo "⏹️  Press CTRL+C to stop"
echo "================================================"
echo ""

# Start Streamlit
streamlit run frontend/app.py





#!/bin/bash

# WorldAway - Exoplanet Seeker Setup Script
# this script sets up the development environment for both backend and frontend

set -e  # exit on error

echo "🌌 WorldAway - Exoplanet Seeker Setup"
echo "======================================"
echo ""

# colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.9+${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python $(python3 --version) found"

# check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 16+${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Node.js $(node --version) found"

# setup Backend
echo ""
echo "📦 Setting up Backend..."
echo "========================"

cd backend

# create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${YELLOW}⚠${NC} Virtual environment already exists"
fi

# activate virtual environment
source venv/bin/activate

# upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo -e "${GREEN}✓${NC} Python dependencies installed"

# create storage directories
echo "Creating storage directories..."
mkdir -p storage/uploads storage/jobs models
touch storage/uploads/.gitkeep storage/jobs/.gitkeep
echo -e "${GREEN}✓${NC} Storage directories created"

# create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env file created (please configure as needed)"
else
    echo -e "${YELLOW}⚠${NC} .env file already exists"
fi

# check for model file
if [ ! -f "models/exoplanet_model.pkl" ]; then
    echo -e "${YELLOW}⚠${NC} Model file not found at models/exoplanet_model.pkl"
    echo "   Please place your trained model in the models/ directory"
else
    echo -e "${GREEN}✓${NC} Model file found"
fi

cd ..

# setup Frontend
echo ""
echo "🎨 Setting up Frontend..."
echo "========================="

cd frontend

# install Node dependencies
echo "Installing Node.js dependencies..."
npm install
echo -e "${GREEN}✓${NC} Node.js dependencies installed"

# create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo -e "${GREEN}✓${NC} .env file created"
else
    echo -e "${YELLOW}⚠${NC} .env file already exists"
fi

cd ..

# generate sample data
echo ""
echo "📊 Generating sample data..."
echo "============================"

if [ ! -d "data" ]; then
    mkdir -p data
fi

cd backend
source venv/bin/activate
cd ..

python3 scripts/generate_sample_data.py
echo -e "${GREEN}✓${NC} Sample data generated"

# setup complete
echo ""
echo "======================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "======================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure environment variables:"
echo "   - backend/.env"
echo "   - frontend/.env"
echo ""
echo "2. Place your trained model at:"
echo "   - backend/models/exoplanet_model.pkl"
echo ""
echo "3. Start the backend:"
echo "   cd backend"
echo "   source venv/bin/activate"
echo "   uvicorn main:app --reload"
echo ""
echo "4. In a new terminal, start the frontend:"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "5. Access the application:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🌟 Happy exoplanet hunting!"
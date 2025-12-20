#!/bin/bash
# Build frontend and output to backend static folder

set -e

echo "🔨 Building GUIRA frontend..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Build the frontend
echo "🏗️  Running build..."
npm run build

echo "✅ Build complete!"
echo "📁 Output directory: ../orchestrator/api/static"
echo ""
echo "To test the build:"
echo "  1. Start the API: cd ../orchestrator/api && python app.py"
echo "  2. Open browser: http://localhost:8000"

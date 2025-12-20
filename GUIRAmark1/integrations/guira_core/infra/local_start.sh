#!/usr/bin/env bash

echo "🚀 Starting GUIRA local development infrastructure..."

# Navigate to local infrastructure directory
cd "$(dirname "$0")/local"

# Start local stack for dev
echo "🐳 Starting Docker containers..."
docker compose up -d

# Wait a moment for services to start
sleep 10

# Check services
echo "📊 Checking service status..."
docker compose ps

echo ""
echo "✅ Local infrastructure started successfully!"
echo ""
echo "🔗 Service endpoints:"
echo "   PostgreSQL: localhost:5432 (user: guira, password: guira_pass, db: guira)"
echo "   MinIO: http://localhost:9000 (user: minioadmin, password: minioadmin)"
echo "   Redis: localhost:6379"
echo ""
echo "🧪 To test connections:"
echo "   PostgreSQL: psql -h localhost -U guira -d guira -c '\\dt'"
echo "   MinIO: mc alias set local http://localhost:9000 minioadmin minioadmin"
echo "   Redis: redis-cli ping"
echo ""
echo "🛑 To stop: docker compose down"
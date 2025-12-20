#!/bin/bash
# Setup script for GUIRA PostGIS database
# Initializes the database schema and verifies setup

set -e

# Default connection parameters
DEFAULT_HOST="localhost"
DEFAULT_PORT="5432"
DEFAULT_USER="guira"
DEFAULT_DB="guira"
DEFAULT_PASSWORD="guira_pass"

# Parse command line arguments
HOST=${1:-$DEFAULT_HOST}
PORT=${2:-$DEFAULT_PORT}
USER=${3:-$DEFAULT_USER}
DB=${4:-$DEFAULT_DB}

# Export password for psql
export PGPASSWORD=${POSTGRES_PASSWORD:-$DEFAULT_PASSWORD}

# Connection string
CONN_STRING="postgresql://${USER}:${PGPASSWORD}@${HOST}:${PORT}/${DB}"

echo "=========================================="
echo "GUIRA Database Setup"
echo "=========================================="
echo "Host: $HOST"
echo "Port: $PORT"
echo "User: $USER"
echo "Database: $DB"
echo ""

# Test connection
echo "Testing database connection..."
if psql "$CONN_STRING" -c "SELECT version();" > /dev/null 2>&1; then
    echo "✓ Connection successful"
else
    echo "✗ Connection failed"
    echo "Please ensure PostgreSQL is running and credentials are correct"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="${SCRIPT_DIR}/init_postgis.sql"

# Check if SQL file exists
if [ ! -f "$SQL_FILE" ]; then
    echo "✗ SQL file not found: $SQL_FILE"
    exit 1
fi

# Execute SQL schema
echo ""
echo "Initializing database schema..."
if psql "$CONN_STRING" -f "$SQL_FILE" > /dev/null 2>&1; then
    echo "✓ Schema initialized successfully"
else
    echo "✗ Schema initialization failed"
    exit 1
fi

# Verify tables
echo ""
echo "Verifying tables..."
TABLES=$(psql "$CONN_STRING" -t -c "\dt" | grep -E "detections|forecasts|sessions|embeddings" | wc -l)

if [ "$TABLES" -ge 4 ]; then
    echo "✓ All required tables created"
else
    echo "✗ Some tables missing (found: $TABLES, expected: 4)"
    exit 1
fi

# Show table details
echo ""
echo "Database tables:"
psql "$CONN_STRING" -c "\dt"

echo ""
echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Connection string for applications:"
echo "$CONN_STRING"
echo ""
echo "To test manually:"
echo "  psql \"$CONN_STRING\" -c \"SELECT count(*) FROM detections;\""
echo ""

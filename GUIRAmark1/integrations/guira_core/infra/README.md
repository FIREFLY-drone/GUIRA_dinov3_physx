# GUIRA Infrastructure Bootstrap

This directory provides local development infrastructure and Azure cloud deployment options for the GUIRA (Fire Detection Drone System).

## Local Development Setup

### Prerequisites

- Docker and Docker Compose installed
- Make sure ports 5432, 6379, and 9000 are available

### Quick Start

```bash
# Start local stack for dev  
cd integrations/guira_core/infra/local
docker compose up -d

# Check services
docker compose ps
```

Or use the convenience script:

```bash
# From repository root
./integrations/guira_core/infra/local_start.sh
```

### Local Services

The local infrastructure provides:

- **PostGIS Database** (port 5432)
  - Database: `guira`
  - User: `guira`
  - Password: `guira_pass`
  
- **MinIO Object Storage** (port 9000)
  - Web Console: http://localhost:9000
  - Access Key: `minioadmin`
  - Secret Key: `minioadmin`
  
- **Redis Cache** (port 6379)
  - Standard Redis instance for caching and session storage

### Testing Local Setup

```bash
# Test PostgreSQL connection
psql -h localhost -U guira -d guira -c '\dt'

# Test MinIO connection (requires mc client)
mc alias set local http://localhost:9000 minioadmin minioadmin

# Test Redis connection  
redis-cli ping
```

## Cloud Infrastructure (Azure)

### Option A: Use Existing Azure Resources

Set the following environment variables in your `.env` file:

```bash
# Azure Storage
AZURE_STORAGE_CONNSTR="DefaultEndpointsProtocol=https;AccountName=..."

# Database
POSTGRES_CONN="postgresql://user:pass@host:5432/dbname"

# Object Storage (if using MinIO in cloud)
MINIO_ENDPOINT="https://your-minio-endpoint.com"
MINIO_ACCESS_KEY="your-access-key"
MINIO_SECRET_KEY="your-secret-key"

# Key Vault (optional)
AZURE_KEYVAULT_URL="https://your-keyvault.vault.azure.net/"
USE_KEYVAULT=true
```

### Option B: Create New Azure Resources

#### Prerequisites

- Azure CLI installed and logged in: `az login`
- Appropriate Azure subscription permissions

#### Deploy Infrastructure

```bash
# From repository root
./integrations/guira_core/infra/az_deploy.sh
```

This script will:
1. Create a resource group `guira-rg` in East US
2. Deploy the Bicep template creating:
   - Storage Account with unique name
   - Key Vault named `guira-kv`
   - Output the storage endpoint

#### Manual Deployment

```bash
# Create resource group
az group create -n guira-rg -l eastus

# Deploy Bicep template
az deployment group create \
  -g guira-rg \
  --template-file integrations/guira_core/infra/bicep/guira_infra.bicep \
  --parameters storageAccountName="guirastorage$(shuf -i 1000-9999 -n 1)"

# Check deployment status
az deployment group show -g guira-rg --name guira_infra
```

## Environment Variables Reference

### Required Variables

```bash
# DJI Cloud API (from DJI Developer Portal)
DJI_APP_ID="your_app_id_here"
DJI_APP_KEY="your_app_key_here"  
DJI_APP_LICENSE="your_app_license_here"

# Database Connection
POSTGRES_CONN="postgresql://guira:guira_pass@localhost:5432/guira"  # Local
# POSTGRES_CONN="postgresql://user:pass@azure-db:5432/guira"        # Azure

# Object Storage
MINIO_ENDPOINT="http://localhost:9000"    # Local
MINIO_ACCESS_KEY="minioadmin"             # Local
MINIO_SECRET_KEY="minioadmin"             # Local

# Azure Storage (when using Azure)
AZURE_STORAGE_ACCOUNT="your-storage-account"
AZURE_STORAGE_CONNSTR="DefaultEndpointsProtocol=https;..."

# Search & AI (Azure Cognitive Services)
AZURE_SEARCH_ENDPOINT="https://your-search.search.windows.net"
AZURE_SEARCH_KEY="your-search-key"
AZURE_OPENAI_ENDPOINT="https://your-openai.openai.azure.com"
AZURE_OPENAI_KEY="your-openai-key"

# Key Vault (optional, for secure secret storage)
AZURE_KEYVAULT_URL="https://your-keyvault.vault.azure.net/"
USE_KEYVAULT=false  # Set to true when using Key Vault
```

### Optional Variables

```bash
# Regional API endpoints
DJI_API_ENDPOINT="https://api-gateway.dji.com"          # Global (default)
# DJI_API_ENDPOINT="https://api-gateway-cn.dji.com"     # China

# Redis Configuration  
REDIS_URL="redis://localhost:6379"
REDIS_PASSWORD=""  # Leave empty for local dev

# Application Configuration
DEBUG=true
LOG_LEVEL="INFO"
```

## Key Vault Integration

When `USE_KEYVAULT=true`, the application will attempt to retrieve secrets from Azure Key Vault. The following secret names are expected:

- `dji-app-id`
- `dji-app-key` 
- `dji-app-license`
- `postgres-connection-string`
- `minio-access-key`
- `minio-secret-key`
- `azure-storage-connection-string`
- `azure-search-key`
- `azure-openai-key`

## Troubleshooting

### Local Development Issues

1. **Port conflicts**: Ensure ports 5432, 6379, and 9000 are not in use
2. **Docker issues**: Restart Docker service or run `docker system prune`
3. **Permission issues**: Ensure Docker daemon is running and accessible

### Azure Deployment Issues

1. **Authentication**: Run `az login` and verify correct subscription
2. **Permissions**: Ensure account has Contributor role on subscription
3. **Resource naming**: Storage account names must be globally unique
4. **Regional availability**: Some services may not be available in all regions

### Connection Testing

```bash
# Test local PostgreSQL
docker exec -it $(docker ps -qf "name=postgis") psql -U guira -d guira -c "SELECT version();"

# Test local MinIO
curl -I http://localhost:9000/minio/health/live

# Test local Redis
docker exec -it $(docker ps -qf "name=redis") redis-cli ping
```

## Security Considerations

- **Never commit secrets** to source control
- Use **environment variables** or **Key Vault** for sensitive data
- **Rotate credentials** regularly
- **Monitor access logs** for unauthorized usage
- **Use HTTPS** in production environments
- **Implement network security groups** in Azure deployments

## Data Stores & Ingestion (PH-10)

For comprehensive documentation on PostGIS schema, Kafka message bus, and detection ingestion:

📘 **See [README_DATA_STORES.md](README_DATA_STORES.md)**

This covers:
- PostGIS schema setup (`sql/init_postgis.sql`)
- Kafka topics configuration (`kafka/docker-compose.kafka.yml`)
- Detection ingestion consumer (`data/ingest/ingest_detection.py`)
- Environment variables for Kafka and message bus
- Production deployment to Azure

## Next Steps

1. Choose local or Azure infrastructure option
2. Configure environment variables in `.env` file  
3. Test connections to all services
4. Run integration tests to verify setup
5. Configure application-specific services (DJI controller, vision models, etc.)

For application-specific configuration, see:
- `docs/integrations/dji-cloud-api.md` - DJI Cloud API setup
- `docs/models/` - Model configuration and training
- `src/controller/` - Device integration setup
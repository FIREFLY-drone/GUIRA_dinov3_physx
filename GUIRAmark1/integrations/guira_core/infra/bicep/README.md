# Azure Bicep Templates for GUIRA Core

This directory contains Azure Bicep templates for provisioning GUIRA core infrastructure.

## Templates

- **main.bicep**: Main infrastructure template
- **compute.bicep**: GPU compute resources for DINOv3 and PhysX
- **storage.bicep**: Storage accounts for models and data
- **network.bicep**: Virtual network and security groups
- **monitoring.bicep**: Application Insights and Log Analytics

## Deployment

```bash
# Deploy main template
az deployment group create \
  --resource-group guira-core-rg \
  --template-file main.bicep \
  --parameters @parameters.json

# Verify deployment
az deployment group show \
  --resource-group guira-core-rg \
  --name main
```

## Parameters

See `parameters.json` for configuration options including:
- GPU VM sizes for compute
- Storage replication settings
- Network configuration
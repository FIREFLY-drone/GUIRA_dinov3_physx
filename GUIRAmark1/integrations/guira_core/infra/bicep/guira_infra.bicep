param location string = resourceGroup().location
param clusterName string = 'guira-aks'
param storageAccountName string

resource sa 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageAccountName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
}

resource kv 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: 'guira-kv'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    accessPolicies: []
  }
}

output storageEndpoint string = sa.properties.primaryEndpoints.blob
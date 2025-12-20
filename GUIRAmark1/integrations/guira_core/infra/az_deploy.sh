#!/usr/bin/env bash
RG=guira-rg
LOCATION=eastus
az group create -n $RG -l $LOCATION
az deployment group create -g $RG --template-file infra/bicep/guira_infra.bicep --parameters storageAccountName=guirastorage$RANDOM
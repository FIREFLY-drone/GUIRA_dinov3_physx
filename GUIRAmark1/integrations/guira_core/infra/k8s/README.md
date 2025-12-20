# Kubernetes Infrastructure for GUIRA Core

This directory contains Kubernetes manifests for deploying GUIRA core components.

## Components

- **embed-service**: DINOv3 embedding service deployment
- **physx-simulator**: PhysX fire simulation server
- **orchestrator**: Task scheduler and management
- **ingress**: Load balancing and routing

## Deployment

```bash
# Apply all manifests
kubectl apply -f .

# Check deployment status
kubectl get pods -n guira-core
```

## Scaling

```bash
# Scale embedding service
kubectl scale deployment embed-service --replicas=3

# Scale PhysX simulator
kubectl scale deployment physx-simulator --replicas=2
```
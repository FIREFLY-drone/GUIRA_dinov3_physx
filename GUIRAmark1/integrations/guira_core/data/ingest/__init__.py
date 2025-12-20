"""
GUIRA Core Data Ingestion Module.

Kafka consumers and data pipeline components for ingesting
detection events, embeddings, and simulation results.
"""

from .ingest_detection import DetectionIngestionConsumer, DetectionEvent

__all__ = ['DetectionIngestionConsumer', 'DetectionEvent']

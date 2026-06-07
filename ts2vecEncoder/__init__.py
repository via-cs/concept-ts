"""TS2Vec-style encoder for ProtoLLM.

Exposes the drop-in replacement class ``TS2VecEmbedding_Learning`` which
has the same constructor signature and forward contract as the existing
``MixerEmbedding_Learning`` / ``ConvEmbedding_Learning`` encoders. Wire
it into the pipeline by flipping the alias in ``model/embedding.py``.

For self-supervised pretraining (the recommended workflow):
    PYTHONPATH=. python -u ts2vecEncoder/train_ts2vec.py ...
"""

from .ts2vec_embedding import TS2VecEmbedding_Learning
from .tsencoder import TSEncoder
from .losses import (
    hierarchical_contrastive_loss,
    instance_contrastive_loss,
    temporal_contrastive_loss,
)

__all__ = [
    "TS2VecEmbedding_Learning",
    "TSEncoder",
    "hierarchical_contrastive_loss",
    "instance_contrastive_loss",
    "temporal_contrastive_loss",
]

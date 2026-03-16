#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

import importlib
from typing import TYPE_CHECKING, Any

from .base_vector_store import BaseVectorStore
from .langchain_vector_store_adapter import LangChainVectorStoreAdapter
from .vector_store import VectorStore
from .vector_store_connector import VectorStoreConnector

if TYPE_CHECKING:
    from .adapters.chroma_adapter import ChromaVectorStore
    from .adapters.db2_adapter import DB2VectorStore
    from .adapters.es_adapter import ElasticsearchVectorStore
    from .adapters.es_utils import HybridStrategyElasticsearch, RetrievalOptions
    from .adapters.milvus_adapter import MilvusVectorStore
    from .adapters.milvus_utils import (
        MilvusBM25BuiltinFunction,
        MilvusSpladeEmbeddingFunction,
    )

__all__ = [
    "BaseVectorStore",
    "LangChainVectorStoreAdapter",
    "VectorStoreConnector",
    "VectorStore",
    "MilvusVectorStore",
    "MilvusBM25BuiltinFunction",
    "MilvusSpladeEmbeddingFunction",
    "ElasticsearchVectorStore",
    "RetrievalOptions",
    "HybridStrategyElasticsearch",
    "ChromaVectorStore",
    "DB2VectorStore",
]

_module_lookup = {
    "ElasticsearchVectorStore": ".adapters.es_adapter",
    "RetrievalOptions": ".adapters.es_utils",
    "HybridStrategyElasticsearch": ".adapters.es_utils",
    "MilvusVectorStore": ".adapters.milvus_adapter",
    "MilvusBM25BuiltinFunction": ".adapters.milvus_utils",
    "MilvusSpladeEmbeddingFunction": ".adapters.milvus_utils",
    "ChromaVectorStore": ".adapters.chroma_adapter",
    "DB2VectorStore": ".adapters.db2_adapter",
}
_root_module = "ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores"


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    if name in _module_lookup:
        module = importlib.import_module(_root_module + _module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")

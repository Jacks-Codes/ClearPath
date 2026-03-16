#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import hashlib
import logging
from collections import defaultdict
from typing import Any, Generic, TypeVar, cast

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings
from ibm_watsonx_ai.foundation_models.extensions.rag.utils.utils import verbose_search
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.base_vector_store import (
    BaseVectorStore,
)
from ibm_watsonx_ai.utils.utils import is_lib_installed
from ibm_watsonx_ai.wml_client_error import MissingExtension, MissingMetadata
from ibm_watsonx_ai.wml_resource import WMLResource

if not is_lib_installed(ext := "langchain-core"):
    raise MissingExtension(ext, extra_info="rag")

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from langchain_core.vectorstores import (
    VectorStoreRetriever as LangChainVectorStoreRetriever,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=LangChainVectorStore)

DEFAULT_DOCUMENT_NAME_FIELD = "document_id"
DEFAULT_CHUNK_SEQUENCE_NUMBER_FIELD = "sequence_number"


def merge_window_into_a_document(window: list[Document]) -> Document:
    """
    Merges a list of chunks into a single document.
    If consecutive chunks have intersecting merged_text, the merged_text is merged to avoid duplications.

    :param window: ordered list of documents for merging
    :type window: list[langchain_core.documents.Document]

    :return: document that contains the merged merged_text of the window documents
    :rtype: langchain_core.documents.Document
    """

    def merge_metadata(metadatas: list[dict]) -> dict:
        """
        Merges a list of dictionaries (metadata) into one metadata
        The keys remain the same but the values are changed into lists of values from all metadata

        :param metadatas: list of metadata dictionaries to be merged
        :type metadatas: list[dict]

        :return: single merged dictionary (metadata) with unique values in lists for each key
        :rtype: Tuple[str, bool]
        """
        if len(metadatas) == 1:
            return metadatas[0]

        merged_metadata = defaultdict(set)
        for metadata in metadatas:
            for key, value in metadata.items():
                if isinstance(value, list):
                    merged_metadata[key].update(value)
                else:
                    merged_metadata[key].add(value)

        result = {}
        for key, value_set in merged_metadata.items():
            value_list = sorted(value_set)
            if len(value_list) == 1:
                result[key] = value_list[0]
            else:
                result[key] = value_list
        return result

    def get_str2_without_intersecting_text(str1: str, str2: str) -> tuple[str, bool]:
        """
        Finds the intersecting merged_text between the suffix of str1 and the prefix of str2.

        :param str1: the first string
        :type str1: str

        :param str2: the second string
        :type str2: str

        :return: a tuple of:
            1. str2 without its intersection to str1
            2. whether there was an intersection or not
        :rtype: Tuple[str, bool]
        """
        # Start checking from the longest possible overlap to the shortest
        for i in range(min(len(str1), len(str2)), 0, -1):
            if str1[-i:] == str2[:i]:
                return str2[i:], True
        return str2, False

    def merge_texts(texts: list[str]) -> str:
        """
        Merges a list of texts into a single text string.
        If consecutive texts have intersecting parts, the text is merged to avoid duplications.

        :param texts: ordered list of text strings to be merged
        :type texts: List[str]

        :return: single string that contains the merged text
        :rtype: str
        """
        merged_text = ""
        for text in texts:
            text_to_add, has_intersection = get_str2_without_intersecting_text(
                merged_text, text
            )
            if merged_text and not has_intersection:
                merged_text += " "  # Add a space between non-overlapping texts (chunks)
            merged_text += text_to_add
        return merged_text

    texts = [doc.page_content for doc in window]
    merged_text = merge_texts(texts)

    metadata = [doc.metadata for doc in window]
    merged_metadata = merge_metadata(metadata)

    return Document(page_content=merged_text, metadata=merged_metadata)


class LangChainVectorStoreAdapter(Generic[T], BaseVectorStore):
    """
    Adapter for LangChain ``VectorStore`` base class.

    :param vector_store: concrete LangChain vector store object
    :type vector_store: langchain_core.vectorstore.VectorStore

    :param document_name_field: mapping field for document name, defaults to `document_id`
    :type document_name_field: str, optional

    :param chunk_sequence_number_field: mapping field for chunk sequence number, defaults to `sequence_number`
    :type chunk_sequence_number_field: str, optional
    """

    def __init__(
        self,
        vector_store: T,
        document_name_field: str = DEFAULT_DOCUMENT_NAME_FIELD,
        chunk_sequence_number_field: str = DEFAULT_CHUNK_SEQUENCE_NUMBER_FIELD,
    ) -> None:
        super().__init__()
        self._langchain_vector_store: T = vector_store
        self._document_name_field = document_name_field
        self._chunk_sequence_number_field = chunk_sequence_number_field

    def get_client(self) -> T:
        return self._langchain_vector_store

    def _set_embeddings(self, embedding_fn: BaseEmbeddings) -> None:
        if hasattr(self._langchain_vector_store, "embedding"):
            self._langchain_vector_store.embedding = embedding_fn
        elif hasattr(self._langchain_vector_store, "_embedding"):
            self._langchain_vector_store._embedding = embedding_fn
        elif hasattr(self._langchain_vector_store, "_embedding_function"):
            self._langchain_vector_store._embedding_function = embedding_fn
        elif hasattr(self._langchain_vector_store, "embedding_function"):
            self._langchain_vector_store.embedding_function = embedding_fn
        else:
            raise AttributeError(
                "Could not set an embedding function for this vector store."
            )

    def add_documents(
        self, content: list[str] | list[dict] | list[Document], **kwargs: Any
    ) -> list[str]:
        """
        Embed documents and add to the vectorstore.

        :param content: Documents to add to the vectorstore.
        :type content: list[str] | list[dict] | list[langchain_core.documents.Document]

        :return: List of IDs of the added texts.
        :rtype: list[str]
        """
        ids, docs = self._process_documents(content)
        return self._langchain_vector_store.add_documents(docs, ids=ids, **kwargs)

    async def add_documents_async(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        """
        Embed documents and add to the vectorstore in asynchronous manner.

        :param content: Documents to add to the vectorstore.
        :type content: list[str] | list[dict] | list[langchain_core.documents.Document]

        :return: List of IDs of the added texts.
        :rtype: list[str]
        """
        ids, docs = self._process_documents(content)
        return await self._langchain_vector_store.aadd_documents(
            docs, ids=ids, **kwargs
        )

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[Document] | list[tuple[Document, float]]:
        """Searches for documents most similar to the query.

        The method is designed as a wrapper for respective LangChain VectorStores' similarity search methods.
        Therefore, additional search parameters passed in ``kwargs`` should be consistent with those methods,
        and can be found in the LangChain documentation.

        :param query: text query
        :type query: str

        :param k: number of documents to retrieve
        :type k: int

        :param include_scores: whether similarity scores of found documents should be returned, defaults to False
        :type include_scores: bool

        :param verbose: whether to display a table with the found documents, defaults to False
        :type verbose: bool

        :return: list of found documents
        :rtype: list
        """
        result: list[Document] | list[tuple[Document, float]]
        if include_scores:
            result = self._langchain_vector_store.similarity_search_with_score(
                query, k=k, **kwargs
            )
        else:
            result = self._langchain_vector_store.similarity_search(
                query, k=k, **kwargs
            )

        if verbose:
            verbose_search(query, result)
        return result

    def window_search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list:
        """Searches for documents most similar to the query and extend a document (a chunk) to its adjacent chunks (if they exist) from the same origin document.

        The method is designed as a wrapper for respective LangChain VectorStores' similarity search methods.
        Therefore, additional search parameters passed in ``kwargs`` should be consistent with those methods,
        and can be found in the LangChain documentation.

        :param query: text query
        :type query: str

        :param k: number of documents to retrieve
        :type k: int

        :param include_scores: whether similarity scores of found documents should be returned, defaults to False
        :type include_scores: bool

        :param verbose: whether to display a table with the found documents, defaults to False
        :type verbose: bool

        :param window_size: number of chunks
        :type window_size: int, optional

        :return: list of found documents
        :rtype: list
        """
        documents = self.search(query, k, include_scores, verbose, **kwargs)
        if window_size <= 0:
            return documents

        if not include_scores:
            documents = cast(list[Document], documents)
            return [
                self._window_extend_and_merge(document, window_size)
                for document in documents
            ]
        else:
            documents_and_scores = cast(list[tuple[Document, float]], documents)
            documents = [t[0] for t in documents_and_scores]
            scores = [t[1] for t in documents_and_scores]
            extended_documents = [
                self._window_extend_and_merge(document, window_size)
                for document in documents
            ]
            return list(zip(extended_documents, scores))

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """Delete by vector ID or other criteria. Sor more details see LangChain documentation
        https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore
        """
        self._langchain_vector_store.delete(ids, **kwargs)

    def clear(self) -> None:
        raise NotImplementedError(
            "Use concrete wrapper if you need to use this functionality."
        )

    def count(self) -> int:
        raise NotImplementedError(
            "Use concrete wrapper if you need to use this functionality."
        )

    def as_langchain_retriever(self, **kwargs: Any) -> Any:
        """Return Langchain VectorStoreRetriever initialized from this VectorStore."""
        return LangChainVectorStoreRetriever(
            vectorstore=self._langchain_vector_store, **kwargs
        )

    def _process_documents(
        self, content: list[str] | list[dict] | list
    ) -> tuple[list[str], list[Document]]:
        """Processes arbitrary list of data to produce two lists: one with unique IDs, and one with LangChain documents.

        Handles duplicate documents.

        :param content: arbitrary data
        :type content: list[str] | list[dict] | list

        :return: lists with IDs and docs
        :rtype: tuple[list[str], list[langchain_core.documents.Document]
        """
        WMLResource._validate_type(content, "content", list)
        docs = self._as_langchain_documents(content)
        if docs:
            # Take only unique ID document. Get two lists, one with ids, one with documents
            # For some documents, not all chars can be encoded properly.
            # In such cases, replace invalid chars by question marks, i.e. setting errors="replace"
            return tuple(
                map(  # type: ignore[return-value]
                    list,
                    zip(
                        *{
                            hashlib.sha256(
                                str(doc).encode(errors="replace")
                            ).hexdigest(): doc
                            for doc in docs
                        }.items()
                    ),
                )
            )
        else:
            return [], []

    def _as_langchain_documents(
        self, content: list[str] | list[dict] | list
    ) -> list[Document]:
        """Creates a LangChain ``Document`` list from a list of potentially unstructured data.

        :param content: list of unstructured data to be parsed
        :type content: list[str] | list[dict] | list

        :raises AttributeError: raised when data does not fit the required schema
        :return: list of LangChain Documents
        :rtype: list[langchain_core.documents.Document]
        """
        result = []
        for doc in content:
            if isinstance(doc, str):
                result.append(Document(page_content=doc))
            elif isinstance(doc, dict):
                content_str: str | None = doc.get("content", None)
                metadata = doc.get("metadata", {})

                if content_str:
                    if isinstance(metadata, dict):
                        result.append(
                            Document(page_content=content_str, metadata=metadata)
                        )
                    else:
                        logger.warning(
                            f"Document: {doc} is incorrect. Metadata needs to be given with 'metadata' attribute and it needs to be a serializable dict. Skipping."
                        )
                        continue
                else:
                    logger.warning(
                        f"Document: {doc} is incorrect. Field 'content' is required"
                    )
                    continue
            else:
                try:
                    result.append(
                        Document(page_content=doc.page_content, metadata=doc.metadata)
                    )
                except AttributeError:
                    logger.warning(
                        f"Document: {doc} is not a dict, nor string, nor LangChain Document-like object. Skipping."
                    )

        return result

    def _window_extend_and_merge(
        self, document: Document, window_size: int
    ) -> Document:
        """
        Extends a document (a chunk) to its adjacent chunks (if they exist) from the same origin document.
        Then merges the adjacent chunks into one chunk while keeping their order,
        and merges intersecting text between them (if it exists).
        This requires chunks to have "document_id" and "sequence_number" in their metadata.

        :param document: document (chunk) to be extended to its window and merged
        :type document: Document

        :param window_size: number of adjacent chunks to retrieve before and after the center, according to the sequence_number
        :type window_size: int

        :return: merged window
        :rtype: Document
        """
        if self._document_name_field not in document.metadata:
            raise MissingMetadata(
                f'document must have "{self._document_name_field}" in its metadata'
            )
        if self._chunk_sequence_number_field not in document.metadata:
            raise MissingMetadata(
                f'document must have "{self._chunk_sequence_number_field}" in its metadata'
            )
        doc_id = document.metadata[self._document_name_field]
        seq_num = document.metadata[self._chunk_sequence_number_field]
        seq_nums_window = [seq_num + i for i in range(-window_size, window_size + 1, 1)]

        vs_type = self._langchain_vector_store.__class__.__name__

        match vs_type:
            case "Milvus" | "Chroma" | "ElasticsearchStore" | "DB2VS":
                window_documents = self._get_window_documents(doc_id, seq_nums_window)
            case _:
                raise TypeError(
                    f"Currently we only support Milvus, Chroma, Elasticsearch and DB2VS. "
                    f"Received {type(self._langchain_vector_store)}."
                )

        window_documents.sort(
            key=lambda x: x.metadata[self._chunk_sequence_number_field]
        )
        return merge_window_into_a_document(window_documents)

    def _get_window_documents(
        self, doc_id: str, seq_nums_window: list[int]
    ) -> list[Document]:
        """
        Receives a document ID and a list of chunks' sequence_numbers,
        and searches the vector store according to the metadata.

        :param doc_id: ID of document
        :type doc_id: str

        :param seq_nums_window: list of sequence numbers
        :type seq_nums_window: list[int]

        :return: list of documents from that document with these sequence_numbers
        :rtype: list[Document]
        """
        raise NotImplementedError()

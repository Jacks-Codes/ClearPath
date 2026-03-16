#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC

from ibm_watsonx_ai.helpers.connections.connections import (
    DatabaseLocation,
    DataConnection,
)
from ibm_watsonx_ai.utils.autoai.enums import DataConnectionTypes


class BaseKnowledgeBase(ABC):
    """Base class for knowledge base objects in AutoAI RAG.

    :param name: name of the knowledge base
    :type name: str

    :param description: description of the knowledge base
    :type description: str

    :param knowledge_base_type: type of the knowledge base
    :type knowledge_base_type: str

    :param connection: connection to the knowledge base, must be a connection asset
    :type connection: DataConnection

    :raises: ValueError if connection is not a connection asset
    """

    def __init__(
        self,
        name: str,
        description: str,
        knowledge_base_type: str,
        connection: DataConnection,
    ) -> None:
        self.name = name
        self.description = description
        self.knowledge_base_type = knowledge_base_type

        if connection.type != DataConnectionTypes.CA:
            raise ValueError(
                f"Connection must be a connection asset, instead got '{connection.type}'"
            )

        self.connection = connection

    def to_dict(self) -> dict:
        """Convert knowledge base instance to dict.

        :return: dict representing knowledge base
        :rtype: dict

        **Example:**

        .. code-block:: python

            knowledge_base.to_dict()
        """

        return {
            "name": self.name,
            "description": self.description,
            "type": self.knowledge_base_type,
            "reference": self.connection.to_dict(),
        }


class DatabaseKnowledgeBase(BaseKnowledgeBase):
    """Reference to SQL DB knowledge.

    :param name: name of the knowledge base
    :type name: str

    :param description: description of the knowledge base
    :type description: str

    :param connection: connection to the knowledge base, must be a connection asset
    :type connection: DataConnection

    :param settings: settings of the database knowledge base
    :type settings: dict, optional

    :raises: ValueError if connection is not a connection asset
    :raises: TypeError if connection location is not a database location

    **Example:**

    .. code-block:: python

        database_knowledge_base = DatabaseKnowledgeBase(
            name="customers",
            description="Database containing information about customers",
            connection=DataConnection(
                connection_asset_id="<connection_id>",
                location=DatabaseLocation(schema_name="customers"),
            ),
            settings={"dialect": "mysql"},
        )

    """

    def __init__(
        self,
        name: str,
        description: str,
        connection: DataConnection,
        settings: dict | None = None,
    ) -> None:
        super().__init__(name, description, "database", connection)

        if not isinstance(connection.location, DatabaseLocation):
            raise TypeError("Connection location should be a database location")

        self.settings = settings

    def to_dict(self) -> dict:
        results = super().to_dict()

        if self.settings is not None:
            results["settings"] = self.settings

        return results


class VectorStoreKnowledgeBase(BaseKnowledgeBase):
    """Reference to vector store knowledge.

    :param name: name of the vector store knowledge base
    :type name: str

    :param description: description of the vector store knowledge base
    :type description: str

    :param connection: connection to the vector store knowledge base, must be a connection asset
    :type connection: DataConnection

    :param settings: settings of the vector store knowledge base
    :type settings: dict

    :raises: ValueError if connection is not a connection asset

    **Example:**

    .. code-block:: python

        vector_store_knowledge_base = VectorStoreKnowledgeBase(
            name="customers",
            description="Database containing information about customers",
            connection=DataConnection(connection_asset_id="<connection_id>"),
            settings={
                "index_name": "autoai_rag_id_pipeline_id_index",
                "fields_mapping": [
                    {
                        "role": KnowledgeBaseFieldRole.DENSE_VECTOR_EMBEDDINGS,
                        "field_name": "vector_embeddings",
                    }
                ],
                "embeddings": {"model_id": "ibm/slate-125m-english-rtrvr"},
                "hybrid_ranker": {"sparse_vectors": {"model_id": "BM25"}},
            },
        )

    """

    def __init__(
        self, name: str, description: str, connection: DataConnection, settings: dict
    ) -> None:
        super().__init__(name, description, "vector_store", connection)

        self.settings = settings

    def to_dict(self) -> dict:
        return super().to_dict() | {"settings": self.settings}

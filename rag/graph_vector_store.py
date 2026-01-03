"""
A Neo4j-based vector store for graph RAG.
"""
import os
from langchain_community.vectorstores import Neo4jVector
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from typing import List, Any, Iterable
from daniel_agent.utils.model import embedding_model

# Default values for Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


class Neo4jVectorStore(VectorStore):
    """
    A Neo4j-based vector store for graph RAG.
    """

    def __init__(
        self,
        embedding,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        **kwargs: Any,
    ):
        """
        Initialize the Neo4j vector store.

        Args:
            embedding: The embedding function to use.
            url: The URL of the Neo4j instance.
            username: The username for the Neo4j instance.
            password: The password for the Neo4j instance.
        """
        self.embedding = embedding
        self._vector_store = Neo4jVector(
            url=url,
            username=username,
            password=password,
            embedding=embedding,
            **kwargs,
        )

    @property
    def embeddings(self):
        return self.embedding

    def add_documents(
        self,
        documents: List[Document],
        ids: List[str] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add documents to the vector store.
        """
        return self._vector_store.add_documents(documents, **kwargs)

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: List[str] | None = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add documents to the vector store.
        """
        return await self._vector_store.aadd_documents(documents, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Perform a similarity search.
        """
        return self._vector_store.similarity_search(query, k=k, **kwargs)

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """
        Asynchronously perform a similarity search.
        """
        return await self._vector_store.asimilarity_search(query, k=k, **kwargs)

    def delete(self, ids: List[str] | None = None, **kwargs: Any) -> None:
        """Delete by vector IDs."""
        self._vector_store.delete(ids, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Any,
        metadatas: List[dict] | None = None,
        **kwargs: Any,
    ) -> "Neo4jVectorStore":
        """Create a Neo4jVectorStore from a list of texts."""
        store = cls(embedding=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

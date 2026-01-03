from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.graph_vector_store import Neo4jVectorStore
from daniel_agent.utils.model import embedding_model

loader = UnstructuredMarkdownLoader(
    "./data/output_text.md",
    mode="single",
    strategy="fast",
)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)

all_splits = text_splitter.split_documents(docs)

print(f"Split Daniel's data into {len(all_splits)} sub-documents.")

vector_store = Neo4jVectorStore(embedding=embedding_model)

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])

from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
from llama_index import SimpleDirectoryReader, StorageContext, VectorStoreIndex, ComposableGraph
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.vector_stores.neo4j import Neo4jVectorStore
import os

# 1. Load your documents (e.g., PDFs)
documents = SimpleDirectoryReader("data").load_data()

# 2. Create storage contexts for each retrieval backend

# Pinecone: vector similarity search
pinecone_store = PineconeVectorStore(pinecone_index=... )
pinecone_ctx = StorageContext.from_defaults(vector_store=pinecone_store)

# Elasticsearch: full-text keyword search
es_store = ElasticsearchStore(es_url="http://localhost:9200", index_name="my_es_index")
es_ctx = StorageContext.from_defaults(vector_store=es_store)

# Postgres (pgvector): structured + vector search
pg_store = PGVectorStore(connection_string="postgresql://user:pass@localhost/db", table_name="my_pg_table")
pg_ctx = StorageContext.from_defaults(vector_store=pg_store)

# Neo4j: knowledge graph search with optional hybrid capability
neo4j_store = Neo4jVectorStore(username="neo4j", password="pass", url="bolt://localhost:7687", embed_dim=1536, hybrid_search=True)
neo4j_ctx = StorageContext.from_defaults(vector_store=neo4j_store)

# 3. Build individual indexes
pinecone_idx = VectorStoreIndex.from_documents(documents, storage_context=pinecone_ctx)
es_idx       = VectorStoreIndex.from_documents(documents, storage_context=es_ctx)
pg_idx       = VectorStoreIndex.from_documents(documents, storage_context=pg_ctx)
neo4j_idx    = VectorStoreIndex.from_documents(documents, storage_context=neo4j_ctx)

# 4. Compose these into a single hybrid index
hybrid_index = ComposableGraph.from_indices(
    {
        "semantic": pinecone_idx,
        "keyword": es_idx,
        "structured": pg_idx,
        "graph": neo4j_idx,
    },
    include_root=True,
)

# 5. Create a unified query engine
query_engine = hybrid_index.as_query_engine()

# 6. Run a single query against all these systems
response = query_engine.query("Find papers discussing self-attention published after 2017 and their key contributions.")
print(response)

from sentence_transformers import SentenceTransformer
from chromadb.utils import embedding_functions
import chromadb


sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='moka-ai/m3e-base')
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(
    name="TTTTT", 
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"} 
)

collection.add(
    documents=["人", "地球"],
    metadatas=[{"source": "my_source"}, {"source": "my_source"}],
    ids=["id1", "id2"]
)



results = collection.query(
    query_texts=["月亮"],
    n_results=1
)

print(results)
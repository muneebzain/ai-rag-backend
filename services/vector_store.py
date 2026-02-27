import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection("rag_collection")

def add_chunks(doc_id, chunks, vectors):
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=vectors,
        ids=ids
    )

    print("Total vectors in DB:", collection.count())


def search(query_vector, top_k=3):
    return collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
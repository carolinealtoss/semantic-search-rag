from services.embedding_service import MODEL


def search_documents(query, index, documents, top_k=3):

    # Transforma a consulta em embedding
    query_embedding = MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    top_k = min(top_k, index.ntotal)

    # Pesquisa os documentos mais semelhantes
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        document = documents[idx]

        results.append({
            "id": document["id"],
            "filename": document["filename"],
            "chunk_index": document["chunk_index"],
            "content": document["content"],
            "similarity": float(score)
        })

    return results
import faiss
import json

from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

DOCUMENTS_PATH = Path("documents")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL = SentenceTransformer(MODEL_NAME)

STORAGE_PATH = Path("storage")
INDEX_PATH = STORAGE_PATH / "documents.index"
METADATA_PATH = STORAGE_PATH / "metadata.json"


def load_documents():
    documents = []

    for file_path in DOCUMENTS_PATH.glob("*.txt"):

        content = file_path.read_text(
            encoding="utf-8"
        )

        documents.append({
            "id": file_path.stem,
            "filename": file_path.name,
            "content": content
        })

        print(documents)

    return documents


def generate_embeddings(documents):

    texts = [
        document["content"]
        for document in documents
    ]

    embeddings = MODEL.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")


def create_vector_store(documents, embeddings):

    STORAGE_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    dimension = embeddings.shape[1]

    # Cria o índice FAISS
    index = faiss.IndexFlatIP(dimension)

    # Adiciona os vetores ao índice
    index.add(embeddings)

    # Persiste o índice no disco
    faiss.write_index(
        index,
        str(INDEX_PATH)
    )

    # Salva os documentos e seus metadados
    with open(METADATA_PATH, "w", encoding="utf-8") as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

    return index


def search_documents(query, index, documents, top_k=3):

    # Transforma a consulta em embedding
    query_embedding = MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

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
            "content": document["content"],
            "similarity": float(score)
        })

    return results


def load_vector_store():

    index = faiss.read_index(
        str(INDEX_PATH)
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    return index, documents


def main():

    print("Inicializando busca semântica...")

    documents = load_documents()

    print(f"Documentos encontrados: {len(documents)}")

    embeddings = generate_embeddings(documents)

    print("Embeddings gerados com sucesso!")

    index = create_vector_store(
        documents,
        embeddings
    )

    print("Vector Store criada com sucesso!")

    while True:

        query = input("\nDigite sua pergunta (ou 'sair'): ")

        if query.lower() == "sair":
            break

        results = search_documents(
            query,
            index,
            documents,
            top_k=3
        )

        print("-" * 60)
        print("\nDocumentos encontrados:\n")

        for result in results:

            print(f"Documento: {result['filename']}")

            print(
                f"Similaridade: {result['similarity']:.4f}"
            )

            print(f"Conteúdo: {result['content']}")

            print("-" * 60)


if __name__ == "__main__":
    main()
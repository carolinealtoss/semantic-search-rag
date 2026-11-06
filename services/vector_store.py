import json
from pathlib import Path

import faiss

STORAGE_PATH = Path("storage")
INDEX_PATH = STORAGE_PATH / "documents.index"
METADATA_PATH = STORAGE_PATH / "metadata.json"


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
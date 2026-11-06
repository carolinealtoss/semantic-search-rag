from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DOCUMENTS_PATH = Path("documents")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL = SentenceTransformer(MODEL_NAME)


def load_documents():

    documents = []

    for file_path in DOCUMENTS_PATH.glob("*.txt"):

        content = file_path.read_text(
            encoding="utf-8"
        )

        chunks = split_text(
            content,
            chunk_size=500,
            overlap=100
        )

        for index, chunk in enumerate(chunks):
            documents.append({
                "id": f"{file_path.stem}_{index}",
                "filename": file_path.name,
                "chunk_index": index,
                "content": chunk
            })

    return documents


def generate_embeddings(documents):

    if not documents:
        raise ValueError(
            "Nenhum documento encontrado para indexação."
        )

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


def split_text(text, chunk_size=500, overlap=100):

    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap deve ser maior ou igual a zero "
            "e menor que chunk_size"
        )

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Parágrafos maiores que o limite
        # precisam ser divididos separadamente.
        if len(paragraph) > chunk_size:

            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            start = 0

            while start < len(paragraph):

                end = start + chunk_size

                chunks.append(paragraph[start:end])

                start += chunk_size - overlap

            continue

        candidate = (
            current_chunk + "\n\n" + paragraph
            if current_chunk
            else paragraph
        )

        if len(candidate) <= chunk_size:

            current_chunk = candidate

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # Recupera o final do chunk anterior
            # para manter a continuidade do contexto.
            previous_context = current_chunk[-overlap:] if overlap else ""

            current_chunk = (
                previous_context + "\n\n" + paragraph
                if previous_context
                else paragraph
            )

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
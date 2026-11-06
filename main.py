from services.embedding_service import (
    generate_embeddings,
    load_documents,
)
from services.search_service import search_documents
from services.vector_store import (
    INDEX_PATH,
    METADATA_PATH,
    create_vector_store,
    load_vector_store,
)


def main():

    print("Inicializando busca semântica...")

    if INDEX_PATH.exists() and METADATA_PATH.exists():

        print("Carregando Vector Store existente...")

        index, documents = load_vector_store()

    else:

        print("Criando nova Vector Store...")

        documents = load_documents()

        embeddings = generate_embeddings(documents)

        index = create_vector_store(
            documents,
            embeddings
        )

    document_count = len({
        document["filename"]
        for document in documents
    })
    print(f"Documentos indexados: {document_count}")
    print(f"Trechos indexados: {len(documents)}")

    print("Sistema pronto!")

    while True:

        query = input("\nDigite sua pergunta (ou 'sair'): ")

        if query.lower() == "sair":
            break

        if not query.strip():
            continue

        results = search_documents(
            query,
            index,
            documents,
            top_k=3
        )

        for result in results:

            print(f"\nDocumento: {result['filename']}")

            print(f"Similaridade: {result['similarity']:.4f}")

            print(f"Conteúdo: {result['content']}")

            print("-" * 60)


if __name__ == "__main__":
    main()
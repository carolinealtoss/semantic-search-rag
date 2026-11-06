# semantic-search-rag

Aplicação de linha de comando para busca semântica em documentos de texto.
Ela encontra trechos relacionados ao significado de uma pergunta, mesmo quando
os termos usados na consulta não são idênticos aos do documento.

## Como funciona

1. Lê os arquivos `.txt` em UTF-8 diretamente da pasta `documents`.
2. Divide os textos em trechos, com tamanho configurado de 500 caracteres e
   sobreposição de 100 caracteres.
3. Gera embeddings usando o modelo multilíngue
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` e os indexa
   com FAISS.
4. Recebe uma pergunta no terminal e exibe até três trechos semelhantes,
   indicando o arquivo de origem, a pontuação de similaridade e o conteúdo.

A aplicação retorna os trechos encontrados; ela não gera uma resposta usando
um modelo de linguagem. A quantidade de trechos indexados pode ser maior que
a quantidade de documentos, pois cada arquivo pode ser dividido em vários
trechos.

## Como executar

É necessário ter Python e `pip` instalados. Os comandos abaixo são para o
PowerShell no Windows e devem ser executados na pasta raiz do projeto:

1. Crie um ambiente virtual e instale as dependências:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Coloque os documentos `.txt` em UTF-8 na pasta `documents`. Para criar um
   novo índice, é necessário que pelo menos um documento contenha texto.

3. Inicie a aplicação:

   ```powershell
   .\.venv\Scripts\python.exe main.py
   ```

4. Digite uma pergunta e pressione Enter. Para encerrar, digite `sair`.

Na primeira execução, o modelo pode precisar ser baixado, o que requer
conexão com a internet. O índice FAISS e os metadados são salvos em
`storage\documents.index` e `storage\metadata.json`.

### Atualizar documentos indexados

Se os dois arquivos de armazenamento já existirem, a aplicação reutiliza o
índice e não detecta automaticamente alterações na pasta `documents`. Para
reindexar após adicionar, alterar ou remover documentos, faça backup ou remova
os dois arquivos de `storage` e execute a aplicação novamente. O novo índice
será criado a partir dos documentos atuais.

## Como testar

Com as dependências instaladas, execute os testes unitários dos serviços:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
```

Os testes cobrem casos de sucesso e falha. Eles utilizam um modelo simulado e
diretórios temporários, sem modificar o índice existente nem baixar o modelo de
embeddings.
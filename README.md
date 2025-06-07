# Simple Semantic Chunker

`simple-semantic-chunker` is a Python library designed to split text documents into semantically coherent chunks. This is particularly useful for preparing text for indexing in vector databases or for other NLP tasks that benefit from contextually grouped text segments.

The library leverages OpenAI's embedding models to understand the semantic meaning of sentences and groups them based on a configurable similarity threshold.

## Features

- Splits text into sentences.
- Generates embeddings for sentences using specified OpenAI models.
- Compares semantic similarity between consecutive sentences.
- Groups sentences into chunks based on a similarity threshold.
- Asynchronous support for document processing.
- Allows customization of OpenAI model, API key, and base URL.

## Installation

You can install `simple-semantic-chunker` from PyPI:

```bash
pip install simple-semantic-chunker
```

## Usage

Here's a basic example of how to use the `DocumentChunker`:

```python
import asyncio
from simple_semantic_chunker.chunker import DocumentChunker

async def main():
    # Initialize the chunker
    # You can specify your OpenAI API key and a custom base URL if needed
    # chunker = DocumentChunker(openai_api_key="YOUR_API_KEY", openai_base_url="YOUR_CUSTOM_ENDPOINT")
    chunker = DocumentChunker(openai_model="text-embedding-ada-002", similarity_threshold=0.5)

    document_text = """
    The quick brown fox jumps over the lazy dog. This sentence is about an animal.
    The weather is sunny today. The sky is clear and blue. This is about the weather.
    AI is transforming many industries. Machine learning models are becoming more powerful.
    """

    print(f"Processing document with model: {chunker.openai_model}")

    # Process the document asynchronously
    chunks = await chunker.process_document(document_text)

    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        # The 'content' of a chunk is a list of sentences
        print("Sentences:", " ".join(chunk['content']))
        # print("Embedding:", chunk['embedding'][:5], "...") # Print first 5 elements of the embedding
        print(f"Number of sentences in chunk: {len(chunk['content'])}")
        print("---")

    # Synchronous processing is also available:
    # chunks_sync = chunker.process_document_sync(document_text)
    # print(f"\nGenerated {len(chunks_sync)} chunks (synchronously):")
    # for i, chunk in enumerate(chunks_sync):
    #     print(f"--- Chunk {i+1} (sync) ---")
    #     print("Sentences:", " ".join(chunk['content']))
    #     print("---")


if __name__ == "__main__":
    asyncio.run(main())
```

### Configuration

When initializing `DocumentChunker`, you can specify:

- `openai_model`: The OpenAI embedding model to use (e.g., `"text-embedding-ada-002"`, `"text-embedding-3-small"`). Defaults to `"text-embedding-ada-002"`.
- `similarity_threshold`: A float between 0 and 1. Sentences with similarity below this threshold will start a new chunk. Defaults to `0.45`.
- `logger`: An optional custom logger instance.
- `openai_api_key`: Your OpenAI API key. If not provided, the library will attempt to use the `OPENAI_API_KEY` environment variable.
- `openai_base_url`: A custom base URL for the OpenAI API (e.g., for use with Azure OpenAI or other compatible endpoints). If not provided, the library will attempt to use the `OPENAI_BASE_URL` environment variable or the default OpenAI API URL.


## How it Works

1.  **Sentence Splitting**: The input document is first split into individual sentences.
2.  **Embedding Generation**: Each sentence is converted into a numerical vector (embedding) using the specified OpenAI model.
3.  **Similarity Comparison**: The cosine similarity between the embedding of the current sentence and the previous sentence (or the representative embedding of the current chunk) is calculated.
4.  **Chunk Creation**:
    *   If the similarity is above the `similarity_threshold`, the current sentence is added to the current chunk.
    *   If the similarity is below the threshold, the current chunk is finalized (its overall embedding is calculated from its constituent sentences), and a new chunk begins with the current sentence.
5.  **Final Output**: The process results in a list of chunks, where each chunk contains a list of sentences and the embedding for the entire chunk.

The core idea is that sentences that are semantically similar will be grouped together. The `similarity_threshold` controls how "tightly" related sentences must be to stay in the same chunk.

## Development & Contributing

This project is managed by TeaBranch.

### Setup for Development

```bash
git clone https://github.com/TeaBranch/simple-semantic-chunker.git # Replace with your repo URL
cd simple-semantic-chunker
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt # (You'll need to create this: pip freeze > requirements.txt)
pip install -e . # Install in editable mode
```

### Running Tests
(Test setup to be added)

### Publishing to PyPI (Manual)
This project is configured with a GitHub Action to automatically publish to PyPI when changes are merged to the `main` branch. For manual publishing:

1.  Ensure `setuptools`, `wheel`, and `twine` are installed: `pip install setuptools wheel twine`
2.  Increment the version in `setup.py`.
3.  Build the package: `python setup.py sdist bdist_wheel`
4.  Upload to PyPI: `twine upload dist/*` (You will need a PyPI account and API token).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

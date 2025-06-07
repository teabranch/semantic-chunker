import re
import numpy as np
from openai import OpenAI, AsyncOpenAI
import logging

# Configure basic logging if no logger is provided
default_logger = logging.getLogger(__name__)
if not default_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    default_logger.addHandler(handler)
    default_logger.setLevel(logging.INFO)

class DocumentChunker:
    """
    A class to split documents into semantically coherent chunks.
    """
    def __init__(self, openai_model: str = "text-embedding-ada-002", 
                 similarity_threshold: float = 0.45, 
                 logger=None,
                 openai_api_key: str = None,
                 openai_base_url: str = None):
        """
        Initializes a new instance of DocumentChunker.
        Args:
            openai_model (str): The OpenAI model to use for embeddings (e.g., "text-embedding-ada-002").
            similarity_threshold (float): Threshold for similarity to combine sentences.
            logger: Optional logger instance.
            openai_api_key (str, optional): OpenAI API key. Defaults to env var OPENAI_API_KEY.
            openai_base_url (str, optional): OpenAI API base URL. Defaults to env var OPENAI_BASE_URL.
        """
        self.openai_model = openai_model
        self.similarity_threshold = similarity_threshold
        self.logger = logger if logger else default_logger

        # Initialize OpenAI clients
        # The OpenAI library automatically picks up OPENAI_API_KEY and OPENAI_BASE_URL from env vars if not provided
        self.aclient = AsyncOpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_base_url
        )
        self.log_info(f"Initialized DocumentChunker with model: {self.openai_model}, threshold: {self.similarity_threshold}")

    def log_info(self, message: str):
        self.logger.info(f"DocumentChunker - {message}")

    def log_debug(self, message: str):
        self.logger.debug(f"DocumentChunker - {message}")

    def log_error(self, message: str):
        self.logger.error(f"DocumentChunker - {message}")

    async def _generate_embedding_async(self, text: str) -> list[float]:
        """Generates embedding for the given text using OpenAI (async)."""
        if not text or not text.strip():
            self.log_debug("Empty text provided for embedding, returning empty list.")
            return []
        try:
            text_to_embed = text.strip().replace("\\n", " ")
            response = await self.aclient.embeddings.create(input=[text_to_embed], model=self.openai_model)
            return response.data[0].embedding
        except Exception as e:
            self.log_error(f"Error generating embedding for text '{text[:100]}...': {e}")
            raise

    def _generate_embedding_sync(self, text: str) -> list[float]:
        """Generates embedding for the given text using OpenAI (sync)."""
        if not text or not text.strip():
            self.log_debug("Empty text provided for embedding, returning empty list.")
            return []
        try:
            text_to_embed = text.strip().replace("\\n", " ")
            response = self.client.embeddings.create(input=[text_to_embed], model=self.openai_model)
            return response.data[0].embedding
        except Exception as e:
            self.log_error(f"Error generating embedding for text '{text[:100]}...': {e}")
            raise

    def compare_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Compares two embeddings using cosine similarity.
        """
        if not isinstance(embedding1, (list, np.ndarray)) or not isinstance(embedding2, (list, np.ndarray)):
            raise ValueError("Embeddings must be lists or numpy arrays.")
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        if vec1.size == 0 or vec2.size == 0: # Handle empty embeddings
            self.log_debug("One or both embeddings are empty, similarity is 0.")
            return 0.0
            
        if vec1.shape != vec2.shape or vec1.ndim != 1:
            self.log_error(f"Invalid embedding shapes for comparison: {vec1.shape}, {vec2.shape}")
            raise ValueError("Embeddings must be 1D arrays of the same size.")

        # Check for zero vectors explicitly to avoid division by zero if norm is zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            self.log_debug("Magnitude of one or both embeddings is zero, similarity is 0.")
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        return dot_product / (norm1 * norm2)

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Splits text into sentences.
        This implementation uses a regex based on the original JavaScript version.
        JS regex: /[^\.!\?\\n]+[\.!\?]*|[^\n]+/g
        """
        if not text:
            return []
        
        # Python equivalent of the JS regex:
        # `[^.!?\\n]+` matches one or more characters that are not '.', '!', '?', or newline.
        # `(?:[.!?])*` matches zero or more '.', '!', or '?'. (non-capturing group)
        # `|` is OR.
        # `[^\n]+` matches one or more characters that are not newline.
        raw_sentences = re.findall(r"[^.!?\\n]+(?:[.!?])*|[^\n]+", text)
        
        # The original JS code had `|| [text]`.
        # This handles cases where `re.findall` returns an empty list but the input text was not empty (e.g., just whitespace or unusual format).
        if not raw_sentences and text.strip():
            return [text.strip()]
        
        # Filter out empty strings that might result from splitting, and strip whitespace from each sentence.
        return [s.strip() for s in raw_sentences if s and s.strip()]

    async def process_document(self, document: str, on_progress=None) -> list[dict]:
        """
        Processes a document by splitting it into semantically coherent chunks based on sentences (async).
        Args:
            document (str): The document text to process.
            on_progress (function, optional): Callback for progress tracking (receives float 0.0 to 1.0).
        Returns:
            list[dict]: A list of processed chunks, each chunk is a dict with "content" (list of sentences) and "embedding".
        """
        if not document or not document.strip():
            self.log_error("Document is empty or contains only whitespace.")
            raise ValueError('Document is required and cannot be empty.')

        processed_chunks = []
        try:
            sentences = self.split_into_sentences(document)
            if not sentences:
                self.log_info("No sentences found in the document after splitting.")
                # If the document was not empty but split_into_sentences returned nothing,
                # treat the whole document as one chunk if it has content.
                doc_strip = document.strip()
                if doc_strip:
                    try:
                        embedding = await self._generate_embedding_async(doc_strip)
                        if embedding:
                            return [{"content": [doc_strip], "embedding": embedding}]
                    except Exception as e:
                        self.log_error(f"Failed to embed single-block document: {e}")
                        # Fall through to return empty list or re-raise depending on desired strictness
                return []

            self.log_debug(f"Number of sentences: {len(sentences)}")
            total_sentences = len(sentences)
            processed_sentences_count = 0
            
            current_chunk_sentences_list = []
            previous_sentence_embedding = None

            for i, sentence_text in enumerate(sentences):
                sentence_text_stripped = sentence_text.strip()
                if not sentence_text_stripped: # Skip empty sentences that might have passed splitting
                    processed_sentences_count += 1
                    if on_progress and total_sentences > 0: on_progress(processed_sentences_count / total_sentences)
                    continue

                try:
                    current_sentence_embedding = await self._generate_embedding_async(sentence_text_stripped)
                except Exception: # Error already logged in _generate_embedding_async
                    processed_sentences_count += 1
                    if on_progress and total_sentences > 0: on_progress(processed_sentences_count / total_sentences)
                    continue # Skip sentence if embedding fails

                if not current_sentence_embedding: # Embedding might return empty for valid reasons or errors
                    self.log_warning(f"Could not generate embedding for sentence: '{sentence_text_stripped[:50]}...'")
                    processed_sentences_count += 1
                    if on_progress and total_sentences > 0: on_progress(processed_sentences_count / total_sentences)
                    continue
                
                self.log_debug(f"Processing sentence {i+1}/{total_sentences}: '{sentence_text_stripped[:50]}...' Embedding dim: {len(current_sentence_embedding)}")

                if previous_sentence_embedding:
                    try:
                        similarity = self.compare_similarity(previous_sentence_embedding, current_sentence_embedding)
                        self.log_debug(f"Similarity with previous sentence: {similarity:.4f}")
                    except ValueError as e:
                        self.log_error(f"Could not compare embeddings: {e}. Assuming dissimilar.")
                        similarity = -1.0 # Force a new chunk if comparison fails

                    if similarity < self.similarity_threshold:
                        if current_chunk_sentences_list:
                            chunk_text = " ".join(current_chunk_sentences_list)
                            self.log_debug(f"Chunk break due to low similarity. Finalizing chunk: '{chunk_text[:100]}...'")
                            try:
                                chunk_overall_embedding = await self._generate_embedding_async(chunk_text)
                                if chunk_overall_embedding:
                                    processed_chunks.append({
                                        "content": list(current_chunk_sentences_list),
                                        "embedding": chunk_overall_embedding
                                    })
                                else:
                                    self.log_warning(f"Could not generate embedding for finalized chunk: '{chunk_text[:100]}...'")
                            except Exception: # Error already logged
                                self.log_warning(f"Skipping chunk due to embedding failure: '{chunk_text[:100]}...'")
                            current_chunk_sentences_list = [] 
                
                current_chunk_sentences_list.append(sentence_text_stripped)
                previous_sentence_embedding = current_sentence_embedding

                processed_sentences_count += 1
                if on_progress and total_sentences > 0:
                    on_progress(processed_sentences_count / total_sentences)
            
            if current_chunk_sentences_list:
                chunk_text = " ".join(current_chunk_sentences_list)
                self.log_debug(f"Finalizing remaining chunk: '{chunk_text[:100]}...'")
                try:
                    chunk_overall_embedding = await self._generate_embedding_async(chunk_text)
                    if chunk_overall_embedding:
                        processed_chunks.append({
                            "content": list(current_chunk_sentences_list),
                            "embedding": chunk_overall_embedding
                        })
                    else:
                        self.log_warning(f"Could not generate embedding for final remaining chunk: '{chunk_text[:100]}...'")
                except Exception:
                     self.log_warning(f"Skipping final chunk due to embedding failure: '{chunk_text[:100]}...'")


            self.log_info(f"Document processing complete. Generated {len(processed_chunks)} chunks.")
            return processed_chunks

        except Exception as e:
            self.log_error(f"Critical error in process_document: {e}")
            raise RuntimeError(f"Failed to process document: {e}") from e

    def process_document_sync(self, document: str, on_progress=None) -> list[dict]:
        """
        Synchronous version of process_document.
        """
        if not document or not document.strip():
            self.log_error("Document is empty or contains only whitespace.")
            raise ValueError('Document is required and cannot be empty.')

        processed_chunks = []
        try:
            sentences = self.split_into_sentences(document)
            if not sentences:
                self.log_info("No sentences found in the document after splitting.")
                doc_strip = document.strip()
                if doc_strip:
                    try:
                        embedding = self._generate_embedding_sync(doc_strip)
                        if embedding:
                            return [{"content": [doc_strip], "embedding": embedding}]
                    except Exception as e:
                        self.log_error(f"Failed to embed single-block document (sync): {e}")
                return []

            self.log_debug(f"Number of sentences: {len(sentences)}")
            total_sentences = len(sentences)
            processed_sentences_count = 0
            
            current_chunk_sentences_list = []
            previous_sentence_embedding = None

            for i, sentence_text in enumerate(sentences):
                sentence_text_stripped = sentence_text.strip()
                if not sentence_text_stripped:
                    processed_sentences_count += 1
                    if on_progress and total_sentences > 0: on_progress(processed_sentences_count / total_sentences)
                    continue
                
                try:
                    current_sentence_embedding = self._generate_embedding_sync(sentence_text_stripped)
                except Exception:
                    processed_sentences_count += 1
                    if on_progress and total_sentences > 0: on_progress(processed_sentences_count / total_sentences)
                    continue

                if not current_sentence_embedding:
                    self.log_warning(f"Could not generate embedding for sentence (sync): '{sentence_text_stripped[:50]}...'")
                    processed_sentences_count += 1
                    if on_progress and total_sentences > 0: on_progress(processed_sentences_count / total_sentences)
                    continue
                
                self.log_debug(f"Processing sentence {i+1}/{total_sentences} (sync): '{sentence_text_stripped[:50]}...' Embedding dim: {len(current_sentence_embedding)}")

                if previous_sentence_embedding:
                    try:
                        similarity = self.compare_similarity(previous_sentence_embedding, current_sentence_embedding)
                        self.log_debug(f"Similarity with previous sentence (sync): {similarity:.4f}")
                    except ValueError as e:
                        self.log_error(f"Could not compare embeddings (sync): {e}. Assuming dissimilar.")
                        similarity = -1.0

                    if similarity < self.similarity_threshold:
                        if current_chunk_sentences_list:
                            chunk_text = " ".join(current_chunk_sentences_list)
                            self.log_debug(f"Chunk break (sync). Finalizing chunk: '{chunk_text[:100]}...'")
                            try:
                                chunk_overall_embedding = self._generate_embedding_sync(chunk_text)
                                if chunk_overall_embedding:
                                    processed_chunks.append({
                                        "content": list(current_chunk_sentences_list),
                                        "embedding": chunk_overall_embedding
                                    })
                                else:
                                    self.log_warning(f"Could not generate embedding for finalized chunk (sync): '{chunk_text[:100]}...'")
                            except Exception:
                                self.log_warning(f"Skipping chunk due to embedding failure (sync): '{chunk_text[:100]}...'")
                            current_chunk_sentences_list = []
                
                current_chunk_sentences_list.append(sentence_text_stripped)
                previous_sentence_embedding = current_sentence_embedding

                processed_sentences_count += 1
                if on_progress and total_sentences > 0:
                    on_progress(processed_sentences_count / total_sentences)
            
            if current_chunk_sentences_list:
                chunk_text = " ".join(current_chunk_sentences_list)
                self.log_debug(f"Finalizing remaining chunk (sync): '{chunk_text[:100]}...'")
                try:
                    chunk_overall_embedding = self._generate_embedding_sync(chunk_text)
                    if chunk_overall_embedding:
                        processed_chunks.append({
                            "content": list(current_chunk_sentences_list),
                            "embedding": chunk_overall_embedding
                        })
                    else:
                        self.log_warning(f"Could not generate embedding for final remaining chunk (sync): '{chunk_text[:100]}...'")
                except Exception:
                    self.log_warning(f"Skipping final chunk due to embedding failure (sync): '{chunk_text[:100]}...'")

            self.log_info(f"Document processing complete (sync). Generated {len(processed_chunks)} chunks.")
            return processed_chunks

        except Exception as e:
            self.log_error(f"Critical error in process_document_sync: {e}")
            raise RuntimeError(f"Failed to process document (sync): {e}") from e


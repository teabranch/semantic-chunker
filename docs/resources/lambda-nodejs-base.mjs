import { generateEmbedding } from "./bedrock.mjs";

export class DocumentChunker {
    /**
     * Initializes a new instance of DocumentChunker.
     * @param {Object} config - Configuration options.
     * @param {String} config.modelId - Model Id in Bedrock for embeddings.
     * @param {number} config.similarityThreshold - Threshold for similarity.
     */
    constructor(config = {}) {
        this.modelId = config.modelId || 'amazon.nova-pro-v1:0';
        this.similarityThreshold = config.similarityThreshold || 0.45;
        this.logger = config.logger || null;
  
        this.logInfo(`${this.modelId}, ${this.similarityThreshold}`);
    }
  
    /**
     * Logs an info message if logger is defined.
     * @param {string} message - The message to log.
     */
    logInfo(message) {
        this.logger?.info(`DocumentChunker - ${message}`);
    }
  
    /**
     * Logs a debug message if logger is defined.
     * @param {string} message - The message to log.
     */
    logDebug(message) {
        this.logger?.debug(`DocumentChunker - ${message}`);
    }
  
    /**
     * Logs an error message if logger is defined.
     * @param {string} message - The message to log.
     */
    logError(message) {
        this.logger?.error(`DocumentChunker - ${message}`);
    }
  
    compareSimilarity(embedding1, embedding2) {
        if (!Array.isArray(embedding1) || !Array.isArray(embedding2) || 
            embedding1.length !== embedding2.length) {
            throw new Error('Invalid embeddings provided');
        }
  
        const dotProduct = embedding1.reduce((sum, val, idx) => sum + val * embedding2[idx], 0);
        const magnitude1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0));
        return dotProduct / (magnitude1 * magnitude2);
    }
  
    /**
     * Splits text into sentences, including by newlines.
     * @param {string} text - The text to split.
     * @returns {string[]} - Array of sentences.
     */
    splitIntoSentences(text) {
        return text.match(/[^\.!\?\n]+[\.!\?]*|[^\n]+/g) || [text];
    }
  
    /**
     * Processes a document by splitting it into semantically coherent chunks based on sentences
     * @param {string} document - The document text to process
     * @param {function} [onProgress] - Callback for progress tracking
     * @throws {Error} If document is empty or processing fails
     * @returns {Promise<Array>} Returns an array of processed chunks
     */
    async processDocument(document, onProgress) {
        if (!document) {
            throw new Error('Document is required');
        }
        const processedChunks = [];
        try {
            const sentences = this.splitIntoSentences(document);
            this.logDebug(`Number of sentences: ${sentences.length}`);
            const totalSentences = sentences.length;
            let processedSentences = 0;
            let currentChunk = [];
            let currentEmbedding = null;
  
            for (let sentence of sentences) {
                if (sentence.trim().length === 0) {
                  continue;
                }
                const embedding = await generateEmbedding(sentence.trim(), this.modelId);
                this.logDebug(`Embedding length: ${embedding.length}`);
  
                if (currentEmbedding) {
                    const similarity = this.compareSimilarity(currentEmbedding, embedding);
                    this.logDebug(`Similarity: ${similarity}`);
                    if (similarity < this.similarityThreshold) {
                        processedChunks.push({
                            content: currentChunk,
                            embedding: currentEmbedding
                        });
                        currentChunk = [];
                    }
                }
  
                currentChunk.push(sentence);
                currentEmbedding = embedding;
                processedSentences += 1;
  
                if (onProgress) {
                    onProgress(processedSentences / totalSentences);
                }
            }
  
            if (currentChunk.length > 0) {
                const chunkEmbedding = await generateEmbedding(currentChunk.join(" "), this.modelId);
                this.logDebug(`Chunk embedding length: ${chunkEmbedding.length}`);
  
                processedChunks.push({ content: currentChunk,
                    embedding: chunkEmbedding
                });
            }
  
            return processedChunks;
        } catch (error) {
            this.logError("Error processing document");
            throw new Error(`Failed to process document: ${error.message}\n${error}`);
        }
    }
  }

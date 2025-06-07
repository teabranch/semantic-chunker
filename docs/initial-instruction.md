lambda-nodejs-base.mjs is a lambda written to chunk text for indexing

it relies on createEmbedding - replace that with OpenAI library, and add a requirement for selecting a model (OpenAI library will allow override of endpoint, making it global)

Rewrite this as python, add deployment package to pypi called simple-semantic-chunker
Organization for pypi is TeaBranch
Add deployment pipeline for github for pypi, so merging to master creates a new version, and kicks in the deploy to pypi 

Note I implemented a similar function here - change the implementation if there is a better way to do it - similiarity check between embeddings is the top poriority of it.

Update README.md accordingly.
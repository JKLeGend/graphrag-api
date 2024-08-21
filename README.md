# graphrag-api
<div align="left">
  <p><strong>graphrag-api provides api service for microsoft graphrag.</strong></p>
</div>

## Overview

This project packages microsoft graphrag into api service, provides indexing api and query api, which is more effective to integrate into other project; Localizes the promotes, which performs better for chinese;

### Query

1. **Local Search**
   - Utilizes GraphRAG technology for efficient retrieval in local knowledge bases
   - Suitable for quick access to pre-defined structured information
   - Leverages graph structures to improve retrieval accuracy and relevance

2. **Global Search**
   - Searches for information in a broader scope, beyond local knowledge bases
   - Suitable for queries requiring more comprehensive information
   - Utilizes GraphRAG's global context understanding capabilities to provide richer search results

### Local LLM

Supports various open-source LLMs run through Ollama
- Config in `./indexing/settings.yaml`

## Installation

Ensure that you have Python 3.8 or higher installed on your system:
1. Install the GraphRAG dir from this repo under `./graphrag` (0.3.0, with an bug fixed, see latest commit in `./graphrag`, you can upgrade it accordingly):

   ```bash
   pip install -e ./graphrag
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Launch the API server:

   ```bash
   python api.py --host 0.0.0.0 --port 8012 --reload
   ```

## Usage

   1. API Endpoints:
      - `/v1/index`: POST request for performing indexing
      - `/v1/index/{resume}`: GET request for checking indexing status
      - `/v1/chat/completions`: POST request for performing searches
      - `/v1/models`: GET request to retrieve the list of available models

   2. Example:
      - indexing:
      ```
      curl http://localhost:8012/v1/index \
       -X POST \
       -d '{
              "resume": "ragtest"
           }' \
       -H "Content-Type: application/json"
      ```
      ```
      curl http://localhost:8012/v1/index/ragtest \
       -X GET
      ```
      - query:
      ```
      curl http://localhost:8012/v1/chat/completions \
       -X POST \
       -d '{
              "resume": "ragtest",
              "model": "graphrag-local-search:latest",
              "messages": [{"role": "user", "content": "who are you?"}],
              "temperature": 0.7   
           }' \
       -H "Content-Type: application/json"
      ```

## Available Models

- `graphrag-local-search:latest`: Local search
- `graphrag-global-search:latest`: Global search

## Notes

- Ensure that you have origin input files in `./indexing/input` directory.
- Ensure that you have launched and configured llm service api, embeddings service api.

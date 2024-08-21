
import uuid
import time
import logging
import yaml
import asyncio
import argparse
from fastapi import Body
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path
from graphrag.config import (
    create_graphrag_config,
)
from indexing import indexing
from query import query
from query import ChatCompletionRequest

logger = logging.getLogger(__name__)

INDEXING_ROOT = "./indexing"

class IndexingRequest(BaseModel):
    root: Optional[str] = INDEXING_ROOT
    verbose: Optional[bool] = False
    resume: Optional[str] = None
    memprofile: Optional[bool] = False
    nocache: Optional[bool] = False
    reporter: Optional[str] = "rich"
    config: Optional[str] = None
    emit: Optional[str] = None
    dryrun: Optional[bool] = False
    init: Optional[bool] = False
    overlay_defaults: Optional[bool] = False
    cli: Optional[bool] = False

def _read_config_parameters(root: str, config: str | None):
    _root = Path(root)
    settings_yaml = (
        Path(config)
        if config and Path(config).suffix in [".yaml", ".yml"]
        else _root / "settings.yaml"
    )
    if not settings_yaml.exists():
        settings_yaml = _root / "settings.yml"

    if settings_yaml.exists():
        logger.info(f"Reading settings from {settings_yaml}")
        with settings_yaml.open(
            "rb",
        ) as file:
            import yaml

            data = yaml.safe_load(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root)

    settings_json = (
        Path(config)
        if config and Path(config).suffix == ".json"
        else _root / "settings.json"
    )
    if settings_json.exists():
        logger.info(f"Reading settings from {settings_json}")
        with settings_json.open("rb") as file:
            import json

            data = json.loads(file.read().decode(encoding="utf-8", errors="strict"))
            return create_graphrag_config(data, root)

    logger.info("Reading settings from environment variables")
    return create_graphrag_config(root_dir=root)

def load_settings():
    settings = _read_config_parameters(INDEXING_ROOT, None)
    return settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings
    try:
        logger.info("Loading settings...")
        settings = load_settings()
        logger.info("Settings loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading settings: {str(e)}")
        raise

    yield

    logger.info("Shutting down...")

app = FastAPI(lifespan=lifespan)

async def run_indexing(request: IndexingRequest):
    try:
        await indexing(
            root=request.root,
            verbose=request.verbose or False,
            resume=request.resume,
            memprofile=request.memprofile or False,
            nocache=request.nocache or False,
            reporter=request.reporter,
            config=request.config,
            emit=request.emit,
            dryrun=request.dryrun or False,
            init=request.init or False,
            overlay_defaults=request.overlay_defaults or False,
            cli=False,
        )

        logger.info("Indexing completed successfully")
        return
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        raise

@app.post("/v1/index")
async def start_indexing(request: IndexingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_indexing, request)
    return {"status": 200, "message": f"Indexing process has been started in the background, checkout {INDEXING_ROOT}/output/{request.resume}"}

@app.get("/v1/index/{resume}")
async def indexing_status(resume: str):
    root_dir = INDEXING_ROOT
    logs = ''
    indexing_log = ( Path(root_dir) / "output" / resume / "reports" / "indexing-engine.log" )
    if not Path(indexing_log).exists():
        return {
            "status": 404,
            "message": "Indexing process not found.",
            "progress": 0,
            "logs": ""
        }

    with open(indexing_log, 'r') as file:
        logs = file.read()

    def message():
        if "Indexing completed successfully" in logs or "Indexing failed" in logs:
            return "Indexing process end."
        return "Indexing..."
    def progress():
        if "Indexing completed successfully" in logs or "Indexing failed" in logs:
            return 1
        if "create_final_documents.parquet" in logs:
            return 0.98
        if "create_base_documents.parquet" in logs:
            return 0.97
        if "create_final_text_units.parquet" in logs:
            return 0.96
        if "create_final_community_reports.parquet" in logs:
            return 0.95
        if "join_text_units_to_relationship_ids.parquet" in logs:
            return 0.94
        if "create_final_relationships.parquet" in logs:
            return 0.93
        if "join_text_units_to_entity_ids.parquet" in logs:
            return 0.92
        if "create_final_communities.parquet" in logs:
            return 0.91
        if "create_final_nodes.parquet" in logs:
            return 0.90
        if "create_final_entities.parquet" in logs:
            return 0.88
        if "create_base_entity_graph.parquet" in logs:
            return 0.86
        if "join_text_units_to_covariate_ids.parquet" in logs:
            return 0.84
        if "create_summarized_entities.parquet" in logs:
            return 0.82
        if "create_final_covariates.parquet" in logs:
            return 0.80
        if "create_base_extracted_entities.parquet" in logs:
            return 0.35
        if "create_base_text_units.parquet" in logs:
            return 0.05
        return 0
    return {
        "status": 200,
        "message": message(),
        "progress": progress(),
        "logs": logs
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    return await query(request, settings)

@app.get("/v1/models")
async def list_models():
    logger.info("[api]/v1/models")
    current_time = int(time.time())
    models = [
        {"id": f"{uuid.uuid4()}", "name": "graphrag-local-search:latest", "type": "model", "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time - 100000)), "cteated_by": "graphrag-api"},
        {"id": f"{uuid.uuid4()}", "name": "graphrag-global-search:latest", "type": "model", "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time - 95000)), "cteated_by": "graphrag-api"},
    ]

    response = {
        "data": models
    }
    return JSONResponse(content=response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the GraphRAG API server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default="8012", help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload mode")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

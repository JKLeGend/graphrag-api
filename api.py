
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
from indexing import indexing
from query import query
from query import ChatCompletionRequest

# Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndexingRequest(BaseModel):
    root: Optional[str] = "./indexing"
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



def load_settings():
    with open('./indexing/settings.yaml', 'r') as f:
        config = yaml.safe_load(f)

    settings = {
        'llm': config.get('llm'),
        'embeddings': config.get('embeddings'),
        'input': config.get('input'),
        'storage': config.get('storage'),
        'reporting': config.get('reporting'),
    }
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
        #return {"status": "success", "message": "Indexing completed successfully"}
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        #return {"status": "error", "message": f"Indexing failed: {str(e)}"}



@app.post("/v1/index")
async def start_indexing(request: IndexingRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_indexing, request)
    #await run_indexing(request)
    return {"status": 200, "message": "Indexing process has been started in the background"}

@app.get("/v1/index/{resume}")
async def indexing_status(resume: str):
    root_dir = './indexing'
    logs = ''
    indexing_log = ( Path(root_dir) / "output" / resume / "reports" / "indexing-engine.log" )
    with open(indexing_log, 'r') as file:
        logs = file.read()
    return {
        "status": 200,
        "message": "Indexing...",
        "logs": logs
    }

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

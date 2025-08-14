import os
import json
import logging
import tempfile
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

from universal_solver import orchestrate  # <-- NEW IMPORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Data Analyst Agent",
    description="LLM-powered data analysis API that can scrape, analyze, and visualize data",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve upload UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Data Analyst Agent is running"}


@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    Main API endpoint — expects:
    - questions.txt (mandatory)
    - Other data files (optional)
    """
    try:
        logger.info(f"Received {len(files)} files for analysis")
        
        questions_path = None
        attachment_paths = []

        for file in files:
            content = await file.read()
            filename = file.filename

            # Save each file to temp
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix)
            tmp_file.write(content)
            tmp_file.close()

            if filename.lower() == "questions.txt":
                questions_path = tmp_file.name
                logger.info(f"Saved questions.txt to {questions_path}")
            else:
                attachment_paths.append(tmp_file.name)
                logger.info(f"Saved attachment {filename} to {tmp_file.name}")

        if not questions_path:
            raise HTTPException(status_code=400, detail="questions.txt file is required")

        # Run the universal solver
        result = orchestrate(questions_path, attachment_paths)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Analysis failed: {str(e)}"}
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

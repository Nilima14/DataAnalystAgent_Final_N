import os
import json
import base64
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask.helpers import send_file
import requests
from werkzeug.utils import secure_filename

from data_analyzer import DataAnalyzer
from web_scraper import WebScraper
from code_executor import CodeExecutor
from gemini_client import GeminiClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Initialize components (without Gemini client to avoid startup errors)
web_scraper = WebScraper()
code_executor = CodeExecutor()

@app.route("/")
def root():
    """Serve the main testing interface"""
    return render_template("index.html")

@app.route("/health")
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Data Analyst Agent is running"})

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_from_directory("static", filename)

@app.route("/api/", methods=["POST"])
def analyze_data():
    """
    Main API endpoint for data analysis tasks.
    Accepts files including questions.txt and optional data files.
    """
    try:
        logger.info(f"Received analysis request")
        
        # Get uploaded files
        files = request.files.getlist('files')
        logger.info(f"Received {len(files)} files for analysis")
        
        # Process uploaded files
        questions_content = None
        attachments = {}
        
        for file in files:
            if file.filename == "questions.txt":
                questions_content = file.read().decode("utf-8")
                logger.info(f"Questions received: {questions_content[:200]}...")
            else:
                # Store other attachments
                content = file.read()
                attachments[file.filename] = {
                    "content": content,
                    "content_type": file.content_type
                }
                logger.info(f"Attachment received: {file.filename} ({file.content_type})")
        
        if not questions_content:
            return jsonify({"error": "questions.txt file is required"}), 400
        
        # Process the analysis task synchronously (convert async to sync)
        result = process_analysis_task_sync(questions_content, attachments)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing analysis request: {str(e)}")
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

def process_analysis_task_sync(questions: str, attachments: Dict[str, Any]) -> Any:
    """
    Process the analysis task synchronously (wrapper for async function)
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_analysis_task(questions, attachments))
    finally:
        loop.close()

async def process_analysis_task(questions: str, attachments: Dict[str, Any]) -> Any:
    """
    Process the analysis task following the workflow:
    1. Break down the task using LLM
    2. Generate code to solve the task
    3. Execute code with error handling and retry
    4. Return formatted results
    """
    try:
        logger.info("Starting task breakdown")
        
        # Step 1: Task breakdown - Initialize Gemini client with error handling
        try:
            gemini_client = GeminiClient()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            # Use fallback mode without Gemini API
            return {"error": "AI service temporarily unavailable", "fallback_used": True}
        
        try:
            # Set timeout for task breakdown
            task_breakdown = await asyncio.wait_for(
                gemini_client.break_down_task(questions, attachments),
                timeout=30
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Task breakdown failed: {e}, using fallback")
            task_breakdown = gemini_client._create_fallback_breakdown(questions)
        logger.info(f"Task breakdown completed: {task_breakdown[:200]}...")
        
        # Step 2: Generate code based on breakdown
        logger.info("Generating Python code")
        try:
            code = await asyncio.wait_for(
                gemini_client.generate_code(task_breakdown, questions, attachments),
                timeout=30
            )
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Code generation failed: {e}, using fallback")
            code = gemini_client._create_fallback_code(questions)
        logger.info(f"Generated code ({len(code)} characters)")
        
        # Step 3: Execute code with retry mechanism
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Executing code (attempt {attempt + 1}/{max_retries})")
                result = await code_executor.execute_code(code, attachments)
                
                # If execution successful, return result
                if result.get("success"):
                    logger.info("Code execution successful")
                    return result["output"]
                else:
                    raise Exception(result.get("error", "Unknown execution error"))
                    
            except Exception as e:
                logger.warning(f"Code execution failed (attempt {attempt + 1}): {str(e)}")
                
                if attempt < max_retries - 1:
                    # Try to fix the code using LLM
                    logger.info("Attempting to fix code errors")
                    error_message = str(e)
                    try:
                        code = await asyncio.wait_for(
                            gemini_client.fix_code_errors(code, error_message, questions),
                            timeout=30
                        )
                    except (asyncio.TimeoutError, Exception) as fix_error:
                        logger.warning(f"Code fixing failed: {fix_error}, using original code")
                        # Continue with original code for next attempt
                else:
                    # Final attempt failed
                    raise Exception(f"Code execution failed after {max_retries} attempts: {str(e)}")
        
    except Exception as e:
        logger.error(f"Task processing failed: {str(e)}")
        raise e

@app.route("/api/test", methods=["POST"])
def test_endpoint():
    """Test endpoint for debugging"""
    return jsonify({"message": "Test endpoint working", "status": "ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
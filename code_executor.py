import subprocess
import tempfile
import os
import json
import logging
import asyncio
import sys
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

logger = logging.getLogger(__name__)

class CodeExecutor:
    """Secure code execution environment"""
    
    def __init__(self):
        self.timeout = 180  # 3 minutes timeout
        self.temp_dir = Path(tempfile.gettempdir()) / "data_analyst_temp"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def execute_code(self, code: str, attachments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Python code in a controlled environment"""
        try:
            logger.info("Preparing code execution environment")
            
            # Extract code from markdown if present
            clean_code = self._extract_python_code(code)
            
            # Create temporary script
            script_path = self.temp_dir / f"script_{os.getpid()}.py"
            
            # Write attachments to temporary files
            attachment_paths = {}
            for filename, file_data in attachments.items():
                file_path = self.temp_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(file_data['content'])
                attachment_paths[filename] = str(file_path)
            
            # Modify code to handle file paths
            modified_code = self._modify_code_for_execution(clean_code, attachment_paths)
            
            # Write the script
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(modified_code)
            
            logger.info(f"Executing script: {script_path}")
            
            # Execute the script
            result = await self._run_script(script_path)
            
            # Cleanup
            self._cleanup_files([script_path] + list(Path(p) for p in attachment_paths.values()))
            
            return result
            
        except Exception as e:
            logger.error(f"Code execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _extract_python_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks"""
        import re
        
        # Look for ```python code blocks
        python_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if python_blocks:
            return python_blocks[0].strip()
        
        # Look for ``` code blocks (assume Python)
        code_blocks = re.findall(r'```\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Return original text if no code blocks found
        return text.strip()
    
    def _modify_code_for_execution(self, code: str, attachment_paths: Dict[str, str]) -> str:
        """Modify code to work with local file paths"""
        modified = code
        
        # Add standard imports if not present
        imports_to_add = [
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import matplotlib",
            "matplotlib.use('Agg')",
            "import seaborn as sns",
            "import requests",
            "from bs4 import BeautifulSoup",
            "import base64",
            "import io",
            "import json",
            "import re",
            "from scipy import stats",
            "import warnings",
            "warnings.filterwarnings('ignore')"
        ]
        
        # Check if imports are already present and add if missing
        for imp in imports_to_add:
            if imp.split()[1] not in modified:  # Check if module is not already imported
                modified = imp + "\n" + modified
        
        # Replace file references with actual paths
        for filename, filepath in attachment_paths.items():
            modified = modified.replace(f'"{filename}"', f'"{filepath}"')
            modified = modified.replace(f"'{filename}'", f"'{filepath}'")
        
        # Ensure the script prints output in a way we can capture
        if "print(" not in modified:
            modified += "\n\n# Output the final result\nprint(json.dumps(result) if 'result' in locals() else 'No result variable found')"
        
        return modified
    
    async def _run_script(self, script_path: Path) -> Dict[str, Any]:
        """Run the Python script and capture output"""
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir)
            )
            
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.timeout
            )
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            if process.returncode == 0:
                logger.info("Script executed successfully")
                
                # Try to parse output as JSON, otherwise return as string
                try:
                    output = json.loads(stdout_text.strip())
                except json.JSONDecodeError:
                    output = stdout_text.strip()
                
                return {
                    "success": True,
                    "output": output,
                    "stdout": stdout_text,
                    "stderr": stderr_text
                }
            else:
                logger.error(f"Script execution failed with return code: {process.returncode}")
                return {
                    "success": False,
                    "error": f"Script failed with return code {process.returncode}",
                    "stdout": stdout_text,
                    "stderr": stderr_text
                }
                
        except asyncio.TimeoutError:
            logger.error("Script execution timed out")
            return {
                "success": False,
                "error": f"Script execution timed out after {self.timeout} seconds"
            }
        except Exception as e:
            logger.error(f"Script execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    def _cleanup_files(self, file_paths: list):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Cleaned up: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {file_path}: {e}")

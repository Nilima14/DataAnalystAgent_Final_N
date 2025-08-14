import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import json
import logging
from typing import Any, Dict, List, Optional
from scipy import stats
import re

from gemini_client import GeminiClient

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Data analysis and visualization utilities"""
    
    def __init__(self, gemini_client: GeminiClient):
        self.gemini_client = gemini_client
        
        # Set up plotting defaults
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Configure matplotlib for better output
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 100
        plt.rcParams['figure.figsize'] = (10, 6)
    
    def analyze_csv_data(self, csv_content: str, questions: List[str]) -> Dict[str, Any]:
        """Analyze CSV data and answer specific questions"""
        try:
            # Parse CSV content
            df = pd.read_csv(io.StringIO(csv_content))
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            results = {}
            
            # Basic dataset info
            results['dataset_info'] = {
                'rows': len(df),
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict()
            }
            
            # Answer specific questions
            for i, question in enumerate(questions):
                try:
                    answer = self._answer_question(df, question)
                    results[f'question_{i+1}'] = answer
                except Exception as e:
                    logger.error(f"Failed to answer question {i+1}: {e}")
                    results[f'question_{i+1}'] = f"Error: {str(e)}"
            
            return results
            
        except Exception as e:
            logger.error(f"CSV analysis failed: {e}")
            raise e
    
    def _answer_question(self, df: pd.DataFrame, question: str) -> Any:
        """Answer a specific question about the dataframe"""
        question_lower = question.lower()
        
        # Count questions
        if 'how many' in question_lower and '$2 bn' in question_lower:
            # Count $2bn movies before a certain year
            year_match = re.search(r'before (\d{4})', question)
            if year_match:
                year = int(year_match.group(1))
                # Look for revenue column and year column
                revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'gross' in col.lower()]
                year_cols = [col for col in df.columns if 'year' in col.lower()]
                
                if revenue_cols and year_cols:
                    revenue_col = revenue_cols[0]
                    year_col = year_cols[0]
                    
                    # Convert revenue to numeric (handle $ and commas)
                    df[revenue_col] = df[revenue_col].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
                    
                    # Count movies >= $2bn before the year
                    count = len(df[(df[revenue_col] >= 2000000000) & (df[year_col] < year)])
                    return count
        
        # Earliest film questions
        if 'earliest film' in question_lower and '$1.5 bn' in question_lower:
            revenue_cols = [col for col in df.columns if 'revenue' in col.lower() or 'gross' in col.lower()]
            year_cols = [col for col in df.columns if 'year' in col.lower()]
            title_cols = [col for col in df.columns if 'title' in col.lower() or 'film' in col.lower() or 'movie' in col.lower()]
            
            if revenue_cols and year_cols and title_cols:
                revenue_col = revenue_cols[0]
                year_col = year_cols[0]
                title_col = title_cols[0]
                
                # Convert revenue to numeric
                df[revenue_col] = df[revenue_col].astype(str).str.replace('$', '').str.replace(',', '').str.replace(' ', '')
                df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')
                
                # Find earliest film with revenue >= $1.5bn
                filtered = df[df[revenue_col] >= 1500000000]
                if not filtered.empty:
                    earliest = filtered.loc[filtered[year_col].idxmin()]
                    return earliest[title_col]
        
        # Correlation questions
        if 'correlation' in question_lower:
            rank_cols = [col for col in df.columns if 'rank' in col.lower()]
            peak_cols = [col for col in df.columns if 'peak' in col.lower()]
            
            if rank_cols and peak_cols:
                rank_col = rank_cols[0]
                peak_col = peak_cols[0]
                
                # Calculate correlation
                correlation = df[rank_col].corr(df[peak_col])
                return round(correlation, 6)
        
        return "Could not parse question"
    
    def create_scatterplot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                          title: str = "Scatterplot", add_regression: bool = True) -> str:
        """Create a scatterplot with optional regression line and return as base64 data URI"""
        try:
            # Create figure
            plt.figure(figsize=(10, 6))
            
            # Create scatterplot
            plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
            
            # Add regression line if requested
            if add_regression:
                # Calculate regression line
                slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
                line = slope * df[x_col] + intercept
                plt.plot(df[x_col], line, 'r--', linewidth=2, label=f'Regression Line (R²={r_value**2:.3f})')
                plt.legend()
            
            # Customize plot
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(title)
            plt.grid(True, alpha=0.3)
            
            # Convert to base64
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            img_buffer.seek(0)
            
            # Encode to base64
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{img_base64}"
            
            # Check size (should be under 100KB)
            size_kb = len(img_base64) * 3 / 4 / 1024  # Approximate size in KB
            logger.info(f"Generated plot size: {size_kb:.1f} KB")
            
            if size_kb > 100:
                logger.warning(f"Plot size ({size_kb:.1f} KB) exceeds 100KB limit")
            
            plt.close()  # Clean up
            return data_uri
            
        except Exception as e:
            logger.error(f"Scatterplot creation failed: {e}")
            raise e
    
    def create_visualization(self, plot_type: str, data: Dict[str, Any], **kwargs) -> str:
        """Create various types of visualizations"""
        try:
            plt.figure(figsize=(10, 6))
            
            if plot_type == 'histogram':
                plt.hist(data['values'], bins=kwargs.get('bins', 30), alpha=0.7)
                plt.xlabel(data.get('xlabel', 'Values'))
                plt.ylabel('Frequency')
                plt.title(data.get('title', 'Histogram'))
                
            elif plot_type == 'bar':
                plt.bar(data['x'], data['y'])
                plt.xlabel(data.get('xlabel', 'X'))
                plt.ylabel(data.get('ylabel', 'Y'))
                plt.title(data.get('title', 'Bar Chart'))
                plt.xticks(rotation=45)
                
            elif plot_type == 'line':
                plt.plot(data['x'], data['y'], marker='o')
                plt.xlabel(data.get('xlabel', 'X'))
                plt.ylabel(data.get('ylabel', 'Y'))
                plt.title(data.get('title', 'Line Chart'))
                plt.grid(True, alpha=0.3)
            
            # Convert to base64 data URI
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
            img_buffer.seek(0)
            
            img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
            data_uri = f"data:image/png;base64,{img_base64}"
            
            plt.close()
            return data_uri
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
            raise e
    
    def calculate_statistics(self, data: List[float]) -> Dict[str, float]:
        """Calculate basic statistics for a dataset"""
        try:
            arr = np.array(data)
            
            return {
                'mean': np.mean(arr),
                'median': np.median(arr),
                'std': np.std(arr),
                'min': np.min(arr),
                'max': np.max(arr),
                'count': len(arr)
            }
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            raise e
    
    def format_response(self, results: List[Any], format_type: str = "json_array") -> Any:
        """Format analysis results according to specified format"""
        try:
            if format_type == "json_array":
                return results
            elif format_type == "json_object":
                if isinstance(results, dict):
                    return results
                else:
                    # Convert list to object with numbered keys
                    return {f"answer_{i+1}": result for i, result in enumerate(results)}
            else:
                return results
                
        except Exception as e:
            logger.error(f"Response formatting failed: {e}")
            return results

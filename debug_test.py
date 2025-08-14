import requests
import pandas as pd
import traceback

# Test the Wikipedia scraping directly
url = "https://en.wikipedia.org/wiki/List_of_highest-grossing_films"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

try:
    print("Fetching Wikipedia page...")
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    print(f"Page fetched successfully. Length: {len(response.text)}")
    
    print("Parsing tables with pandas...")
    tables = pd.read_html(response.text, header=0)
    print(f"Found {len(tables)} tables")
    
    for i, table in enumerate(tables):
        print(f"Table {i}: Shape {table.shape}, Columns: {list(table.columns)[:5]}")
        if i < 3:  # Show first few rows of first 3 tables
            print(f"First few rows:\n{table.head(2)}")
            print("---")
        
except Exception as e:
    print(f"Error: {e}")
    print(f"Traceback: {traceback.format_exc()}")
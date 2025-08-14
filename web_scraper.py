import trafilatura
import requests
from bs4 import BeautifulSoup
import logging
from typing import Optional, Dict, Any
import asyncio
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

class WebScraper:
    """Web scraping utilities for data collection"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_website_text_content(self, url: str) -> str:
        """
        Extract main text content from a website using trafilatura.
        Returns clean, readable text content.
        """
        try:
            logger.info(f"Fetching content from: {url}")
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                raise Exception(f"Failed to download content from {url}")
            
            text = trafilatura.extract(downloaded)
            
            if not text:
                raise Exception(f"Failed to extract text content from {url}")
            
            logger.info(f"Successfully extracted {len(text)} characters of text content")
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed for {url}: {e}")
            raise e
    
    def get_website_html(self, url: str) -> str:
        """
        Get raw HTML content from a website.
        Useful when you need to parse specific HTML elements.
        """
        try:
            logger.info(f"Fetching HTML from: {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            logger.info(f"Successfully fetched HTML ({len(response.text)} characters)")
            return response.text
            
        except Exception as e:
            logger.error(f"HTML fetch failed for {url}: {e}")
            raise e
    
    def parse_html_table(self, html: str, table_selector: str = "table") -> Dict[str, Any]:
        """
        Parse HTML table data using BeautifulSoup.
        Returns structured data from tables.
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            tables = soup.select(table_selector)
            
            if not tables:
                raise Exception(f"No tables found with selector: {table_selector}")
            
            # Parse the first table found
            table = tables[0]
            headers = []
            rows = []
            
            # Extract headers
            header_row = table.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Extract data rows
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if cells:
                    rows.append(cells)
            
            logger.info(f"Parsed table with {len(headers)} columns and {len(rows)} rows")
            return {
                "headers": headers,
                "rows": rows,
                "total_rows": len(rows)
            }
            
        except Exception as e:
            logger.error(f"Table parsing failed: {e}")
            raise e
    
    async def scrape_with_playwright(self, url: str, wait_for_selector: Optional[str] = None) -> str:
        """
        Scrape dynamic content using Playwright browser automation.
        Useful for JavaScript-heavy websites.
        """
        try:
            logger.info(f"Scraping with Playwright: {url}")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                # Navigate to page
                await page.goto(url, wait_until='domcontentloaded', timeout=60000)
                
                # Wait for specific selector if provided
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=30000)
                
                # Get page content
                content = await page.content()
                await browser.close()
                
                logger.info(f"Playwright scraping successful ({len(content)} characters)")
                return content
                
        except Exception as e:
            logger.error(f"Playwright scraping failed for {url}: {e}")
            raise e
    
    def extract_links(self, html: str, base_url: str = "") -> list:
        """Extract all links from HTML content"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            links = []
            
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text(strip=True)
                
                # Handle relative URLs
                if href.startswith('/') and base_url:
                    href = base_url.rstrip('/') + href
                elif not href.startswith(('http://', 'https://', 'mailto:', 'tel:')):
                    continue
                
                links.append({
                    'url': href,
                    'text': text
                })
            
            logger.info(f"Extracted {len(links)} links")
            return links
            
        except Exception as e:
            logger.error(f"Link extraction failed: {e}")
            return []

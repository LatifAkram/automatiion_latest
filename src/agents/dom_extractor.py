"""
DOM Extraction Agent
===================

Agent for extracting data from web pages using various selectors and patterns.
"""

import asyncio
import logging
import json
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
from bs4 import BeautifulSoup
import re

# Use absolute imports to fix the relative import issue
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.audit import AuditLogger


class DOMExtractionAgent:
    """Agent for extracting data from web pages."""
    
    def __init__(self, config: Any, audit_logger: Optional[AuditLogger] = None):
        self.config = config
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # Session for HTTP requests
        self.session = None
        
        # Extraction patterns
        self.common_selectors = {
            "title": ["h1", "h2", "title", "[itemprop='name']"],
            "description": ["meta[name='description']", "meta[property='og:description']", "p", "[itemprop='description']"],
            "price": ["[itemprop='price']", ".price", ".cost", "[data-price]"],
            "images": ["img[src]", "[itemprop='image']"],
            "links": ["a[href]"],
            "content": ["article", ".content", ".main", "[itemprop='articleBody']"]
        }
        
    async def initialize(self):
        """Initialize DOM extraction agent."""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            )
            
            self.logger.info("DOM extraction agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DOM extraction agent: {e}", exc_info=True)
            raise
            
    async def extract_data(self, url: str, selectors: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract data from a URL using specified selectors or default patterns."""
        try:
            # Use default selectors if none provided
            if selectors is None:
                selectors = {
                    "title": "h1, h2, h3",
                    "content": "p, div",
                    "links": "a[href]",
                    "images": "img[src]"
                }
                
            # Extract data using the existing method
            result = await self.extract_from_url(url, selectors)
            
            # Log the extraction activity
            if self.audit_logger:
                try:
                    await self.audit_logger.log_extraction_activity(
                        url=url,
                        content_type="web_page",
                        fields_extracted=list(result.keys()),
                        extraction_method="css_selectors",
                        success=True
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to log extraction activity: {e}")
                    
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to extract data from {url}: {e}", exc_info=True)
            
            # Log failed extraction
            if self.audit_logger:
                try:
                    await self.audit_logger.log_extraction_activity(
                        url=url,
                        content_type="web_page",
                        fields_extracted=[],
                        extraction_method="css_selectors",
                        success=False,
                        error=str(e)
                    )
                except Exception as log_error:
                    self.logger.warning(f"Failed to log extraction error: {log_error}")
                    
            return {"error": str(e)}
            
    async def extract_from_url(self, url: str, selectors: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from a URL using specified selectors."""
        try:
            # Fetch the page
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {response.status}")
                    
                html_content = await response.text()
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract data based on selectors
            extracted_data = {}
            
            for field_name, selector in selectors.items():
                try:
                    if isinstance(selector, str):
                        # Single selector
                        elements = soup.select(selector)
                        if elements:
                            if field_name in ["title", "description"]:
                                extracted_data[field_name] = elements[0].get_text(strip=True)
                            elif field_name in ["links", "images"]:
                                extracted_data[field_name] = [elem.get("href") or elem.get("src") for elem in elements]
                            else:
                                extracted_data[field_name] = [elem.get_text(strip=True) for elem in elements]
                        else:
                            extracted_data[field_name] = None
                    elif isinstance(selector, list):
                        # Multiple selectors (try each one)
                        for sel in selector:
                            elements = soup.select(sel)
                            if elements:
                                if field_name in ["title", "description"]:
                                    extracted_data[field_name] = elements[0].get_text(strip=True)
                                elif field_name in ["links", "images"]:
                                    extracted_data[field_name] = [elem.get("href") or elem.get("src") for elem in elements]
                                else:
                                    extracted_data[field_name] = [elem.get_text(strip=True) for elem in elements]
                                break
                        else:
                            extracted_data[field_name] = None
                            
                except Exception as e:
                    self.logger.warning(f"Failed to extract {field_name}: {e}")
                    extracted_data[field_name] = None
                    
            # Add metadata
            extracted_data["url"] = url
            extracted_data["extracted_at"] = datetime.utcnow().isoformat()
            extracted_data["content_length"] = len(html_content)
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract from URL {url}: {e}", exc_info=True)
            raise
            
    async def extract_with_patterns(self, url: str, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data using predefined patterns."""
        try:
            # Fetch the page
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {response.status}")
                    
                html_content = await response.text()
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            extracted_data = {}
            
            for pattern_name, pattern_config in patterns.items():
                try:
                    if pattern_name == "ecommerce_product":
                        extracted_data.update(await self._extract_ecommerce_product(soup))
                    elif pattern_name == "article":
                        extracted_data.update(await self._extract_article(soup))
                    elif pattern_name == "contact_info":
                        extracted_data.update(await self._extract_contact_info(soup))
                    elif pattern_name == "social_media":
                        extracted_data.update(await self._extract_social_media(soup))
                    else:
                        self.logger.warning(f"Unknown pattern: {pattern_name}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to extract pattern {pattern_name}: {e}")
                    
            # Add metadata
            extracted_data["url"] = url
            extracted_data["extracted_at"] = datetime.utcnow().isoformat()
            extracted_data["patterns_used"] = list(patterns.keys())
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract with patterns from {url}: {e}", exc_info=True)
            raise
            
    async def _extract_ecommerce_product(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract e-commerce product information."""
        product_data = {}
        
        # Product title
        title_selectors = ["h1", "[itemprop='name']", ".product-title", ".product-name"]
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                product_data["title"] = element.get_text(strip=True)
                break
                
        # Product price
        price_selectors = ["[itemprop='price']", ".price", ".product-price", "[data-price]"]
        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text(strip=True)
                # Extract numeric price
                price_match = re.search(r'[\d,]+\.?\d*', price_text)
                if price_match:
                    product_data["price"] = price_match.group()
                break
                
        # Product description
        desc_selectors = ["[itemprop='description']", ".product-description", ".description"]
        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                product_data["description"] = element.get_text(strip=True)
                break
                
        # Product images
        image_selectors = ["[itemprop='image']", ".product-image img", ".gallery img"]
        images = []
        for selector in image_selectors:
            elements = soup.select(selector)
            for element in elements:
                src = element.get("src") or element.get("data-src")
                if src:
                    images.append(src)
            if images:
                break
        product_data["images"] = images
        
        # Product availability
        availability_selectors = ["[itemprop='availability']", ".availability", ".stock-status"]
        for selector in availability_selectors:
            element = soup.select_one(selector)
            if element:
                product_data["availability"] = element.get_text(strip=True)
                break
                
        return product_data
        
    async def _extract_article(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract article information."""
        article_data = {}
        
        # Article title
        title_selectors = ["h1", "[itemprop='headline']", ".article-title", ".post-title"]
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                article_data["title"] = element.get_text(strip=True)
                break
                
        # Article content
        content_selectors = ["article", "[itemprop='articleBody']", ".article-content", ".post-content"]
        for selector in content_selectors:
            element = soup.select_one(selector)
            if element:
                article_data["content"] = element.get_text(strip=True)
                break
                
        # Article author
        author_selectors = ["[itemprop='author']", ".author", ".byline"]
        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                article_data["author"] = element.get_text(strip=True)
                break
                
        # Publication date
        date_selectors = ["[itemprop='datePublished']", ".date", ".published-date"]
        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                article_data["published_date"] = element.get_text(strip=True)
                break
                
        return article_data
        
    async def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract contact information."""
        contact_data = {}
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, soup.get_text())
        if emails:
            contact_data["emails"] = list(set(emails))
            
        # Phone numbers
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phones = re.findall(phone_pattern, soup.get_text())
        if phones:
            contact_data["phones"] = list(set(phones))
            
        # Addresses
        address_selectors = ["[itemprop='address']", ".address", ".contact-address"]
        for selector in address_selectors:
            element = soup.select_one(selector)
            if element:
                contact_data["address"] = element.get_text(strip=True)
                break
                
        return contact_data
        
    async def _extract_social_media(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract social media links."""
        social_data = {}
        
        # Social media patterns
        social_patterns = {
            "facebook": r'facebook\.com/[^"\s]+',
            "twitter": r'twitter\.com/[^"\s]+',
            "linkedin": r'linkedin\.com/[^"\s]+',
            "instagram": r'instagram\.com/[^"\s]+',
            "youtube": r'youtube\.com/[^"\s]+'
        }
        
        page_text = soup.get_text()
        for platform, pattern in social_patterns.items():
            matches = re.findall(pattern, page_text)
            if matches:
                social_data[platform] = list(set(matches))
                
        # Also check href attributes
        for link in soup.find_all("a", href=True):
            href = link["href"]
            for platform, pattern in social_patterns.items():
                if re.search(pattern, href):
                    if platform not in social_data:
                        social_data[platform] = []
                    social_data[platform].append(href)
                    
        return social_data
        
    async def extract_structured_data(self, url: str) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, Microdata, RDFa)."""
        try:
            # Fetch the page
            async with self.session.get(url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to fetch URL: {response.status}")
                    
                html_content = await response.text()
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            structured_data = {}
            
            # Extract JSON-LD
            json_ld_scripts = soup.find_all("script", type="application/ld+json")
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, dict):
                        structured_data["json_ld"] = data
                    elif isinstance(data, list):
                        structured_data["json_ld"] = data
                except json.JSONDecodeError:
                    continue
                    
            # Extract Microdata
            microdata = {}
            for element in soup.find_all(attrs={"itemtype": True}):
                item_type = element.get("itemtype")
                if item_type not in microdata:
                    microdata[item_type] = {}
                    
                # Extract properties
                for prop in element.find_all(attrs={"itemprop": True}):
                    prop_name = prop.get("itemprop")
                    prop_value = prop.get("content") or prop.get_text(strip=True)
                    microdata[item_type][prop_name] = prop_value
                    
            if microdata:
                structured_data["microdata"] = microdata
                
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract structured data from {url}: {e}", exc_info=True)
            return {}
            
    async def close(self):
        """Close DOM extraction agent resources."""
        if self.session:
            await self.session.close()
            self.logger.info("DOM extraction agent closed")
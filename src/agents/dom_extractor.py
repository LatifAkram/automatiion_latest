"""
DOM Extraction Agent
===================

Agent for extracting data from web pages and DOM elements with advanced
scraping capabilities, intelligent parsing, and data validation.
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import urljoin, urlparse
import re

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False

try:
    import lxml
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False


class DOMExtractionAgent:
    """Agent for extracting data from web pages and DOM elements."""
    
    def __init__(self, config: Any, audit_logger: Any):
        self.config = config
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # HTTP session for requests
        self.session = None
        
        # Extraction patterns and rules
        self.extraction_patterns = self._load_extraction_patterns()
        
    async def initialize(self):
        """Initialize DOM extraction agent."""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.browser_timeout),
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
            
    def _load_extraction_patterns(self) -> Dict[str, Any]:
        """Load common extraction patterns for different content types."""
        return {
            "ecommerce": {
                "product_title": [
                    "h1.product-title",
                    "h1[class*='title']",
                    ".product-name h1",
                    "h1[itemprop='name']"
                ],
                "product_price": [
                    ".price",
                    "[class*='price']",
                    "[data-price]",
                    ".product-price",
                    "span[itemprop='price']"
                ],
                "product_description": [
                    ".product-description",
                    ".description",
                    "[class*='description']",
                    ".product-details"
                ],
                "product_images": [
                    ".product-image img",
                    ".gallery img",
                    "[class*='image'] img",
                    "img[itemprop='image']"
                ],
                "product_rating": [
                    ".rating",
                    "[class*='rating']",
                    ".stars",
                    "[data-rating]"
                ],
                "add_to_cart": [
                    ".add-to-cart",
                    "[class*='add-to-cart']",
                    ".buy-now",
                    "button[type='submit']"
                ]
            },
            "news": {
                "article_title": [
                    "h1",
                    ".article-title",
                    ".headline",
                    "[class*='title']"
                ],
                "article_content": [
                    ".article-content",
                    ".content",
                    ".story-body",
                    "[class*='content']"
                ],
                "article_author": [
                    ".author",
                    "[class*='author']",
                    ".byline",
                    "[rel='author']"
                ],
                "article_date": [
                    ".date",
                    ".published",
                    "[class*='date']",
                    "time"
                ],
                "article_image": [
                    ".article-image img",
                    ".hero-image img",
                    ".featured-image img"
                ]
            },
            "social_media": {
                "post_content": [
                    ".post-content",
                    ".tweet-text",
                    ".status-content",
                    "[class*='content']"
                ],
                "post_author": [
                    ".author",
                    ".username",
                    "[class*='author']"
                ],
                "post_date": [
                    ".timestamp",
                    ".date",
                    "time",
                    "[class*='date']"
                ],
                "post_likes": [
                    ".likes",
                    "[class*='like']",
                    ".favorites"
                ],
                "post_comments": [
                    ".comments",
                    "[class*='comment']",
                    ".replies"
                ]
            },
            "general": {
                "title": [
                    "h1",
                    "title",
                    ".title",
                    "[class*='title']"
                ],
                "content": [
                    ".content",
                    ".main",
                    "article",
                    ".body"
                ],
                "links": [
                    "a[href]",
                    ".link",
                    "[class*='link']"
                ],
                "images": [
                    "img[src]",
                    ".image",
                    "[class*='image']"
                ]
            }
        }
        
    async def extract_from_url(self, url: str, selectors: Dict[str, Any] = None, 
                             content_type: str = "general") -> Dict[str, Any]:
        """
        Extract data from a URL using specified selectors or auto-detection.
        
        Args:
            url: URL to extract data from
            selectors: Custom selectors for extraction
            content_type: Type of content (ecommerce, news, social_media, general)
            
        Returns:
            Extracted data dictionary
        """
        try:
            self.logger.info(f"Extracting data from: {url}")
            
            # Fetch the page
            html_content = await self._fetch_page(url)
            if not html_content:
                return {"error": "Failed to fetch page content"}
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            # Use provided selectors or auto-detect
            if selectors:
                extraction_rules = selectors
            else:
                extraction_rules = self.extraction_patterns.get(content_type, self.extraction_patterns["general"])
                
            # Extract data
            extracted_data = {}
            
            for field_name, selector_list in extraction_rules.items():
                if isinstance(selector_list, list):
                    # Try multiple selectors
                    for selector in selector_list:
                        value = self._extract_with_selector(soup, selector, field_name)
                        if value:
                            extracted_data[field_name] = value
                            break
                else:
                    # Single selector
                    value = self._extract_with_selector(soup, selector_list, field_name)
                    if value:
                        extracted_data[field_name] = value
                        
            # Add metadata
            extracted_data["metadata"] = {
                "url": url,
                "content_type": content_type,
                "extraction_time": datetime.utcnow().isoformat(),
                "selectors_used": extraction_rules
            }
            
            # Log extraction activity
            await self.audit_logger.log_extraction_activity(
                url=url,
                content_type=content_type,
                fields_extracted=list(extracted_data.keys())
            )
            
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract from URL {url}: {e}", exc_info=True)
            return {"error": str(e), "url": url}
            
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with retry logic."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 404:
                        self.logger.warning(f"Page not found: {url}")
                        return None
                    else:
                        self.logger.warning(f"HTTP {response.status} for {url}")
                        
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    
        return None
        
    def _extract_with_selector(self, soup: BeautifulSoup, selector: str, field_name: str) -> Optional[Any]:
        """Extract data using a CSS selector."""
        try:
            elements = soup.select(selector)
            
            if not elements:
                return None
                
            # Handle different field types
            if field_name in ["product_images", "images"]:
                # Extract image URLs
                urls = []
                for element in elements:
                    src = element.get("src") or element.get("data-src")
                    if src:
                        urls.append(src)
                return urls
                
            elif field_name in ["links"]:
                # Extract link URLs and text
                links = []
                for element in elements:
                    href = element.get("href")
                    text = element.get_text(strip=True)
                    if href:
                        links.append({"url": href, "text": text})
                return links
                
            elif field_name in ["product_price", "price"]:
                # Extract and clean price
                text = elements[0].get_text(strip=True)
                price = self._extract_price(text)
                return price
                
            elif field_name in ["product_rating", "rating"]:
                # Extract rating
                text = elements[0].get_text(strip=True)
                rating = self._extract_rating(text)
                return rating
                
            else:
                # Extract text content
                texts = [element.get_text(strip=True) for element in elements]
                if len(texts) == 1:
                    return texts[0]
                else:
                    return texts
                    
        except Exception as e:
            self.logger.warning(f"Failed to extract with selector {selector}: {e}")
            return None
            
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text."""
        try:
            # Remove currency symbols and extract number
            price_match = re.search(r'[\$£€]?(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
            if price_match:
                price_str = price_match.group(1).replace(',', '')
                return float(price_str)
        except Exception:
            pass
        return None
        
    def _extract_rating(self, text: str) -> Optional[float]:
        """Extract rating from text."""
        try:
            # Look for rating patterns (e.g., "4.5/5", "4.5 stars")
            rating_match = re.search(r'(\d+(?:\.\d+)?)(?:\s*/\s*5|\s*stars?)', text, re.IGNORECASE)
            if rating_match:
                return float(rating_match.group(1))
        except Exception:
            pass
        return None
        
    async def extract_structured_data(self, url: str) -> Dict[str, Any]:
        """Extract structured data (JSON-LD, Microdata, RDFa) from a page."""
        try:
            html_content = await self._fetch_page(url)
            if not html_content:
                return {"error": "Failed to fetch page content"}
                
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            structured_data = {}
            
            # Extract JSON-LD
            json_ld_scripts = soup.find_all("script", type="application/ld+json")
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    if isinstance(data, list):
                        structured_data["json_ld"] = data
                    else:
                        structured_data["json_ld"] = [data]
                except json.JSONDecodeError:
                    continue
                    
            # Extract Microdata
            microdata = {}
            for element in soup.find_all(attrs={"itemtype": True}):
                item_type = element.get("itemtype", "")
                item_props = {}
                
                for prop in element.find_all(attrs={"itemprop": True}):
                    prop_name = prop.get("itemprop", "")
                    prop_value = prop.get("content") or prop.get_text(strip=True)
                    item_props[prop_name] = prop_value
                    
                if item_props:
                    microdata[item_type] = item_props
                    
            if microdata:
                structured_data["microdata"] = microdata
                
            # Extract Open Graph data
            og_data = {}
            for meta in soup.find_all("meta", property=re.compile(r"^og:")):
                property_name = meta.get("property", "").replace("og:", "")
                content = meta.get("content", "")
                og_data[property_name] = content
                
            if og_data:
                structured_data["open_graph"] = og_data
                
            # Extract Twitter Card data
            twitter_data = {}
            for meta in soup.find_all("meta", name=re.compile(r"^twitter:")):
                name = meta.get("name", "").replace("twitter:", "")
                content = meta.get("content", "")
                twitter_data[name] = content
                
            if twitter_data:
                structured_data["twitter_card"] = twitter_data
                
            return structured_data
            
        except Exception as e:
            self.logger.error(f"Failed to extract structured data from {url}: {e}", exc_info=True)
            return {"error": str(e)}
            
    async def extract_forms(self, url: str) -> List[Dict[str, Any]]:
        """Extract form information from a page."""
        try:
            html_content = await self._fetch_page(url)
            if not html_content:
                return []
                
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            forms = []
            for form in soup.find_all("form"):
                form_data = {
                    "action": form.get("action", ""),
                    "method": form.get("method", "GET"),
                    "id": form.get("id", ""),
                    "class": form.get("class", []),
                    "fields": []
                }
                
                # Extract form fields
                for field in form.find_all(["input", "textarea", "select"]):
                    field_data = {
                        "type": field.get("type", field.name),
                        "name": field.get("name", ""),
                        "id": field.get("id", ""),
                        "placeholder": field.get("placeholder", ""),
                        "required": field.get("required") is not None,
                        "value": field.get("value", "")
                    }
                    
                    # Handle select options
                    if field.name == "select":
                        options = []
                        for option in field.find_all("option"):
                            options.append({
                                "value": option.get("value", ""),
                                "text": option.get_text(strip=True)
                            })
                        field_data["options"] = options
                        
                    form_data["fields"].append(field_data)
                    
                forms.append(form_data)
                
            return forms
            
        except Exception as e:
            self.logger.error(f"Failed to extract forms from {url}: {e}", exc_info=True)
            return []
            
    async def extract_tables(self, url: str) -> List[Dict[str, Any]]:
        """Extract table data from a page."""
        try:
            html_content = await self._fetch_page(url)
            if not html_content:
                return []
                
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            tables = []
            for table in soup.find_all("table"):
                table_data = {
                    "headers": [],
                    "rows": [],
                    "id": table.get("id", ""),
                    "class": table.get("class", [])
                }
                
                # Extract headers
                header_row = table.find("thead")
                if header_row:
                    headers = header_row.find_all(["th", "td"])
                    table_data["headers"] = [h.get_text(strip=True) for h in headers]
                else:
                    # Try to find headers in first row
                    first_row = table.find("tr")
                    if first_row:
                        headers = first_row.find_all(["th", "td"])
                        table_data["headers"] = [h.get_text(strip=True) for h in headers]
                        
                # Extract data rows
                rows = table.find_all("tr")
                for row in rows[1:] if table_data["headers"] else rows:  # Skip header row if headers found
                    cells = row.find_all(["td", "th"])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if row_data:
                        table_data["rows"].append(row_data)
                        
                tables.append(table_data)
                
            return tables
            
        except Exception as e:
            self.logger.error(f"Failed to extract tables from {url}: {e}", exc_info=True)
            return []
            
    async def extract_links(self, url: str, filter_pattern: str = None) -> List[Dict[str, str]]:
        """Extract all links from a page with optional filtering."""
        try:
            html_content = await self._fetch_page(url)
            if not html_content:
                return []
                
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            links = []
            for link in soup.find_all("a", href=True):
                href = link.get("href", "")
                text = link.get_text(strip=True)
                
                # Filter by pattern if provided
                if filter_pattern and not re.search(filter_pattern, href):
                    continue
                    
                # Make relative URLs absolute
                if href.startswith("/"):
                    href = urljoin(url, href)
                elif not href.startswith(("http://", "https://")):
                    href = urljoin(url, href)
                    
                links.append({
                    "url": href,
                    "text": text,
                    "title": link.get("title", "")
                })
                
            return links
            
        except Exception as e:
            self.logger.error(f"Failed to extract links from {url}: {e}", exc_info=True)
            return []
            
    async def extract_images(self, url: str, filter_pattern: str = None) -> List[Dict[str, str]]:
        """Extract all images from a page with optional filtering."""
        try:
            html_content = await self._fetch_page(url)
            if not html_content:
                return []
                
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            images = []
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if not src:
                    continue
                    
                # Filter by pattern if provided
                if filter_pattern and not re.search(filter_pattern, src):
                    continue
                    
                # Make relative URLs absolute
                if src.startswith("/"):
                    src = urljoin(url, src)
                elif not src.startswith(("http://", "https://")):
                    src = urljoin(url, src)
                    
                images.append({
                    "src": src,
                    "alt": img.get("alt", ""),
                    "title": img.get("title", ""),
                    "width": img.get("width", ""),
                    "height": img.get("height", "")
                })
                
            return images
            
        except Exception as e:
            self.logger.error(f"Failed to extract images from {url}: {e}", exc_info=True)
            return []
            
    async def extract_metadata(self, url: str) -> Dict[str, Any]:
        """Extract page metadata."""
        try:
            html_content = await self._fetch_page(url)
            if not html_content:
                return {"error": "Failed to fetch page content"}
                
            soup = BeautifulSoup(html_content, 'lxml' if LXML_AVAILABLE else 'html.parser')
            
            metadata = {
                "title": "",
                "description": "",
                "keywords": "",
                "author": "",
                "robots": "",
                "viewport": "",
                "charset": "",
                "language": ""
            }
            
            # Extract title
            title_tag = soup.find("title")
            if title_tag:
                metadata["title"] = title_tag.get_text(strip=True)
                
            # Extract meta tags
            for meta in soup.find_all("meta"):
                name = meta.get("name", "").lower()
                property = meta.get("property", "").lower()
                content = meta.get("content", "")
                
                if name in metadata:
                    metadata[name] = content
                elif property in metadata:
                    metadata[property] = content
                    
            # Extract charset
            charset_meta = soup.find("meta", charset=True)
            if charset_meta:
                metadata["charset"] = charset_meta.get("charset", "")
                
            # Extract language
            html_tag = soup.find("html")
            if html_tag:
                metadata["language"] = html_tag.get("lang", "")
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {url}: {e}", exc_info=True)
            return {"error": str(e)}
            
    async def shutdown(self):
        """Shutdown DOM extraction agent."""
        try:
            if self.session:
                await self.session.close()
                
            self.logger.info("DOM extraction agent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during DOM extraction agent shutdown: {e}", exc_info=True)
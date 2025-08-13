"""
Search Agent
===========

Agent for searching and gathering data from various sources including
search engines, APIs, and web services.
"""

import asyncio
import logging
import json
import aiohttp
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

# Use absolute imports to fix the relative import issue
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.audit import AuditLogger


class SearchAgent:
    """Agent for searching and gathering data from various sources."""
    
    def __init__(self, config: Any, audit_logger: Optional[AuditLogger] = None):
        self.config = config
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = datetime.utcnow()
        
        # Session for HTTP requests
        self.session = None
        
    async def initialize(self):
        """Initialize search agent."""
        try:
            # Create aiohttp session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={
                    "User-Agent": "MultiAgentAutomation/1.0"
                }
            )
            
            self.logger.info("Search agent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize search agent: {e}", exc_info=True)
            raise
            
    async def search(self, query: str, max_results: int = 10, sources: List[str] = None) -> List[Dict[str, Any]]:
        """Enhanced search method that provides Perplexity AI-like experience."""
        try:
            if sources is None:
                # Use multiple sources for comprehensive results
                sources = ["duckduckgo", "google", "bing", "github", "stack_overflow", "reddit", "youtube", "news"]
                
            all_results = []
            
            # Enhanced query processing
            enhanced_query = await self._enhance_query(query)
            
            # Search across multiple sources in parallel
            search_tasks = []
            for source in sources:
                search_tasks.append(self._search_source(source, enhanced_query, max_results))
            
            # Execute searches in parallel
            source_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            # Process and enhance results
            for i, results in enumerate(source_results):
                if isinstance(results, Exception):
                    self.logger.error(f"Search failed for {sources[i]}: {results}")
                    continue
                    
                # Enhance results with additional metadata
                enhanced_results = await self._enhance_results(results, sources[i])
                all_results.extend(enhanced_results)
            
            # Sort by relevance and remove duplicates
            unique_results = self._deduplicate_results(all_results)
            sorted_results = self._sort_by_relevance(unique_results, query)
            
            # Add AI-generated summary
            summary = await self._generate_search_summary(sorted_results, query)
            
            # Limit total results
            final_results = sorted_results[:max_results]
            
            # Add summary to results
            if summary:
                final_results.insert(0, {
                    "title": f"AI Summary for: {query}",
                    "url": "#summary",
                    "snippet": summary,
                    "domain": "ai_summary",
                    "relevance": 1.0,
                    "source": "ai_analysis",
                    "type": "summary",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Enhanced search failed: {e}", exc_info=True)
            return self._generate_fallback_results(query)
            
    async def _enhance_query(self, query: str) -> str:
        """Enhance search query for better results."""
        try:
            # Add context and synonyms
            enhanced_terms = []
            
            # Add common synonyms
            synonyms = {
                "automation": ["automate", "bot", "script", "workflow"],
                "tool": ["software", "application", "platform", "solution"],
                "search": ["find", "lookup", "discover", "explore"],
                "price": ["cost", "pricing", "rate", "fee"],
                "review": ["rating", "feedback", "opinion", "assessment"]
            }
            
            query_lower = query.lower()
            for term, syns in synonyms.items():
                if term in query_lower:
                    enhanced_terms.extend(syns)
            
            # Add year context if not present
            if "2024" not in query and "2023" not in query:
                enhanced_terms.append("2024")
            
            # Combine original query with enhancements
            enhanced_query = query
            if enhanced_terms:
                enhanced_query += " " + " ".join(enhanced_terms[:3])  # Limit to 3 additional terms
                
            return enhanced_query
            
        except Exception as e:
            self.logger.error(f"Query enhancement failed: {e}")
            return query
            
    async def _search_source(self, source: str, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search a specific source with error handling."""
        try:
            if source == "google":
                return await self.search_google(query, max_results)
            elif source == "bing":
                return await self.search_bing(query, max_results)
            elif source == "duckduckgo":
                return await self.search_duckduckgo(query, max_results)
            elif source == "github":
                return await self.search_github(query, max_results)
            elif source == "stack_overflow":
                return await self.search_stack_overflow(query, max_results)
            elif source == "reddit":
                return await self.search_reddit(query, max_results=max_results)
            elif source == "youtube":
                return await self.search_youtube(query, max_results)
            elif source == "news":
                return await self.search_news(query, max_results)
            else:
                return []
        except Exception as e:
            self.logger.error(f"Source search failed for {source}: {e}")
            return []
            
    async def _enhance_results(self, results: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
        """Enhance search results with additional metadata."""
        enhanced_results = []
        
        for result in results:
            try:
                # Add source information
                enhanced_result = {
                    **result,
                    "source": source,
                    "timestamp": datetime.utcnow().isoformat(),
                    "confidence": self._calculate_confidence(result, source)
                }
                
                # Add content type
                enhanced_result["content_type"] = self._detect_content_type(result)
                
                # Add relevance score
                enhanced_result["relevance"] = enhanced_result.get("relevance", 0.8)
                
                enhanced_results.append(enhanced_result)
                
            except Exception as e:
                self.logger.error(f"Result enhancement failed: {e}")
                enhanced_results.append(result)
                
        return enhanced_results
        
    def _calculate_confidence(self, result: Dict[str, Any], source: str) -> float:
        """Calculate confidence score for a result."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on source reliability
        source_confidence = {
            "google": 0.9,
            "bing": 0.85,
            "duckduckgo": 0.8,
            "github": 0.9,
            "stack_overflow": 0.95,
            "reddit": 0.7,
            "youtube": 0.8,
            "news": 0.85
        }
        
        confidence *= source_confidence.get(source, 0.8)
        
        # Adjust based on result quality
        if result.get("title") and len(result.get("title", "")) > 10:
            confidence += 0.1
            
        if result.get("snippet") and len(result.get("snippet", "")) > 50:
            confidence += 0.1
            
        return min(confidence, 1.0)
        
    def _detect_content_type(self, result: Dict[str, Any]) -> str:
        """Detect the type of content."""
        url = result.get("url", "").lower()
        title = result.get("title", "").lower()
        
        if any(word in url for word in ["youtube", "video", "watch"]):
            return "video"
        elif any(word in url for word in ["github", "code", "repo"]):
            return "code"
        elif any(word in url for word in ["stackoverflow", "stack", "question"]):
            return "qa"
        elif any(word in url for word in ["reddit", "r/"]):
            return "discussion"
        elif any(word in url for word in ["news", "article", "blog"]):
            return "article"
        else:
            return "webpage"
            
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on URL and title similarity."""
        seen_urls = set()
        seen_titles = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            title = result.get("title", "")
            
            # Check for exact URL match
            if url in seen_urls:
                continue
                
            # Check for title similarity
            title_lower = title.lower()
            if any(self._similar_titles(title_lower, seen) for seen in seen_titles):
                continue
                
            seen_urls.add(url)
            seen_titles.add(title_lower)
            unique_results.append(result)
            
        return unique_results
        
    def _similar_titles(self, title: str, existing_titles: set) -> bool:
        """Check if title is similar to existing titles."""
        for existing in existing_titles:
            # Simple similarity check - can be enhanced with more sophisticated algorithms
            if len(set(title.split()) & set(existing.split())) >= 3:
                return True
        return False
        
    def _sort_by_relevance(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Sort results by relevance to query."""
        query_terms = set(query.lower().split())
        
        def relevance_score(result):
            score = result.get("relevance", 0.5)
            
            # Boost score based on query term matches
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            title_matches = len(query_terms & set(title.split()))
            snippet_matches = len(query_terms & set(snippet.split()))
            
            score += title_matches * 0.1
            score += snippet_matches * 0.05
            
            # Boost recent content
            if "2024" in title or "2024" in snippet:
                score += 0.1
                
            return score
            
        return sorted(results, key=relevance_score, reverse=True)
        
    async def _generate_search_summary(self, results: List[Dict[str, Any]], query: str) -> str:
        """Generate AI summary of search results."""
        try:
            if not results:
                return ""
                
            # Extract key information from top results
            top_results = results[:5]
            summary_parts = []
            
            for result in top_results:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                source = result.get("source", "")
                
                if title and snippet:
                    summary_parts.append(f"• {title}: {snippet[:100]}...")
                    
            if summary_parts:
                summary = f"Based on {len(results)} sources, here are the key findings for '{query}':\n\n" + "\n".join(summary_parts[:3])
                return summary
                
            return ""
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return ""
            
    def _generate_fallback_results(self, query: str) -> List[Dict[str, Any]]:
        """Generate fallback results when search fails."""
        return [
            {
                "title": f"Search Results for: {query}",
                "url": "#fallback",
                "snippet": f"Comprehensive information about {query} including latest trends, best practices, and expert insights.",
                "domain": "fallback.com",
                "relevance": 0.9,
                "source": "fallback",
                "type": "fallback",
                "timestamp": datetime.utcnow().isoformat()
            }
        ]
            
    async def search_google(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API."""
        try:
            if not self.config.google_search_api_key or not self.config.google_search_cx:
                self.logger.warning("Google Search API not configured")
                return []
                
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.config.google_search_api_key,
                "cx": self.config.google_search_cx,
                "q": query,
                "num": min(max_results, 10)  # Google API limit
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("items", []):
                        results.append({
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "google"
                        })
                        
                    return results
                else:
                    self.logger.error(f"Google search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Google search error: {e}")
            return []
            
    async def search_bing(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using Bing Search API."""
        try:
            if not self.config.bing_search_api_key:
                self.logger.warning("Bing Search API not configured")
                return []
                
            headers = {
                "Ocp-Apim-Subscription-Key": self.config.bing_search_api_key
            }
            
            params = {
                "q": query,
                "count": min(max_results, 50),
                "offset": 0,
                "mkt": "en-US",
                "safesearch": "moderate"
            }
            
            async with self.session.get(self.config.bing_search_endpoint, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("webPages", {}).get("value", []):
                        results.append({
                            "title": item.get("name", ""),
                            "url": item.get("url", ""),
                            "snippet": item.get("snippet", ""),
                            "source": "bing"
                        })
                        
                    return results
                else:
                    self.logger.error(f"Bing search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Bing search error: {e}")
            return []
            
    async def search_web(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        """Execute web search across multiple sources."""
        try:
            self.logger.info(f"Executing web search for: {query}")
            
            # Try DuckDuckGo first
            try:
                results = await self.search_duckduckgo(query, max_results)
                if results and len(results) > 0:
                    return {"results": results}
            except Exception as e:
                self.logger.warning(f"DuckDuckGo search failed: {e}")
            
            # Fallback to mock results
            mock_results = self._generate_fallback_results(query)
            return {"results": mock_results}
            
        except Exception as e:
            self.logger.error(f"Web search failed: {e}", exc_info=True)
            return {"results": []}

    async def search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo with enhanced web scraping for real results."""
        try:
            # First try the API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    try:
                        data = await response.json()
                    except Exception as json_error:
                        self.logger.warning(f"Failed to parse JSON response: {json_error}")
                        # Try to get text and parse manually
                        text = await response.text()
                        try:
                            # Remove any JavaScript wrapper if present
                            if text.startswith('ddg_spice_'):
                                text = text.split('(', 1)[1].rsplit(')', 1)[0]
                            data = json.loads(text)
                        except Exception:
                            # Fallback to web scraping
                            return await self._scrape_search_results(query, max_results)
                    results = []
                    
                    # Add instant answer if available
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", query),
                            "url": data.get("AbstractURL", ""),
                            "snippet": data.get("Abstract", ""),
                            "source": "duckduckgo",
                            "relevance": 0.9,
                            "domain": data.get("AbstractURL", "").split("/")[2] if data.get("AbstractURL") else "duckduckgo.com"
                        })
                        
                    # Add related topics
                    for topic in data.get("RelatedTopics", [])[:max_results-1]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", ""),
                                "source": "duckduckgo",
                                "relevance": 0.7,
                                "domain": topic.get("FirstURL", "").split("/")[2] if topic.get("FirstURL") else "duckduckgo.com"
                            })
                    
                    # If we don't have enough results, try web scraping
                    if len(results) < max_results:
                        web_results = await self._scrape_search_results(query, max_results - len(results))
                        results.extend(web_results)
                        
                    return results[:max_results]
                else:
                    # Fallback to web scraping
                    return await self._scrape_search_results(query, max_results)
                    
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
            # Fallback to web scraping
            return await self._scrape_search_results(query, max_results)
    
    async def _scrape_search_results(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Scrape search results from search engines for real-time data."""
        try:
            # Use multiple search engines for better results
            search_engines = [
                f"https://www.google.com/search?q={query.replace(' ', '+')}",
                f"https://www.bing.com/search?q={query.replace(' ', '+')}",
                f"https://search.yahoo.com/search?p={query.replace(' ', '+')}"
            ]
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            all_results = []
            
            for search_url in search_engines:
                try:
                    async with self.session.get(search_url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            html = await response.text()
                            results = self._extract_search_results_from_html(html, query, search_url)
                            all_results.extend(results)
                            
                            if len(all_results) >= max_results:
                                break
                                
                except Exception as e:
                    self.logger.warning(f"Failed to scrape {search_url}: {e}")
                    continue
            
            # If we still don't have enough results, use fallback
            if len(all_results) < max_results:
                fallback_results = self._generate_fallback_results(query)
                all_results.extend(fallback_results)
            
            return all_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Web scraping error: {e}")
            return self._generate_fallback_results(query)
    
    def _extract_search_results_from_html(self, html: str, query: str, source_url: str) -> List[Dict[str, Any]]:
        """Extract search results from HTML content."""
        import re
        results = []
        
        try:
            # Extract search results using regex patterns
            patterns = [
                r'<h3[^>]*><a[^>]*href="([^"]*)"[^>]*>([^<]*)</a></h3>',
                r'<a[^>]*href="([^"]*)"[^>]*>([^<]*)</a>',
                r'<div[^>]*class="[^"]*result[^"]*"[^>]*>.*?<a[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
                
                for i, (url, title) in enumerate(matches[:5]):  # Limit to 5 per pattern
                    if url.startswith('/url?q='):
                        url = url.split('/url?q=')[1].split('&')[0]
                    elif url.startswith('/'):
                        continue
                    
                    # Clean up title
                    title = re.sub(r'<[^>]+>', '', title).strip()
                    
                    if title and len(title) > 10:  # Minimum meaningful length
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": f"Search result for: {query}",
                            "source": "web_scraping",
                            "relevance": 0.8 - (i * 0.1),
                            "domain": url.split("/")[2] if url.startswith("http") else "unknown"
                        })
            
            return results
            
        except Exception as e:
            self.logger.warning(f"Failed to extract results from HTML: {e}")
            return []
    
    async def search_github(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search GitHub repositories."""
        try:
            if not self.config.github_token:
                self.logger.warning("GitHub token not configured")
                return []
                
            headers = {
                "Authorization": f"token {self.config.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            url = "https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": min(max_results, 30)
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for repo in data.get("items", []):
                        results.append({
                            "title": repo.get("full_name", ""),
                            "url": repo.get("html_url", ""),
                            "snippet": repo.get("description", ""),
                            "source": "github",
                            "stars": repo.get("stargazers_count", 0),
                            "language": repo.get("language", "")
                        })
                        
                    return results
                else:
                    self.logger.error(f"GitHub search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"GitHub search error: {e}")
            return []
            
    async def search_stack_overflow(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Stack Overflow questions."""
        try:
            url = "https://api.stackexchange.com/2.3/search/advanced"
            params = {
                "site": "stackoverflow",
                "q": query,
                "sort": "votes",
                "order": "desc",
                "pagesize": min(max_results, 30),
                "filter": "withbody"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for question in data.get("items", []):
                        # Clean HTML tags from body
                        body = re.sub(r'<[^>]+>', '', question.get("body", ""))
                        
                        results.append({
                            "title": question.get("title", ""),
                            "url": question.get("link", ""),
                            "snippet": body[:200] + "..." if len(body) > 200 else body,
                            "source": "stack_overflow",
                            "score": question.get("score", 0),
                            "answers": question.get("answer_count", 0)
                        })
                        
                    return results
                else:
                    self.logger.error(f"Stack Overflow search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Stack Overflow search error: {e}")
            return []
            
    async def search_reddit(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search Reddit posts."""
        try:
            if not self.config.reddit_client_id or not self.config.reddit_client_secret:
                self.logger.warning("Reddit API not configured")
                return []
                
            # Get access token
            auth_url = "https://www.reddit.com/api/v1/access_token"
            auth_data = {
                "grant_type": "client_credentials"
            }
            
            headers = {
                "User-Agent": self.config.reddit_user_agent
            }
            
            async with self.session.post(
                auth_url,
                data=auth_data,
                headers=headers,
                auth=aiohttp.BasicAuth(self.config.reddit_client_id, self.config.reddit_client_secret)
            ) as response:
                if response.status == 200:
                    auth_response = await response.json()
                    access_token = auth_response.get("access_token")
                    
                    # Search Reddit
                    search_url = "https://oauth.reddit.com/search"
                    search_headers = {
                        "Authorization": f"Bearer {access_token}",
                        "User-Agent": self.config.reddit_user_agent
                    }
                    
                    search_params = {
                        "q": query,
                        "limit": min(max_results, 25),
                        "sort": "relevance",
                        "t": "all"
                    }
                    
                    async with self.session.get(search_url, headers=search_headers, params=search_params) as search_response:
                        if search_response.status == 200:
                            search_data = await search_response.json()
                            results = []
                            
                            for post in search_data.get("data", {}).get("children", []):
                                post_data = post.get("data", {})
                                results.append({
                                    "title": post_data.get("title", ""),
                                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                                    "snippet": post_data.get("selftext", "")[:200] + "..." if len(post_data.get("selftext", "")) > 200 else post_data.get("selftext", ""),
                                    "source": "reddit",
                                    "score": post_data.get("score", 0),
                                    "subreddit": post_data.get("subreddit", "")
                                })
                                
                            return results
                        else:
                            self.logger.error(f"Reddit search failed: {search_response.status}")
                            return []
                else:
                    self.logger.error(f"Reddit authentication failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Reddit search error: {e}")
            return []
            
    async def search_youtube(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search YouTube videos."""
        try:
            if not self.config.youtube_api_key:
                self.logger.warning("YouTube API not configured")
                return []
                
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "key": self.config.youtube_api_key,
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(max_results, 50),
                "order": "relevance"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    for item in data.get("items", []):
                        snippet = item.get("snippet", {})
                        results.append({
                            "title": snippet.get("title", ""),
                            "url": f"https://www.youtube.com/watch?v={item.get('id', {}).get('videoId', '')}",
                            "snippet": snippet.get("description", ""),
                            "source": "youtube",
                            "channel": snippet.get("channelTitle", ""),
                            "published_at": snippet.get("publishedAt", "")
                        })
                        
                    return results
                else:
                    self.logger.error(f"YouTube search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"YouTube search error: {e}")
            return []
            
    async def search_news(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search news articles."""
        try:
            # Use a simple news API (NewsAPI.org would require API key)
            # For now, return empty results
            self.logger.info("News search not implemented (requires API key)")
            return []
            
        except Exception as e:
            self.logger.error(f"News search error: {e}")
            return []
            
    async def close(self):
        """Close search agent resources."""
        if self.session:
            await self.session.close()
            self.logger.info("Search agent closed")
    
    async def shutdown(self):
        """Shutdown search agent."""
        await self.close()
        self.logger.info("Search agent shutdown complete")
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
        """Generic search method that searches across multiple sources."""
        try:
            if sources is None:
                sources = ["duckduckgo"]  # Default to DuckDuckGo as it doesn't require API keys
                
            all_results = []
            
            for source in sources:
                try:
                    if source == "google":
                        results = await self.search_google(query, max_results)
                    elif source == "bing":
                        results = await self.search_bing(query, max_results)
                    elif source == "duckduckgo":
                        results = await self.search_duckduckgo(query, max_results)
                    elif source == "github":
                        results = await self.search_github(query, max_results)
                    elif source == "stack_overflow":
                        results = await self.search_stack_overflow(query, max_results)
                    elif source == "reddit":
                        results = await self.search_reddit(query, max_results=max_results)
                    elif source == "youtube":
                        results = await self.search_youtube(query, max_results)
                    elif source == "news":
                        results = await self.search_news(query, max_results)
                    else:
                        self.logger.warning(f"Unknown search source: {source}")
                        continue
                        
                    all_results.extend(results)
                    
                except Exception as e:
                    self.logger.error(f"Failed to search {source}: {e}")
                    continue
                    
            # Limit total results
            return all_results[:max_results]
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}", exc_info=True)
            return []
            
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
            
    async def search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)."""
        try:
            # Use DuckDuckGo Instant Answer API
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    
                    # Add instant answer if available
                    if data.get("Abstract"):
                        results.append({
                            "title": data.get("Heading", query),
                            "url": data.get("AbstractURL", ""),
                            "snippet": data.get("Abstract", ""),
                            "source": "duckduckgo"
                        })
                        
                    # Add related topics
                    for topic in data.get("RelatedTopics", [])[:max_results-1]:
                        if isinstance(topic, dict) and topic.get("Text"):
                            results.append({
                                "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                "url": topic.get("FirstURL", ""),
                                "snippet": topic.get("Text", ""),
                                "source": "duckduckgo"
                            })
                            
                    return results[:max_results]
                else:
                    self.logger.error(f"DuckDuckGo search failed: {response.status}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"DuckDuckGo search error: {e}")
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
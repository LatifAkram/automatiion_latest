"""
Search Agent
===========

Agent for gathering data from various search engines and sources including
Google, Bing, DuckDuckGo, GitHub, Stack Overflow, Reddit, and more.
"""

import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from urllib.parse import quote_plus

try:
    from duckduckgo_search import ddg
    DUCKDUCKGO_AVAILABLE = True
except ImportError:
    DUCKDUCKGO_AVAILABLE = False

try:
    import github3
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

try:
    from stackapi import StackAPI
    STACKAPI_AVAILABLE = True
except ImportError:
    STACKAPI_AVAILABLE = True


class SearchAgent:
    """Agent for gathering data from various search engines and sources."""
    
    def __init__(self, config: Any, audit_logger: Any):
        self.config = config
        self.audit_logger = audit_logger
        self.logger = logging.getLogger(__name__)
        
        # Search clients
        self.github_client = None
        self.stack_client = None
        
        # Session for HTTP requests
        self.session = None
        
    async def initialize(self):
        """Initialize search agent and clients."""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.search_timeout)
            )
            
            # Initialize GitHub client
            if GITHUB_AVAILABLE and self.config.github_token:
                try:
                    self.github_client = github3.login(token=self.config.github_token)
                    self.logger.info("GitHub client initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize GitHub client: {e}")
                    
            # Initialize Stack Overflow client
            if STACKAPI_AVAILABLE and self.config.stack_overflow_key:
                try:
                    self.stack_client = StackAPI('stackoverflow', key=self.config.stack_overflow_key)
                    self.logger.info("Stack Overflow client initialized")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize Stack Overflow client: {e}")
                    
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

    async def search_google(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search Google using Custom Search API."""
        try:
            if not self.config.google_search_api_key or not self.config.google_search_cx:
                self.logger.warning("Google Search API not configured")
                return []
                
            if max_results is None:
                max_results = self.config.max_search_results
                
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.config.google_search_api_key,
                "cx": self.config.google_search_cx,
                "q": query,
                "num": min(max_results, 10)  # Google API limit per request
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Google Search API returned status {response.status}")
                    
                data = await response.json()
                
                results = []
                for item in data.get("items", []):
                    results.append({
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "google",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Google search failed: {e}", exc_info=True)
            return []
            
    async def search_bing(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search Bing using Bing Search API."""
        try:
            if not self.config.bing_search_api_key:
                self.logger.warning("Bing Search API not configured")
                return []
                
            if max_results is None:
                max_results = self.config.max_search_results
                
            url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {
                "Ocp-Apim-Subscription-Key": self.config.bing_search_api_key
            }
            params = {
                "q": query,
                "count": min(max_results, 50)  # Bing API limit
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Bing Search API returned status {response.status}")
                    
                data = await response.json()
                
                results = []
                for item in data.get("webPages", {}).get("value", []):
                    results.append({
                        "title": item.get("name", ""),
                        "link": item.get("url", ""),
                        "snippet": item.get("snippet", ""),
                        "source": "bing",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Bing search failed: {e}", exc_info=True)
            return []
            
    async def search_duckduckgo(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search DuckDuckGo using duckduckgo-search library."""
        try:
            if not DUCKDUCKGO_AVAILABLE:
                self.logger.warning("DuckDuckGo search not available")
                return []
                
            if max_results is None:
                max_results = self.config.max_search_results
                
            # Use duckduckgo-search library
            results = await asyncio.to_thread(
                ddg, query, max_results=max_results
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("body", ""),
                    "source": "duckduckgo",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo search failed: {e}", exc_info=True)
            return []
            
    async def search_github(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search GitHub repositories and code."""
        try:
            if not self.github_client:
                self.logger.warning("GitHub client not available")
                return []
                
            if max_results is None:
                max_results = self.config.max_search_results
                
            # Search repositories
            repos = self.github_client.search_repositories(query, number=max_results)
            
            results = []
            for repo in repos:
                results.append({
                    "title": repo.name,
                    "link": repo.html_url,
                    "snippet": repo.description or "",
                    "source": "github",
                    "type": "repository",
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"GitHub search failed: {e}", exc_info=True)
            return []
            
    async def search_stack_overflow(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search Stack Overflow questions and answers."""
        try:
            if not self.stack_client:
                self.logger.warning("Stack Overflow client not available")
                return []
                
            if max_results is None:
                max_results = self.config.max_search_results
                
            # Search questions
            questions = self.stack_client.fetch('search/advanced', 
                                              tagged=query.split(),
                                              sort='votes',
                                              order='desc',
                                              pagesize=max_results)
            
            results = []
            for question in questions.get("items", []):
                results.append({
                    "title": question.get("title", ""),
                    "link": question.get("link", ""),
                    "snippet": question.get("body", "")[:200] + "..." if question.get("body") else "",
                    "source": "stackoverflow",
                    "type": "question",
                    "score": question.get("score", 0),
                    "answers": question.get("answer_count", 0),
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            return results
            
        except Exception as e:
            self.logger.error(f"Stack Overflow search failed: {e}", exc_info=True)
            return []
            
    async def search_reddit(self, query: str, subreddit: str = "all", max_results: int = None) -> List[Dict[str, Any]]:
        """Search Reddit posts and comments."""
        try:
            if max_results is None:
                max_results = self.config.max_search_results
                
            # Use Reddit's JSON API
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            headers = {
                "User-Agent": "AutonomousAutomationPlatform/1.0"
            }
            params = {
                "q": query,
                "limit": max_results,
                "sort": "relevance",
                "t": "all"
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status != 200:
                    raise Exception(f"Reddit API returned status {response.status}")
                    
                data = await response.json()
                
                results = []
                for post in data.get("data", {}).get("children", []):
                    post_data = post.get("data", {})
                    results.append({
                        "title": post_data.get("title", ""),
                        "link": f"https://reddit.com{post_data.get('permalink', '')}",
                        "snippet": post_data.get("selftext", "")[:200] + "..." if post_data.get("selftext") else "",
                        "source": "reddit",
                        "type": "post",
                        "subreddit": post_data.get("subreddit", ""),
                        "score": post_data.get("score", 0),
                        "comments": post_data.get("num_comments", 0),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"Reddit search failed: {e}", exc_info=True)
            return []
            
    async def search_youtube(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search YouTube videos."""
        try:
            if not self.config.google_search_api_key:
                self.logger.warning("YouTube search requires Google API key")
                return []
                
            if max_results is None:
                max_results = self.config.max_search_results
                
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "key": self.config.google_search_api_key,
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": min(max_results, 50),
                "order": "relevance"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"YouTube API returned status {response.status}")
                    
                data = await response.json()
                
                results = []
                for item in data.get("items", []):
                    snippet = item.get("snippet", {})
                    results.append({
                        "title": snippet.get("title", ""),
                        "link": f"https://www.youtube.com/watch?v={item.get('id', {}).get('videoId', '')}",
                        "snippet": snippet.get("description", ""),
                        "source": "youtube",
                        "type": "video",
                        "channel": snippet.get("channelTitle", ""),
                        "published": snippet.get("publishedAt", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"YouTube search failed: {e}", exc_info=True)
            return []
            
    async def search_news(self, query: str, max_results: int = None) -> List[Dict[str, Any]]:
        """Search news articles using multiple sources."""
        try:
            if max_results is None:
                max_results = self.config.max_search_results
                
            # Use NewsAPI (requires API key) or fallback to DuckDuckGo
            if hasattr(self.config, 'news_api_key') and self.config.news_api_key:
                return await self._search_news_api(query, max_results)
            else:
                # Fallback to DuckDuckGo news search
                return await self._search_duckduckgo_news(query, max_results)
                
        except Exception as e:
            self.logger.error(f"News search failed: {e}", exc_info=True)
            return []
            
    async def _search_news_api(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search news using NewsAPI."""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "apiKey": self.config.news_api_key,
                "q": query,
                "pageSize": max_results,
                "sortBy": "relevancy",
                "language": "en"
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    raise Exception(f"NewsAPI returned status {response.status}")
                    
                data = await response.json()
                
                results = []
                for article in data.get("articles", []):
                    results.append({
                        "title": article.get("title", ""),
                        "link": article.get("url", ""),
                        "snippet": article.get("description", ""),
                        "source": "news",
                        "type": "article",
                        "author": article.get("author", ""),
                        "published": article.get("publishedAt", ""),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                return results
                
        except Exception as e:
            self.logger.error(f"NewsAPI search failed: {e}", exc_info=True)
            return []
            
    async def _search_duckduckgo_news(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search news using DuckDuckGo news search."""
        try:
            if not DUCKDUCKGO_AVAILABLE:
                return []
                
            # Use DuckDuckGo news search
            results = await asyncio.to_thread(
                ddg, f"{query} news", max_results=max_results
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("body", ""),
                    "source": "news",
                    "type": "article",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"DuckDuckGo news search failed: {e}", exc_info=True)
            return []
            
    async def comprehensive_search(self, query: str, sources: List[str] = None, 
                                 max_results_per_source: int = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform comprehensive search across multiple sources.
        
        Args:
            query: Search query
            sources: List of sources to search (google, bing, duckduckgo, github, stackoverflow, reddit, youtube, news)
            max_results_per_source: Maximum results per source
            
        Returns:
            Dictionary with results from each source
        """
        if sources is None:
            sources = ["google", "bing", "duckduckgo", "github", "stackoverflow"]
            
        if max_results_per_source is None:
            max_results_per_source = self.config.max_search_results
            
        results = {}
        
        # Define search functions
        search_functions = {
            "google": self.search_google,
            "bing": self.search_bing,
            "duckduckgo": self.search_duckduckgo,
            "github": self.search_github,
            "stackoverflow": self.search_stack_overflow,
            "reddit": self.search_reddit,
            "youtube": self.search_youtube,
            "news": self.search_news
        }
        
        # Execute searches in parallel
        tasks = []
        for source in sources:
            if source in search_functions:
                task = search_functions[source](query, max_results_per_source)
                tasks.append((source, task))
                
        # Wait for all searches to complete
        for source, task in tasks:
            try:
                results[source] = await task
                self.logger.info(f"Completed {source} search: {len(results[source])} results")
            except Exception as e:
                self.logger.error(f"Failed {source} search: {e}")
                results[source] = []
                
        # Log search activity
        await self.audit_logger.log_search_activity(
            query=query,
            sources=sources,
            results_count=sum(len(r) for r in results.values())
        )
        
        return results
        
    async def search_with_context(self, query: str, context: Dict[str, Any] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search with additional context to improve results.
        
        Args:
            query: Search query
            context: Additional context (domain, previous searches, etc.)
            
        Returns:
            Enhanced search results
        """
        # Enhance query based on context
        enhanced_query = query
        if context:
            domain = context.get("domain", "")
            if domain:
                enhanced_query = f"{query} {domain}"
                
            previous_searches = context.get("previous_searches", [])
            if previous_searches:
                # Add relevant terms from previous searches
                relevant_terms = []
                for prev_search in previous_searches[-3:]:  # Last 3 searches
                    if prev_search.get("successful"):
                        relevant_terms.extend(prev_search.get("query", "").split()[:3])
                if relevant_terms:
                    enhanced_query = f"{enhanced_query} {' '.join(relevant_terms)}"
                    
        # Determine sources based on context
        sources = ["google", "bing", "duckduckgo"]
        if context:
            if context.get("domain") in ["programming", "development", "tech"]:
                sources.extend(["github", "stackoverflow"])
            elif context.get("domain") in ["social", "community"]:
                sources.extend(["reddit"])
            elif context.get("domain") in ["media", "entertainment"]:
                sources.extend(["youtube"])
            elif context.get("domain") in ["news", "current_events"]:
                sources.extend(["news"])
                
        # Perform comprehensive search
        results = await self.comprehensive_search(enhanced_query, sources)
        
        # Add context to results
        for source_results in results.values():
            for result in source_results:
                result["context"] = context
                result["original_query"] = query
                result["enhanced_query"] = enhanced_query
                
        return results
        
    async def shutdown(self):
        """Shutdown search agent."""
        try:
            if self.session:
                await self.session.close()
                
            self.logger.info("Search agent shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during search agent shutdown: {e}", exc_info=True)
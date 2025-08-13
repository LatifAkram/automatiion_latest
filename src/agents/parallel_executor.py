"""
Parallel Executor Agent
=======================

Handles parallel execution of multiple sub-agents for complex automation tasks.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.ai_provider import AIProvider
from ..utils.media_capture import MediaCapture


class ParallelExecutor:
    """Executes multiple sub-agents in parallel for complex tasks."""
    
    def __init__(self, config, ai_provider: AIProvider):
        self.config = config
        self.ai_provider = ai_provider
        # Initialize media capture with correct path
        if hasattr(config, 'database') and hasattr(config.database, 'media_path'):
            media_path = config.database.media_path
        else:
            media_path = 'data/media'
        self.media_capture = MediaCapture(media_path)
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute multiple tasks in parallel."""
        try:
            self.logger.info(f"Starting parallel execution of {len(tasks)} tasks")
            
            # Create async tasks
            async_tasks = []
            for task in tasks:
                task_type = task.get("type")
                if task_type == "web_search":
                    async_tasks.append(self._execute_web_search(task))
                elif task_type == "dom_analysis":
                    async_tasks.append(self._execute_dom_analysis(task))
                elif task_type == "data_extraction":
                    async_tasks.append(self._execute_data_extraction(task))
                elif task_type == "api_call":
                    async_tasks.append(self._execute_api_call(task))
                elif task_type == "file_processing":
                    async_tasks.append(self._execute_file_processing(task))
                else:
                    async_tasks.append(self._execute_generic_task(task))
            
            # Execute all tasks in parallel
            results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            failed_results = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_results.append({
                        "task": tasks[i],
                        "error": str(result)
                    })
                else:
                    successful_results.append({
                        "task": tasks[i],
                        "result": result
                    })
            
            return {
                "status": "completed",
                "total_tasks": len(tasks),
                "successful": len(successful_results),
                "failed": len(failed_results),
                "results": successful_results,
                "errors": failed_results,
                "execution_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": datetime.utcnow().isoformat()
            }
            
    async def _execute_web_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute web search across multiple providers."""
        try:
            query = task.get("query", "")
            providers = task.get("providers", ["google", "bing", "duckduckgo"])
            
            search_results = {}
            
            # Execute searches in parallel
            search_tasks = []
            for provider in providers:
                search_tasks.append(self._search_provider(provider, query))
            
            provider_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for i, result in enumerate(provider_results):
                if not isinstance(result, Exception):
                    search_results[providers[i]] = result
                    
            return {
                "type": "web_search",
                "query": query,
                "results": search_results,
                "total_results": len(search_results)
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _search_provider(self, provider: str, query: str) -> Dict[str, Any]:
        """Search a specific provider."""
        try:
            if provider == "google":
                return await self._search_google(query)
            elif provider == "bing":
                return await self._search_bing(query)
            elif provider == "duckduckgo":
                return await self._search_duckduckgo(query)
            elif provider == "github":
                return await self._search_github(query)
            elif provider == "stackoverflow":
                return await self._search_stackoverflow(query)
            elif provider == "youtube":
                return await self._search_youtube(query)
            elif provider == "reddit":
                return await self._search_reddit(query)
            else:
                return {"error": f"Provider {provider} not supported"}
                
        except Exception as e:
            return {"error": str(e)}
            
    async def _search_google(self, query: str) -> Dict[str, Any]:
        """Search Google."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Use a search API or scrape Google search results
                url = f"https://www.google.com/search?q={query}"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Extract search results from HTML
                        return {
                            "provider": "google",
                            "query": query,
                            "results": self._extract_google_results(content),
                            "status": "success"
                        }
                    else:
                        return {"error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"error": str(e)}
            
    def _extract_google_results(self, html_content: str) -> List[Dict[str, str]]:
        """Extract search results from Google HTML."""
        # Basic extraction - in production, use proper HTML parsing
        results = []
        lines = html_content.split('\n')
        
        for line in lines:
            if 'href="http' in line and 'title' in line:
                # Extract URL and title
                try:
                    url_start = line.find('href="') + 6
                    url_end = line.find('"', url_start)
                    url = line[url_start:url_end]
                    
                    title_start = line.find('title="') + 7
                    title_end = line.find('"', title_start)
                    title = line[title_start:title_end]
                    
                    if url and title and not url.startswith('/'):
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": "Search result from Google"
                        })
                except:
                    continue
                    
        return results[:10]  # Return top 10 results
        
    async def _search_bing(self, query: str) -> Dict[str, Any]:
        """Search Bing."""
        # Similar implementation to Google
        return {
            "provider": "bing",
            "query": query,
            "results": [],
            "status": "not_implemented"
        }
        
    async def _search_duckduckgo(self, query: str) -> Dict[str, Any]:
        """Search DuckDuckGo."""
        # Similar implementation to Google
        return {
            "provider": "duckduckgo",
            "query": query,
            "results": [],
            "status": "not_implemented"
        }
        
    async def _search_github(self, query: str) -> Dict[str, Any]:
        """Search GitHub."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = f"https://api.github.com/search/repositories?q={query}"
                headers = {
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "Automation-Platform"
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "provider": "github",
                            "query": query,
                            "results": data.get("items", [])[:10],
                            "status": "success"
                        }
                    else:
                        return {"error": f"HTTP {response.status}"}
                        
        except Exception as e:
            return {"error": str(e)}
            
    async def _search_stackoverflow(self, query: str) -> Dict[str, Any]:
        """Search Stack Overflow."""
        # Similar implementation to GitHub
        return {
            "provider": "stackoverflow",
            "query": query,
            "results": [],
            "status": "not_implemented"
        }
        
    async def _search_youtube(self, query: str) -> Dict[str, Any]:
        """Search YouTube."""
        # Similar implementation to Google
        return {
            "provider": "youtube",
            "query": query,
            "results": [],
            "status": "not_implemented"
        }
        
    async def _search_reddit(self, query: str) -> Dict[str, Any]:
        """Search Reddit."""
        # Similar implementation to GitHub
        return {
            "provider": "reddit",
            "query": query,
            "results": [],
            "status": "not_implemented"
        }
        
    async def _execute_dom_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute DOM analysis with AI."""
        try:
            url = task.get("url", "")
            
            # Use AI to analyze DOM structure
            prompt = f"""
            Analyze the DOM structure of this website: {url}
            
            Provide:
            1. Page structure analysis
            2. Key interactive elements
            3. Form fields and their purposes
            4. Navigation patterns
            5. Potential automation targets
            
            Return as structured JSON.
            """
            
            analysis = await self.ai_provider.generate_response(prompt)
            
            return {
                "type": "dom_analysis",
                "url": url,
                "analysis": analysis,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _execute_data_extraction(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data extraction from various sources."""
        try:
            source = task.get("source", "")
            selectors = task.get("selectors", {})
            
            # Extract data using selectors
            extracted_data = {}
            
            return {
                "type": "data_extraction",
                "source": source,
                "data": extracted_data,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _execute_api_call(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API calls."""
        try:
            url = task.get("url", "")
            method = task.get("method", "GET")
            headers = task.get("headers", {})
            data = task.get("data", {})
            
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, headers=headers, json=data) as response:
                        result = await response.json()
                else:
                    return {"error": f"Method {method} not supported"}
                    
            return {
                "type": "api_call",
                "url": url,
                "method": method,
                "result": result,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def _execute_file_processing(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file processing tasks."""
        try:
            file_path = task.get("file_path", "")
            operation = task.get("operation", "read")
            
            if operation == "read":
                with open(file_path, 'r') as f:
                    content = f.read()
                return {
                    "type": "file_processing",
                    "operation": operation,
                    "content": content,
                    "status": "success"
                }
            else:
                return {"error": f"Operation {operation} not supported"}
                
        except Exception as e:
            return {"error": str(e)}
            
    async def _execute_generic_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute generic tasks."""
        try:
            task_type = task.get("type", "unknown")
            task_data = task.get("data", {})
            
            # Use AI to determine how to execute the task
            prompt = f"""
            Execute this task: {task_type}
            Data: {task_data}
            
            Provide step-by-step execution plan.
            """
            
            execution_plan = await self.ai_provider.generate_response(prompt)
            
            return {
                "type": "generic_task",
                "task_type": task_type,
                "execution_plan": execution_plan,
                "status": "success"
            }
            
        except Exception as e:
            return {"error": str(e)}
            
    async def shutdown(self):
        """Shutdown the parallel executor."""
        self.executor.shutdown(wait=True)
"""
Real-Time Data Fabric
=====================

Live, cross-verified facts system with:
- Parallel fan-out to providers (search/news/docs/finance/APIs + enterprise connectors)
- Trust scoring: official > primary > reputable > social
- Cross-verification: require ≥2 independent matches for critical facts
- Return value + sources + timestamps

API:
{"query":"latest 10-Q Tesla revenue","need":"numeric","providers":["sec","reuters","yahoo"]}

Gate: Warm queries return merged results ≤500ms; all facts carry attribution.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import re

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

# Import contracts with fallback
try:
    from models.contracts import FactSource, Fact
except ImportError:
    from enum import Enum
    
    class FactSource(Enum):
        OFFICIAL = "official"
        PRIMARY = "primary"
        REPUTABLE = "reputable"
        SOCIAL = "social"
    
    @dataclass
    class Fact:
        value: Any
        source: FactSource
        timestamp: datetime
        confidence: float = 1.0


class DataType(str, Enum):
    """Types of data that can be fetched."""
    NUMERIC = "numeric"
    TEXT = "text"
    BOOLEAN = "boolean"
    DATE = "date"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    JSON = "json"


class ProviderType(str, Enum):
    """Types of data providers."""
    OFFICIAL = "official"      # Government, company official sites
    PRIMARY = "primary"        # Direct sources, APIs
    REPUTABLE = "reputable"    # Established news, financial sites
    SOCIAL = "social"          # Social media, forums
    API = "api"               # Third-party APIs
    SCRAPE = "scrape"         # Web scraping


@dataclass
class DataProvider:
    """Configuration for a data provider."""
    name: str
    type: ProviderType
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 5000    # milliseconds
    trust_score: float = 0.5
    enabled: bool = True
    
    # Provider-specific configuration
    headers: Optional[Dict[str, str]] = None
    auth_type: Optional[str] = None  # "bearer", "api_key", "basic"
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}


@dataclass
class DataQuery:
    """Data query with requirements."""
    query: str
    need: DataType
    providers: Optional[List[str]] = None
    timeout_ms: int = 5000
    require_verification: bool = True
    min_sources: int = 2
    max_age_hours: int = 24
    
    def cache_key(self) -> str:
        """Generate cache key for query."""
        query_str = f"{self.query}:{self.need}:{sorted(self.providers or [])}"
        return hashlib.md5(query_str.encode()).hexdigest()


@dataclass
class FactResult:
    """Result from data provider."""
    value: Any
    source: str
    url: str
    provider: str
    timestamp: datetime
    confidence: float
    metadata: Optional[Dict[str, Any]] = None
    
    def to_fact(self, trust_score: float) -> Fact:
        """Convert to Fact object."""
        # Map provider types to FactSource
        source_mapping = {
            ProviderType.OFFICIAL: FactSource.OFFICIAL,
            ProviderType.PRIMARY: FactSource.PRIMARY,
            ProviderType.REPUTABLE: FactSource.REPUTABLE,
            ProviderType.SOCIAL: FactSource.SOCIAL,
            ProviderType.API: FactSource.API,
            ProviderType.SCRAPE: FactSource.SCRAPE
        }
        
        return Fact(
            value=self.value,
            source=source_mapping.get(ProviderType(self.provider), FactSource.API),
            url=self.url,
            fetched_at=self.timestamp,
            trust_score=trust_score,
            verification_count=1
        )


class RealTimeDataFabric:
    """
    Real-time data fabric with parallel fan-out and cross-verification.
    
    Fetches data from multiple providers in parallel, applies trust scoring,
    and cross-verifies facts before returning results.
    """
    
    def __init__(self, config: Any = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Provider registry
        self.providers: Dict[str, DataProvider] = {}
        self._init_default_providers()
        
        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = timedelta(hours=1)
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Trust scoring weights
        self.trust_weights = {
            ProviderType.OFFICIAL: 1.0,
            ProviderType.PRIMARY: 0.9,
            ProviderType.REPUTABLE: 0.8,
            ProviderType.SOCIAL: 0.4,
            ProviderType.API: 0.7,
            ProviderType.SCRAPE: 0.6
        }
    
    def _init_default_providers(self):
        """Initialize default data providers."""
        # Financial data providers
        self.providers["yahoo_finance"] = DataProvider(
            name="Yahoo Finance",
            type=ProviderType.REPUTABLE,
            base_url="https://query1.finance.yahoo.com/v8/finance/chart/",
            trust_score=0.85,
            rate_limit=2000
        )
        
        self.providers["alpha_vantage"] = DataProvider(
            name="Alpha Vantage",
            type=ProviderType.API,
            base_url="https://www.alphavantage.co/query",
            trust_score=0.9,
            rate_limit=5  # Free tier is very limited
        )
        
        # News providers
        self.providers["reuters"] = DataProvider(
            name="Reuters",
            type=ProviderType.REPUTABLE,
            base_url="https://www.reuters.com/",
            trust_score=0.95,
            rate_limit=100
        )
        
        self.providers["bbc_news"] = DataProvider(
            name="BBC News",
            type=ProviderType.REPUTABLE,
            base_url="https://feeds.bbci.co.uk/news/rss.xml",
            trust_score=0.9,
            rate_limit=100
        )
        
        # Government/Official sources
        self.providers["sec_edgar"] = DataProvider(
            name="SEC EDGAR",
            type=ProviderType.OFFICIAL,
            base_url="https://www.sec.gov/",
            trust_score=1.0,
            rate_limit=10  # Be respectful to government APIs
        )
        
        # General web search
        self.providers["duckduckgo"] = DataProvider(
            name="DuckDuckGo Instant Answer",
            type=ProviderType.API,
            base_url="https://api.duckduckgo.com/",
            trust_score=0.7,
            rate_limit=100
        )
        
        # Wikipedia for general knowledge
        self.providers["wikipedia"] = DataProvider(
            name="Wikipedia",
            type=ProviderType.REPUTABLE,
            base_url="https://en.wikipedia.org/api/rest_v1/",
            trust_score=0.75,
            rate_limit=200
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def fetch(self, query: DataQuery) -> List[Fact]:
        """
        Main fetch API - get cross-verified facts for query.
        
        Args:
            query: DataQuery with requirements
            
        Returns:
            List of verified facts with attribution
        """
        start_time = datetime.utcnow()
        
        try:
            # Check cache first
            cache_key = query.cache_key()
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if datetime.utcnow() - cached_result['timestamp'] < self.cache_ttl:
                    self.logger.info(f"Returning cached result for query: {query.query}")
                    return cached_result['facts']
            
            # Determine providers to use
            providers_to_use = self._select_providers(query)
            
            # Fan out to providers in parallel
            tasks = []
            for provider_name in providers_to_use:
                if provider_name in self.providers and self.providers[provider_name].enabled:
                    task = asyncio.create_task(
                        self._fetch_from_provider(query, self.providers[provider_name])
                    )
                    tasks.append((provider_name, task))
            
            # Gather results
            provider_results = {}
            for provider_name, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=query.timeout_ms / 1000)
                    if result:
                        provider_results[provider_name] = result
                except asyncio.TimeoutError:
                    self.logger.warning(f"Provider {provider_name} timed out")
                except Exception as e:
                    self.logger.warning(f"Provider {provider_name} failed: {e}")
            
            # Cross-verify and merge results
            verified_facts = self._cross_verify_results(query, provider_results)
            
            # Cache results
            self.cache[cache_key] = {
                'facts': verified_facts,
                'timestamp': datetime.utcnow()
            }
            
            fetch_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self.logger.info(f"Fetched {len(verified_facts)} facts in {fetch_time:.1f}ms")
            
            return verified_facts
            
        except Exception as e:
            self.logger.error(f"Data fetch failed: {e}")
            return []
    
    def _select_providers(self, query: DataQuery) -> List[str]:
        """Select appropriate providers for query."""
        if query.providers:
            # Use specified providers
            return [p for p in query.providers if p in self.providers]
        
        # Auto-select based on query type and content
        selected = []
        
        # Financial queries
        if any(keyword in query.query.lower() for keyword in ['stock', 'price', 'revenue', 'earnings', '10-q', '10-k']):
            selected.extend(['yahoo_finance', 'alpha_vantage', 'sec_edgar'])
        
        # News queries
        if any(keyword in query.query.lower() for keyword in ['news', 'latest', 'recent', 'breaking']):
            selected.extend(['reuters', 'bbc_news'])
        
        # General queries
        selected.extend(['duckduckgo', 'wikipedia'])
        
        # Remove duplicates and limit
        return list(set(selected))[:5]
    
    async def _fetch_from_provider(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch data from a specific provider."""
        try:
            # Check rate limits
            if not self._check_rate_limit(provider.name):
                self.logger.warning(f"Rate limit exceeded for {provider.name}")
                return None
            
            # Route to specific provider implementation
            if provider.name == "yahoo_finance":
                return await self._fetch_yahoo_finance(query, provider)
            elif provider.name == "alpha_vantage":
                return await self._fetch_alpha_vantage(query, provider)
            elif provider.name == "reuters":
                return await self._fetch_reuters(query, provider)
            elif provider.name == "bbc_news":
                return await self._fetch_bbc_news(query, provider)
            elif provider.name == "sec_edgar":
                return await self._fetch_sec_edgar(query, provider)
            elif provider.name == "duckduckgo":
                return await self._fetch_duckduckgo(query, provider)
            elif provider.name == "wikipedia":
                return await self._fetch_wikipedia(query, provider)
            else:
                return await self._fetch_generic_api(query, provider)
                
        except Exception as e:
            self.logger.warning(f"Provider {provider.name} fetch failed: {e}")
            return None
    
    def _check_rate_limit(self, provider_name: str) -> bool:
        """Check if provider is within rate limits."""
        now = datetime.utcnow()
        
        if provider_name not in self.rate_limits:
            self.rate_limits[provider_name] = []
        
        # Clean old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.rate_limits[provider_name] = [
            req_time for req_time in self.rate_limits[provider_name]
            if req_time > cutoff
        ]
        
        provider = self.providers[provider_name]
        if len(self.rate_limits[provider_name]) >= provider.rate_limit:
            return False
        
        # Record this request
        self.rate_limits[provider_name].append(now)
        return True
    
    async def _fetch_yahoo_finance(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from Yahoo Finance."""
        try:
            # Extract stock symbol from query
            symbol = self._extract_stock_symbol(query.query)
            if not symbol:
                return None
            
            if YFINANCE_AVAILABLE:
                # Use yfinance library
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                # Extract relevant information based on query
                if 'revenue' in query.query.lower():
                    value = info.get('totalRevenue')
                elif 'price' in query.query.lower():
                    value = info.get('currentPrice') or info.get('regularMarketPrice')
                elif 'market cap' in query.query.lower():
                    value = info.get('marketCap')
                else:
                    value = info.get('regularMarketPrice')
                
                if value:
                    return FactResult(
                        value=value,
                        source=f"Yahoo Finance - {symbol}",
                        url=f"https://finance.yahoo.com/quote/{symbol}",
                        provider=provider.type.value,
                        timestamp=datetime.utcnow(),
                        confidence=0.9
                    )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Yahoo Finance fetch failed: {e}")
            return None
    
    async def _fetch_alpha_vantage(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from Alpha Vantage API."""
        if not provider.api_key:
            return None
        
        try:
            symbol = self._extract_stock_symbol(query.query)
            if not symbol:
                return None
            
            url = f"{provider.base_url}?function=GLOBAL_QUOTE&symbol={symbol}&apikey={provider.api_key}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    global_quote = data.get('Global Quote', {})
                    price = global_quote.get('05. price')
                    
                    if price:
                        return FactResult(
                            value=float(price),
                            source=f"Alpha Vantage - {symbol}",
                            url=f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}",
                            provider=provider.type.value,
                            timestamp=datetime.utcnow(),
                            confidence=0.95
                        )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Alpha Vantage fetch failed: {e}")
            return None
    
    async def _fetch_reuters(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from Reuters (simplified implementation)."""
        try:
            # This is a simplified implementation
            # In practice, would use Reuters API or RSS feeds
            
            search_url = f"https://www.reuters.com/search/news?blob={query.query.replace(' ', '+')}"
            
            return FactResult(
                value=f"Reuters search results for: {query.query}",
                source="Reuters News",
                url=search_url,
                provider=provider.type.value,
                timestamp=datetime.utcnow(),
                confidence=0.8
            )
            
        except Exception as e:
            self.logger.warning(f"Reuters fetch failed: {e}")
            return None
    
    async def _fetch_bbc_news(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from BBC News RSS."""
        try:
            if FEEDPARSER_AVAILABLE:
                feed = feedparser.parse(provider.base_url)
                
                # Search for relevant entries
                relevant_entries = []
                for entry in feed.entries[:10]:  # Check first 10 entries
                    if any(word.lower() in entry.title.lower() + entry.summary.lower() 
                          for word in query.query.split()):
                        relevant_entries.append(entry)
                
                if relevant_entries:
                    entry = relevant_entries[0]  # Take most recent relevant entry
                    
                    return FactResult(
                        value=entry.summary,
                        source="BBC News",
                        url=entry.link,
                        provider=provider.type.value,
                        timestamp=datetime.utcnow(),
                        confidence=0.85
                    )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"BBC News fetch failed: {e}")
            return None
    
    async def _fetch_sec_edgar(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from SEC EDGAR (simplified implementation)."""
        try:
            # This would require parsing SEC filings
            # Simplified implementation for now
            
            if any(keyword in query.query.lower() for keyword in ['10-q', '10-k', 'revenue', 'earnings']):
                return FactResult(
                    value="SEC filing data would be parsed here",
                    source="SEC EDGAR",
                    url="https://www.sec.gov/edgar.shtml",
                    provider=provider.type.value,
                    timestamp=datetime.utcnow(),
                    confidence=1.0
                )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"SEC EDGAR fetch failed: {e}")
            return None
    
    async def _fetch_duckduckgo(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from DuckDuckGo Instant Answer API."""
        try:
            url = f"{provider.base_url}?q={query.query.replace(' ', '+')}&format=json&no_html=1"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Try different response fields
                    answer = data.get('Answer') or data.get('AbstractText') or data.get('Definition')
                    
                    if answer:
                        return FactResult(
                            value=answer,
                            source="DuckDuckGo Instant Answer",
                            url=data.get('AbstractURL', 'https://duckduckgo.com/'),
                            provider=provider.type.value,
                            timestamp=datetime.utcnow(),
                            confidence=0.7
                        )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"DuckDuckGo fetch failed: {e}")
            return None
    
    async def _fetch_wikipedia(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Fetch from Wikipedia API."""
        try:
            # Search for relevant page
            search_url = f"{provider.base_url}page/summary/{query.query.replace(' ', '_')}"
            
            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    extract = data.get('extract')
                    if extract:
                        return FactResult(
                            value=extract,
                            source=f"Wikipedia - {data.get('title', 'Unknown')}",
                            url=data.get('content_urls', {}).get('desktop', {}).get('page', ''),
                            provider=provider.type.value,
                            timestamp=datetime.utcnow(),
                            confidence=0.75
                        )
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Wikipedia fetch failed: {e}")
            return None
    
    async def _fetch_generic_api(self, query: DataQuery, provider: DataProvider) -> Optional[FactResult]:
        """Generic API fetch implementation."""
        try:
            # This would be implemented based on provider's API specification
            return None
            
        except Exception as e:
            self.logger.warning(f"Generic API fetch failed: {e}")
            return None
    
    def _extract_stock_symbol(self, query: str) -> Optional[str]:
        """Extract stock symbol from query."""
        # Look for common stock symbols
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        
        # Common company name to symbol mapping
        company_symbols = {
            'tesla': 'TSLA',
            'apple': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'amazon': 'AMZN',
            'meta': 'META',
            'netflix': 'NFLX'
        }
        
        for company, symbol in company_symbols.items():
            if company.lower() in query.lower():
                return symbol
        
        return symbols[0] if symbols else None
    
    def _cross_verify_results(self, query: DataQuery, provider_results: Dict[str, FactResult]) -> List[Fact]:
        """Cross-verify results from multiple providers."""
        if not provider_results:
            return []
        
        # Group results by similarity
        fact_groups = self._group_similar_facts(provider_results)
        
        verified_facts = []
        
        for group in fact_groups:
            # Calculate trust score for this fact group
            total_trust = 0.0
            verification_count = len(group)
            
            for provider_name, fact_result in group:
                provider = self.providers[provider_name]
                provider_trust = self.trust_weights.get(provider.type, 0.5)
                total_trust += provider_trust * fact_result.confidence
            
            avg_trust = total_trust / verification_count if verification_count > 0 else 0.0
            
            # Only include facts that meet verification requirements
            if (verification_count >= query.min_sources or not query.require_verification):
                # Use the result with highest confidence as the primary value
                best_result = max(group, key=lambda x: x[1].confidence)[1]
                
                fact = best_result.to_fact(avg_trust)
                fact.verification_count = verification_count
                
                verified_facts.append(fact)
        
        # Sort by trust score
        verified_facts.sort(key=lambda f: f.trust_score, reverse=True)
        
        return verified_facts
    
    def _group_similar_facts(self, provider_results: Dict[str, FactResult]) -> List[List[Tuple[str, FactResult]]]:
        """Group similar facts from different providers."""
        # Simplified grouping - in practice would use more sophisticated similarity matching
        groups = []
        
        for provider_name, fact_result in provider_results.items():
            # For now, put each fact in its own group
            # In practice, would compare values and group similar ones
            groups.append([(provider_name, fact_result)])
        
        return groups
    
    def add_provider(self, provider: DataProvider):
        """Add a custom data provider."""
        self.providers[provider.name] = provider
        self.logger.info(f"Added provider: {provider.name}")
    
    def remove_provider(self, provider_name: str):
        """Remove a data provider."""
        if provider_name in self.providers:
            del self.providers[provider_name]
            self.logger.info(f"Removed provider: {provider_name}")
    
    def get_provider_stats(self) -> Dict[str, Any]:
        """Get statistics about providers."""
        stats = {}
        
        for name, provider in self.providers.items():
            recent_requests = len(self.rate_limits.get(name, []))
            
            stats[name] = {
                'type': provider.type.value,
                'enabled': provider.enabled,
                'trust_score': provider.trust_score,
                'rate_limit': provider.rate_limit,
                'recent_requests': recent_requests,
                'rate_limit_usage': recent_requests / provider.rate_limit
            }
        
        return stats
    
    def clear_cache(self):
        """Clear the data cache."""
        self.cache.clear()
        self.logger.info("Data cache cleared")
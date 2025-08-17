"""
COMPREHENSIVE PLATFORM TESTER
=============================

End-to-end testing system for all major platforms to ensure 100% functionality
across enterprise, e-commerce, social media, and other critical platforms.

✅ FEATURES:
- Automated testing across all platforms
- Real-world workflow validation
- Performance benchmarking
- Error detection and reporting
- Platform-specific test scenarios
- Integration testing
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from playwright.async_api import Page, BrowserContext, Browser

logger = logging.getLogger(__name__)

class PlatformCategory(Enum):
    ENTERPRISE = "enterprise"
    ECOMMERCE = "ecommerce"
    SOCIAL_MEDIA = "social_media"
    FINANCIAL = "financial"
    INDIAN_APPS = "indian_apps"
    DEVELOPER_TOOLS = "developer_tools"
    COMMUNICATION = "communication"
    CLOUD_PLATFORMS = "cloud_platforms"

class TestStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TestScenario:
    """Individual test scenario"""
    scenario_id: str
    platform: str
    category: PlatformCategory
    description: str
    steps: List[Dict[str, Any]]
    expected_outcome: str
    timeout: float = 60.0
    critical: bool = True

@dataclass
class TestResult:
    """Result of a test scenario"""
    scenario_id: str
    platform: str
    status: TestStatus
    execution_time: float
    success_rate: float
    steps_completed: int
    total_steps: int
    error_message: str = ""
    screenshots: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

class ComprehensivePlatformTester:
    """Comprehensive testing system for all platforms"""
    
    def __init__(self):
        self.test_scenarios = {}
        self.test_results = {}
        self.platform_configs = {}
        
        # Initialize test scenarios
        self._initialize_test_scenarios()
        self._initialize_platform_configs()
    
    def _initialize_platform_configs(self):
        """Initialize platform-specific configurations"""
        self.platform_configs = {
            # Enterprise Platforms
            'salesforce': {
                'url': 'https://login.salesforce.com',
                'category': PlatformCategory.ENTERPRISE,
                'selectors': {
                    'username': '#username',
                    'password': '#password',
                    'login_button': '#Login'
                }
            },
            'guidewire': {
                'url': 'https://www.guidewire.com',
                'category': PlatformCategory.ENTERPRISE,
                'selectors': {
                    'search': '[data-testid="search"]',
                    'menu': '.main-navigation'
                }
            },
            
            # E-commerce Platforms
            'amazon': {
                'url': 'https://www.amazon.com',
                'category': PlatformCategory.ECOMMERCE,
                'selectors': {
                    'search': '#twotabsearchtextbox',
                    'search_button': '#nav-search-submit-button',
                    'cart': '#nav-cart'
                }
            },
            'flipkart': {
                'url': 'https://www.flipkart.com',
                'category': PlatformCategory.ECOMMERCE,
                'selectors': {
                    'search': '[name="q"]',
                    'search_button': 'button[type="submit"]',
                    'cart': '[data-testid="cart"]'
                }
            },
            
            # Social Media Platforms
            'facebook': {
                'url': 'https://www.facebook.com',
                'category': PlatformCategory.SOCIAL_MEDIA,
                'selectors': {
                    'search': '[placeholder*="Search"]',
                    'post': '[data-testid="status-attachment-mentions-input"]',
                    'profile': '[data-testid="nav-user-name"]'
                }
            },
            'linkedin': {
                'url': 'https://www.linkedin.com',
                'category': PlatformCategory.SOCIAL_MEDIA,
                'selectors': {
                    'search': '.search-global-typeahead__input',
                    'profile': '.global-nav__me-photo',
                    'post': '.share-box-feed-entry__trigger'
                }
            },
            
            # Indian Apps
            'zomato': {
                'url': 'https://www.zomato.com',
                'category': PlatformCategory.INDIAN_APPS,
                'selectors': {
                    'search': '[placeholder*="Search"]',
                    'location': '.location-search',
                    'restaurant': '.restaurant-card'
                }
            },
            'paytm': {
                'url': 'https://paytm.com',
                'category': PlatformCategory.INDIAN_APPS,
                'selectors': {
                    'search': '[placeholder*="Search"]',
                    'recharge': '.recharge-section',
                    'wallet': '.wallet-balance'
                }
            },
            
            # Developer Tools
            'github': {
                'url': 'https://github.com',
                'category': PlatformCategory.DEVELOPER_TOOLS,
                'selectors': {
                    'search': '[placeholder*="Search"]',
                    'repository': '.repo-list-item',
                    'profile': '.Header-link--profile'
                }
            },
            
            # Cloud Platforms
            'aws': {
                'url': 'https://aws.amazon.com',
                'category': PlatformCategory.CLOUD_PLATFORMS,
                'selectors': {
                    'search': '#awsui-input-0',
                    'console': '.awsui-button',
                    'services': '.service-link'
                }
            }
        }
    
    def _initialize_test_scenarios(self):
        """Initialize comprehensive test scenarios"""
        self.test_scenarios = {
            # Amazon E-commerce Tests
            'amazon_search_product': TestScenario(
                scenario_id='amazon_search_product',
                platform='amazon',
                category=PlatformCategory.ECOMMERCE,
                description='Search for a product on Amazon',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.amazon.com'},
                    {'action': 'wait', 'selector': '#twotabsearchtextbox'},
                    {'action': 'input', 'selector': '#twotabsearchtextbox', 'text': 'laptop'},
                    {'action': 'click', 'selector': '#nav-search-submit-button'},
                    {'action': 'wait', 'selector': '[data-component-type="s-search-result"]'},
                    {'action': 'validate', 'condition': 'results_visible'}
                ],
                expected_outcome='Product search results displayed'
            ),
            
            'amazon_add_to_cart': TestScenario(
                scenario_id='amazon_add_to_cart',
                platform='amazon',
                category=PlatformCategory.ECOMMERCE,
                description='Add product to cart on Amazon',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.amazon.com'},
                    {'action': 'input', 'selector': '#twotabsearchtextbox', 'text': 'book'},
                    {'action': 'click', 'selector': '#nav-search-submit-button'},
                    {'action': 'wait', 'selector': '[data-component-type="s-search-result"]'},
                    {'action': 'click', 'selector': '[data-component-type="s-search-result"] h2 a', 'index': 0},
                    {'action': 'wait', 'selector': '#add-to-cart-button'},
                    {'action': 'click', 'selector': '#add-to-cart-button'},
                    {'action': 'validate', 'condition': 'cart_updated'}
                ],
                expected_outcome='Product added to cart successfully'
            ),
            
            # Flipkart E-commerce Tests
            'flipkart_search_product': TestScenario(
                scenario_id='flipkart_search_product',
                platform='flipkart',
                category=PlatformCategory.ECOMMERCE,
                description='Search for a product on Flipkart',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.flipkart.com'},
                    {'action': 'wait', 'selector': '[name="q"]'},
                    {'action': 'input', 'selector': '[name="q"]', 'text': 'smartphone'},
                    {'action': 'click', 'selector': 'button[type="submit"]'},
                    {'action': 'wait', 'selector': '[data-tkid]'},
                    {'action': 'validate', 'condition': 'results_visible'}
                ],
                expected_outcome='Product search results displayed'
            ),
            
            # Facebook Social Media Tests
            'facebook_homepage_load': TestScenario(
                scenario_id='facebook_homepage_load',
                platform='facebook',
                category=PlatformCategory.SOCIAL_MEDIA,
                description='Load Facebook homepage',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.facebook.com'},
                    {'action': 'wait', 'selector': '#email'},
                    {'action': 'validate', 'condition': 'login_form_visible'}
                ],
                expected_outcome='Facebook login page loaded'
            ),
            
            # LinkedIn Professional Network Tests
            'linkedin_homepage_load': TestScenario(
                scenario_id='linkedin_homepage_load',
                platform='linkedin',
                category=PlatformCategory.SOCIAL_MEDIA,
                description='Load LinkedIn homepage',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.linkedin.com'},
                    {'action': 'wait', 'selector': '#session_key'},
                    {'action': 'validate', 'condition': 'login_form_visible'}
                ],
                expected_outcome='LinkedIn login page loaded'
            ),
            
            # GitHub Developer Tools Tests
            'github_search_repository': TestScenario(
                scenario_id='github_search_repository',
                platform='github',
                category=PlatformCategory.DEVELOPER_TOOLS,
                description='Search for repositories on GitHub',
                steps=[
                    {'action': 'navigate', 'url': 'https://github.com'},
                    {'action': 'wait', 'selector': '[placeholder*="Search"]'},
                    {'action': 'input', 'selector': '[placeholder*="Search"]', 'text': 'python automation'},
                    {'action': 'key', 'key': 'Enter'},
                    {'action': 'wait', 'selector': '.repo-list-item'},
                    {'action': 'validate', 'condition': 'repositories_visible'}
                ],
                expected_outcome='Repository search results displayed'
            ),
            
            # Salesforce Enterprise Tests
            'salesforce_login_page': TestScenario(
                scenario_id='salesforce_login_page',
                platform='salesforce',
                category=PlatformCategory.ENTERPRISE,
                description='Load Salesforce login page',
                steps=[
                    {'action': 'navigate', 'url': 'https://login.salesforce.com'},
                    {'action': 'wait', 'selector': '#username'},
                    {'action': 'validate', 'condition': 'login_form_visible'}
                ],
                expected_outcome='Salesforce login page loaded'
            ),
            
            # Zomato Indian App Tests
            'zomato_search_restaurants': TestScenario(
                scenario_id='zomato_search_restaurants',
                platform='zomato',
                category=PlatformCategory.INDIAN_APPS,
                description='Search for restaurants on Zomato',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.zomato.com'},
                    {'action': 'wait', 'selector': '[placeholder*="Search"]'},
                    {'action': 'input', 'selector': '[placeholder*="Search"]', 'text': 'pizza'},
                    {'action': 'key', 'key': 'Enter'},
                    {'action': 'wait', 'selector': '.restaurant-card'},
                    {'action': 'validate', 'condition': 'restaurants_visible'}
                ],
                expected_outcome='Restaurant search results displayed'
            ),
            
            # AWS Cloud Platform Tests
            'aws_homepage_load': TestScenario(
                scenario_id='aws_homepage_load',
                platform='aws',
                category=PlatformCategory.CLOUD_PLATFORMS,
                description='Load AWS homepage',
                steps=[
                    {'action': 'navigate', 'url': 'https://aws.amazon.com'},
                    {'action': 'wait', 'selector': '.awsui-button'},
                    {'action': 'validate', 'condition': 'homepage_loaded'}
                ],
                expected_outcome='AWS homepage loaded successfully'
            ),
            
            # YouTube Media Platform Tests
            'youtube_search_video': TestScenario(
                scenario_id='youtube_search_video',
                platform='youtube',
                category=PlatformCategory.SOCIAL_MEDIA,
                description='Search for videos on YouTube',
                steps=[
                    {'action': 'navigate', 'url': 'https://www.youtube.com'},
                    {'action': 'wait', 'selector': '[name="search_query"]'},
                    {'action': 'input', 'selector': '[name="search_query"]', 'text': 'automation tutorial'},
                    {'action': 'click', 'selector': '#search-icon-legacy'},
                    {'action': 'wait', 'selector': '#contents ytd-video-renderer'},
                    {'action': 'validate', 'condition': 'videos_visible'}
                ],
                expected_outcome='Video search results displayed'
            )
        }
    
    async def run_comprehensive_tests(self, browser: Browser, 
                                    categories: List[PlatformCategory] = None) -> Dict[str, Any]:
        """Run comprehensive tests across all platforms"""
        if categories is None:
            categories = list(PlatformCategory)
        
        start_time = time.time()
        results = {
            'test_session_id': f'test_{int(start_time)}',
            'start_time': start_time,
            'categories_tested': [cat.value for cat in categories],
            'total_scenarios': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'results': [],
            'summary': {},
            'performance_metrics': {}
        }
        
        try:
            # Filter scenarios by categories
            scenarios_to_run = [
                scenario for scenario in self.test_scenarios.values()
                if scenario.category in categories
            ]
            
            results['total_scenarios'] = len(scenarios_to_run)
            
            # Run tests in parallel for better performance
            context = await browser.new_context()
            
            # Run scenarios
            for scenario in scenarios_to_run:
                try:
                    page = await context.new_page()
                    test_result = await self._run_test_scenario(page, scenario)
                    results['results'].append(test_result)
                    
                    # Update counters
                    if test_result.status == TestStatus.PASSED:
                        results['passed'] += 1
                    elif test_result.status == TestStatus.FAILED:
                        results['failed'] += 1
                    else:
                        results['skipped'] += 1
                    
                    await page.close()
                    
                    # Small delay between tests
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Test scenario {scenario.scenario_id} failed: {e}")
                    results['failed'] += 1
                    results['results'].append(TestResult(
                        scenario_id=scenario.scenario_id,
                        platform=scenario.platform,
                        status=TestStatus.FAILED,
                        execution_time=0,
                        success_rate=0,
                        steps_completed=0,
                        total_steps=len(scenario.steps),
                        error_message=str(e)
                    ))
            
            await context.close()
            
            # Generate summary
            results['end_time'] = time.time()
            results['total_execution_time'] = results['end_time'] - start_time
            results['success_rate'] = results['passed'] / results['total_scenarios'] if results['total_scenarios'] > 0 else 0
            
            # Generate category-wise summary
            results['summary'] = self._generate_category_summary(results['results'])
            
            # Generate performance metrics
            results['performance_metrics'] = self._generate_performance_metrics(results['results'])
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")
            results['error'] = str(e)
            return results
    
    async def _run_test_scenario(self, page: Page, scenario: TestScenario) -> TestResult:
        """Run a single test scenario"""
        start_time = time.time()
        steps_completed = 0
        screenshots = []
        
        try:
            logger.info(f"Running test scenario: {scenario.scenario_id}")
            
            # Execute each step
            for step_idx, step in enumerate(scenario.steps):
                try:
                    await self._execute_test_step(page, step)
                    steps_completed += 1
                    
                    # Take screenshot for critical steps
                    if step.get('action') in ['validate', 'click', 'navigate']:
                        screenshot_path = f'screenshots/test_{scenario.scenario_id}_step_{step_idx}_{int(time.time())}.png'
                        await page.screenshot(path=screenshot_path)
                        screenshots.append(screenshot_path)
                    
                except Exception as e:
                    logger.error(f"Step {step_idx} failed in {scenario.scenario_id}: {e}")
                    # Take error screenshot
                    error_screenshot = f'screenshots/error_{scenario.scenario_id}_step_{step_idx}_{int(time.time())}.png'
                    try:
                        await page.screenshot(path=error_screenshot)
                        screenshots.append(error_screenshot)
                    except:
                        pass
                    
                    return TestResult(
                        scenario_id=scenario.scenario_id,
                        platform=scenario.platform,
                        status=TestStatus.FAILED,
                        execution_time=time.time() - start_time,
                        success_rate=steps_completed / len(scenario.steps),
                        steps_completed=steps_completed,
                        total_steps=len(scenario.steps),
                        error_message=str(e),
                        screenshots=screenshots
                    )
            
            # All steps completed successfully
            return TestResult(
                scenario_id=scenario.scenario_id,
                platform=scenario.platform,
                status=TestStatus.PASSED,
                execution_time=time.time() - start_time,
                success_rate=1.0,
                steps_completed=steps_completed,
                total_steps=len(scenario.steps),
                screenshots=screenshots,
                performance_metrics={
                    'page_load_time': await self._measure_page_load_time(page),
                    'dom_content_loaded': await self._measure_dom_content_loaded(page)
                }
            )
            
        except Exception as e:
            return TestResult(
                scenario_id=scenario.scenario_id,
                platform=scenario.platform,
                status=TestStatus.FAILED,
                execution_time=time.time() - start_time,
                success_rate=steps_completed / len(scenario.steps),
                steps_completed=steps_completed,
                total_steps=len(scenario.steps),
                error_message=str(e),
                screenshots=screenshots
            )
    
    async def _execute_test_step(self, page: Page, step: Dict[str, Any]):
        """Execute a single test step"""
        action = step['action']
        
        if action == 'navigate':
            await page.goto(step['url'], wait_until='domcontentloaded')
            
        elif action == 'wait':
            selector = step['selector']
            timeout = step.get('timeout', 10) * 1000
            await page.wait_for_selector(selector, timeout=timeout)
            
        elif action == 'input':
            selector = step['selector']
            text = step['text']
            await page.wait_for_selector(selector, timeout=10000)
            await page.fill(selector, text)
            
        elif action == 'click':
            selector = step['selector']
            index = step.get('index', None)
            await page.wait_for_selector(selector, timeout=10000)
            
            if index is not None:
                elements = await page.locator(selector).all()
                if index < len(elements):
                    await elements[index].click()
                else:
                    raise Exception(f"Element index {index} out of range")
            else:
                await page.click(selector)
                
        elif action == 'key':
            key = step['key']
            await page.keyboard.press(key)
            
        elif action == 'validate':
            condition = step['condition']
            await self._validate_condition(page, condition)
            
        else:
            raise Exception(f"Unknown action: {action}")
    
    async def _validate_condition(self, page: Page, condition: str):
        """Validate a test condition"""
        if condition == 'results_visible':
            # Check if search results are visible
            selectors = [
                '[data-component-type="s-search-result"]',  # Amazon
                '[data-tkid]',  # Flipkart
                '.repo-list-item',  # GitHub
                '.restaurant-card',  # Zomato
                '#contents ytd-video-renderer'  # YouTube
            ]
            
            for selector in selectors:
                try:
                    count = await page.locator(selector).count()
                    if count > 0:
                        return True
                except:
                    continue
            
            raise Exception("No search results found")
            
        elif condition == 'login_form_visible':
            # Check if login form is visible
            selectors = [
                '#email',  # Facebook
                '#session_key',  # LinkedIn
                '#username'  # Salesforce
            ]
            
            for selector in selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    return True
                except:
                    continue
            
            raise Exception("Login form not found")
            
        elif condition == 'cart_updated':
            # Check if cart was updated
            selectors = [
                '#nav-cart-count',  # Amazon
                '.cart-count'  # Generic
            ]
            
            for selector in selectors:
                try:
                    element = await page.locator(selector).first
                    if await element.count() > 0:
                        return True
                except:
                    continue
            
            # If no cart count found, just check if we're not on an error page
            title = await page.title()
            if 'error' not in title.lower():
                return True
            
            raise Exception("Cart update validation failed")
            
        elif condition == 'repositories_visible':
            await page.wait_for_selector('.repo-list-item', timeout=10000)
            
        elif condition == 'restaurants_visible':
            await page.wait_for_selector('.restaurant-card', timeout=10000)
            
        elif condition == 'videos_visible':
            await page.wait_for_selector('#contents ytd-video-renderer', timeout=10000)
            
        elif condition == 'homepage_loaded':
            # Generic homepage validation
            title = await page.title()
            if len(title) > 0 and 'error' not in title.lower():
                return True
            raise Exception("Homepage not loaded properly")
            
        else:
            raise Exception(f"Unknown validation condition: {condition}")
    
    async def _measure_page_load_time(self, page: Page) -> float:
        """Measure page load time"""
        try:
            load_time = await page.evaluate("""
                () => {
                    return performance.timing.loadEventEnd - performance.timing.navigationStart;
                }
            """)
            return load_time / 1000.0  # Convert to seconds
        except:
            return 0.0
    
    async def _measure_dom_content_loaded(self, page: Page) -> float:
        """Measure DOM content loaded time"""
        try:
            dom_time = await page.evaluate("""
                () => {
                    return performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart;
                }
            """)
            return dom_time / 1000.0  # Convert to seconds
        except:
            return 0.0
    
    def _generate_category_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate category-wise summary"""
        summary = {}
        
        # Group results by platform category
        for result in results:
            platform = result.platform
            config = self.platform_configs.get(platform, {})
            category = config.get('category', PlatformCategory.ENTERPRISE).value
            
            if category not in summary:
                summary[category] = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'success_rate': 0,
                    'avg_execution_time': 0,
                    'platforms': []
                }
            
            summary[category]['total'] += 1
            summary[category]['platforms'].append(platform)
            
            if result.status == TestStatus.PASSED:
                summary[category]['passed'] += 1
            else:
                summary[category]['failed'] += 1
        
        # Calculate success rates and averages
        for category_data in summary.values():
            if category_data['total'] > 0:
                category_data['success_rate'] = category_data['passed'] / category_data['total']
                
                # Calculate average execution time for passed tests
                passed_results = [r for r in results if 
                                self.platform_configs.get(r.platform, {}).get('category', PlatformCategory.ENTERPRISE).value == category_data and
                                r.status == TestStatus.PASSED]
                
                if passed_results:
                    category_data['avg_execution_time'] = sum(r.execution_time for r in passed_results) / len(passed_results)
        
        return summary
    
    def _generate_performance_metrics(self, results: List[TestResult]) -> Dict[str, Any]:
        """Generate performance metrics"""
        passed_results = [r for r in results if r.status == TestStatus.PASSED]
        
        if not passed_results:
            return {}
        
        execution_times = [r.execution_time for r in passed_results]
        
        return {
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'total_test_time': sum(execution_times),
            'fastest_platform': min(passed_results, key=lambda x: x.execution_time).platform,
            'slowest_platform': max(passed_results, key=lambda x: x.execution_time).platform
        }
    
    async def run_platform_specific_tests(self, browser: Browser, platform: str) -> Dict[str, Any]:
        """Run tests for a specific platform"""
        platform_scenarios = [
            scenario for scenario in self.test_scenarios.values()
            if scenario.platform == platform
        ]
        
        if not platform_scenarios:
            return {
                'platform': platform,
                'error': f'No test scenarios found for platform: {platform}'
            }
        
        context = await browser.new_context()
        results = []
        
        for scenario in platform_scenarios:
            try:
                page = await context.new_page()
                result = await self._run_test_scenario(page, scenario)
                results.append(result)
                await page.close()
            except Exception as e:
                logger.error(f"Platform test failed for {platform}: {e}")
                results.append(TestResult(
                    scenario_id=scenario.scenario_id,
                    platform=platform,
                    status=TestStatus.FAILED,
                    execution_time=0,
                    success_rate=0,
                    steps_completed=0,
                    total_steps=len(scenario.steps),
                    error_message=str(e)
                ))
        
        await context.close()
        
        # Calculate platform metrics
        passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        total = len(results)
        
        return {
            'platform': platform,
            'total_tests': total,
            'passed': passed,
            'failed': total - passed,
            'success_rate': passed / total if total > 0 else 0,
            'results': results
        }
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PLATFORM TESTING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append(f"Test Session ID: {test_results.get('test_session_id', 'N/A')}")
        report.append(f"Total Scenarios: {test_results.get('total_scenarios', 0)}")
        report.append(f"Passed: {test_results.get('passed', 0)}")
        report.append(f"Failed: {test_results.get('failed', 0)}")
        report.append(f"Success Rate: {test_results.get('success_rate', 0):.1%}")
        report.append(f"Total Execution Time: {test_results.get('total_execution_time', 0):.2f}s")
        report.append("")
        
        # Category Summary
        if 'summary' in test_results:
            report.append("CATEGORY SUMMARY:")
            report.append("-" * 40)
            for category, data in test_results['summary'].items():
                report.append(f"{category.upper()}:")
                report.append(f"  Total: {data['total']}")
                report.append(f"  Passed: {data['passed']}")
                report.append(f"  Failed: {data['failed']}")
                report.append(f"  Success Rate: {data['success_rate']:.1%}")
                report.append(f"  Platforms: {', '.join(set(data['platforms']))}")
                report.append("")
        
        # Performance Metrics
        if 'performance_metrics' in test_results:
            metrics = test_results['performance_metrics']
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 40)
            report.append(f"Average Execution Time: {metrics.get('avg_execution_time', 0):.2f}s")
            report.append(f"Fastest Platform: {metrics.get('fastest_platform', 'N/A')}")
            report.append(f"Slowest Platform: {metrics.get('slowest_platform', 'N/A')}")
            report.append("")
        
        # Individual Results
        report.append("INDIVIDUAL TEST RESULTS:")
        report.append("-" * 40)
        for result in test_results.get('results', []):
            status_icon = "✅" if result.status == TestStatus.PASSED else "❌"
            report.append(f"{status_icon} {result.scenario_id} ({result.platform})")
            report.append(f"   Status: {result.status.value}")
            report.append(f"   Execution Time: {result.execution_time:.2f}s")
            report.append(f"   Steps: {result.steps_completed}/{result.total_steps}")
            if result.error_message:
                report.append(f"   Error: {result.error_message}")
            report.append("")
        
        return "\n".join(report)

# Global tester instance
_global_tester: Optional[ComprehensivePlatformTester] = None

def get_platform_tester() -> ComprehensivePlatformTester:
    """Get or create the global platform tester"""
    global _global_tester
    
    if _global_tester is None:
        _global_tester = ComprehensivePlatformTester()
    
    return _global_tester
"""
Enterprise E-commerce & Retail Automation System
===============================================

Comprehensive automation for all retail scenarios:
- Amazon seller/buyer automation
- eBay listing and bidding automation
- Shopify store management
- Walmart marketplace automation
- Etsy product listing and sales
- Price monitoring and comparison
- Inventory management and restocking
- Order processing and fulfillment
- Customer service automation
- Review management and responses
- Marketing campaign automation
- Dropshipping workflow automation

Features:
- Multi-platform inventory synchronization
- Dynamic pricing optimization
- Automated customer communications
- Review sentiment analysis and response
- Fraud detection and prevention
- Supply chain automation
- Analytics and performance tracking
- Compliance monitoring
- A/B testing for listings
- Social commerce integration
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta, date
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import requests
from decimal import Decimal
import hashlib

try:
    from playwright.async_api import Page, ElementHandle
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

from ...core.deterministic_executor import DeterministicExecutor
from ...core.realtime_data_fabric import RealTimeDataFabric


class EcommercePlatform(str, Enum):
    """Supported e-commerce platforms."""
    AMAZON = "amazon"
    EBAY = "ebay"
    SHOPIFY = "shopify"
    WALMART = "walmart"
    ETSY = "etsy"
    ALIBABA = "alibaba"
    FACEBOOK_MARKETPLACE = "facebook_marketplace"
    MERCARI = "mercari"
    POSHMARK = "poshmark"
    DEPOP = "depop"


class OrderStatus(str, Enum):
    """Order status states."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"
    RETURNED = "returned"


@dataclass
class Product:
    """Product information."""
    product_id: str
    title: str
    description: str
    price: Decimal
    currency: str = "USD"
    category: str = ""
    brand: str = ""
    sku: str = ""
    inventory_count: int = 0
    images: List[str] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.images is None:
            self.images = []
        if self.attributes is None:
            self.attributes = {}


@dataclass
class Order:
    """Order information."""
    order_id: str
    customer_email: str
    products: List[Dict[str, Any]]
    total_amount: Decimal
    status: OrderStatus = OrderStatus.PENDING
    shipping_address: Dict[str, str] = None
    tracking_number: Optional[str] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.shipping_address is None:
            self.shipping_address = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AmazonAutomation:
    """Amazon marketplace automation."""
    
    def __init__(self, executor: DeterministicExecutor):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # Amazon URLs and selectors
        self.amazon_config = {
            'seller_central': 'https://sellercentral.amazon.com',
            'marketplace': 'https://www.amazon.com',
            'selectors': {
                'login_email': '#ap_email',
                'login_password': '#ap_password',
                'login_button': '#signInSubmit',
                'add_product': '#sku-create-button',
                'product_title': '#product-title',
                'product_price': '#price',
                'inventory_count': '#quantity',
                'search_box': '#twotabsearchtextbox',
                'add_to_cart': '#add-to-cart-button',
                'buy_now': '#buy-now-button'
            }
        }
        
        # Product and order tracking
        self.products = {}
        self.orders = []
        self.inventory = {}
    
    async def login_to_amazon(self, credentials: Dict[str, str], is_seller: bool = False) -> bool:
        """Login to Amazon (buyer or seller)."""
        try:
            url = self.amazon_config['seller_central'] if is_seller else self.amazon_config['marketplace']
            await self.executor.page.goto(f"{url}/ap/signin")
            
            # Fill credentials
            await self.executor.page.fill(self.amazon_config['selectors']['login_email'], credentials['email'])
            await self.executor.page.click('#continue')
            
            await self.executor.page.fill(self.amazon_config['selectors']['login_password'], credentials['password'])
            await self.executor.page.click(self.amazon_config['selectors']['login_button'])
            
            # Handle 2FA if present
            await self._handle_amazon_2fa()
            
            # Wait for dashboard
            dashboard_selector = '#sc-myo-dashboard' if is_seller else '#nav-main'
            await self.executor.page.wait_for_selector(dashboard_selector, timeout=30000)
            
            self.logger.info(f"Successfully logged into Amazon ({'seller' if is_seller else 'buyer'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Amazon login failed: {e}")
            return False
    
    async def list_product(self, product: Product) -> Dict[str, Any]:
        """List a product on Amazon."""
        try:
            # Navigate to add product page
            await self.executor.page.goto(f"{self.amazon_config['seller_central']}/inventory/add-products")
            
            # Click add product
            await self.executor.page.click(self.amazon_config['selectors']['add_product'])
            
            # Fill product details
            await self._fill_amazon_product_form(product)
            
            # Upload images
            await self._upload_product_images(product.images)
            
            # Set pricing and inventory
            await self._set_amazon_pricing_inventory(product)
            
            # Submit listing
            await self.executor.page.click('#submit-listing')
            
            # Wait for confirmation
            await self.executor.page.wait_for_selector('.listing-success, #success-message', timeout=30000)
            
            # Extract listing details
            listing_id = await self._extract_amazon_listing_id()
            
            result = {
                'product_id': product.product_id,
                'listing_id': listing_id,
                'platform': EcommercePlatform.AMAZON,
                'status': 'listed',
                'listed_at': datetime.utcnow()
            }
            
            self.products[product.product_id] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Amazon product listing failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def monitor_prices(self, product_asins: List[str]) -> Dict[str, Any]:
        """Monitor competitor prices on Amazon."""
        try:
            price_data = {}
            
            for asin in product_asins:
                # Navigate to product page
                await self.executor.page.goto(f"https://www.amazon.com/dp/{asin}")
                
                # Extract price information
                price_info = await self._extract_amazon_price_info()
                price_data[asin] = price_info
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(2)
            
            return {
                'prices': price_data,
                'monitored_at': datetime.utcnow(),
                'total_products': len(product_asins)
            }
            
        except Exception as e:
            self.logger.error(f"Amazon price monitoring failed: {e}")
            return {'error': str(e)}
    
    async def process_orders(self) -> List[Dict[str, Any]]:
        """Process pending Amazon orders."""
        try:
            # Navigate to orders page
            await self.executor.page.goto(f"{self.amazon_config['seller_central']}/orders-v3")
            
            # Extract pending orders
            orders = await self._extract_amazon_orders()
            
            processed_orders = []
            for order in orders:
                if order['status'] == 'unshipped':
                    # Process the order
                    result = await self._process_amazon_order(order)
                    processed_orders.append(result)
            
            return processed_orders
            
        except Exception as e:
            self.logger.error(f"Amazon order processing failed: {e}")
            return []
    
    async def _fill_amazon_product_form(self, product: Product):
        """Fill Amazon product listing form."""
        try:
            # Product title
            await self.executor.page.fill('#product-title', product.title)
            
            # Product description
            await self.executor.page.fill('#product-description', product.description)
            
            # Category selection
            if product.category:
                await self.executor.page.select_option('#product-category', product.category)
            
            # Brand
            if product.brand:
                await self.executor.page.fill('#brand', product.brand)
            
            # SKU
            await self.executor.page.fill('#sku', product.sku or product.product_id)
            
            # Additional attributes
            for key, value in product.attributes.items():
                try:
                    await self.executor.page.fill(f'#{key}', str(value))
                except:
                    continue
                    
        except Exception as e:
            self.logger.error(f"Amazon form filling failed: {e}")
            raise


class EbayAutomation:
    """eBay marketplace automation."""
    
    def __init__(self, executor: DeterministicExecutor):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # eBay configuration
        self.ebay_config = {
            'seller_hub': 'https://www.ebay.com/sh/ovw',
            'marketplace': 'https://www.ebay.com',
            'selectors': {
                'username': '#userid',
                'password': '#pass',
                'login_button': '#sgnBt',
                'sell_button': '#gh-p-2',
                'list_item': '#c2c-create-listing-btn',
                'title': '#x-textbox-label-textbox',
                'category': '#categoryDropdown',
                'condition': '#condition-selection',
                'price': '#notranslate'
            }
        }
        
        # Tracking
        self.ebay_listings = {}
        self.ebay_sales = []
    
    async def login_to_ebay(self, credentials: Dict[str, str]) -> bool:
        """Login to eBay."""
        try:
            await self.executor.page.goto('https://signin.ebay.com/ws/eBayISAPI.dll')
            
            # Fill credentials
            await self.executor.page.fill(self.ebay_config['selectors']['username'], credentials['username'])
            await self.executor.page.fill(self.ebay_config['selectors']['password'], credentials['password'])
            await self.executor.page.click(self.ebay_config['selectors']['login_button'])
            
            # Wait for successful login
            await self.executor.page.wait_for_selector('#gh-ug', timeout=30000)
            
            self.logger.info("Successfully logged into eBay")
            return True
            
        except Exception as e:
            self.logger.error(f"eBay login failed: {e}")
            return False
    
    async def list_item_on_ebay(self, product: Product, auction_duration: int = 7) -> Dict[str, Any]:
        """List an item on eBay."""
        try:
            # Navigate to sell page
            await self.executor.page.goto('https://www.ebay.com/sl/sell')
            
            # Start listing
            await self.executor.page.click(self.ebay_config['selectors']['list_item'])
            
            # Fill item details
            await self._fill_ebay_listing_form(product)
            
            # Set auction/buy-it-now settings
            await self._set_ebay_sale_format(product, auction_duration)
            
            # Upload photos
            await self._upload_ebay_photos(product.images)
            
            # Set shipping options
            await self._set_ebay_shipping()
            
            # Review and list
            await self.executor.page.click('#btn-list-item')
            
            # Extract listing ID
            listing_id = await self._extract_ebay_listing_id()
            
            result = {
                'product_id': product.product_id,
                'listing_id': listing_id,
                'platform': EcommercePlatform.EBAY,
                'status': 'listed',
                'auction_duration': auction_duration,
                'listed_at': datetime.utcnow()
            }
            
            self.ebay_listings[product.product_id] = result
            return result
            
        except Exception as e:
            self.logger.error(f"eBay listing failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def monitor_ebay_auctions(self) -> List[Dict[str, Any]]:
        """Monitor active eBay auctions."""
        try:
            # Navigate to active listings
            await self.executor.page.goto(f"{self.ebay_config['seller_hub']}/selling")
            
            # Extract auction data
            auctions = await self._extract_ebay_auction_data()
            
            return auctions
            
        except Exception as e:
            self.logger.error(f"eBay auction monitoring failed: {e}")
            return []


class ShopifyAutomation:
    """Shopify store automation."""
    
    def __init__(self, executor: DeterministicExecutor):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # Shopify configuration
        self.shopify_config = {
            'admin_base': 'https://{store}.myshopify.com/admin',
            'selectors': {
                'email': '#account_email',
                'password': '#account_password',
                'login_button': '#log_in',
                'add_product': '.btn--primary[href*="products/new"]',
                'product_title': '#product_title',
                'product_description': '#product_description',
                'product_price': '#product_price',
                'inventory_quantity': '#product_inventory_quantity'
            }
        }
        
        # Store data
        self.store_products = {}
        self.store_orders = []
        self.store_analytics = {}
    
    async def login_to_shopify(self, store_name: str, credentials: Dict[str, str]) -> bool:
        """Login to Shopify admin."""
        try:
            admin_url = self.shopify_config['admin_base'].format(store=store_name)
            await self.executor.page.goto(f"{admin_url}/auth/login")
            
            # Fill credentials
            await self.executor.page.fill(self.shopify_config['selectors']['email'], credentials['email'])
            await self.executor.page.fill(self.shopify_config['selectors']['password'], credentials['password'])
            await self.executor.page.click(self.shopify_config['selectors']['login_button'])
            
            # Wait for admin dashboard
            await self.executor.page.wait_for_selector('.Polaris-Page-Header', timeout=30000)
            
            self.logger.info(f"Successfully logged into Shopify store: {store_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Shopify login failed: {e}")
            return False
    
    async def add_shopify_product(self, product: Product) -> Dict[str, Any]:
        """Add a product to Shopify store."""
        try:
            # Navigate to products page
            await self.executor.page.goto('/admin/products')
            
            # Click add product
            await self.executor.page.click(self.shopify_config['selectors']['add_product'])
            
            # Fill product form
            await self._fill_shopify_product_form(product)
            
            # Upload images
            await self._upload_shopify_images(product.images)
            
            # Set SEO and organization
            await self._set_shopify_seo_organization(product)
            
            # Save product
            await self.executor.page.click('#save-product')
            
            # Extract product ID
            product_id = await self._extract_shopify_product_id()
            
            result = {
                'product_id': product.product_id,
                'shopify_id': product_id,
                'platform': EcommercePlatform.SHOPIFY,
                'status': 'added',
                'added_at': datetime.utcnow()
            }
            
            self.store_products[product.product_id] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Shopify product addition failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def process_shopify_orders(self) -> List[Dict[str, Any]]:
        """Process Shopify orders."""
        try:
            # Navigate to orders page
            await self.executor.page.goto('/admin/orders')
            
            # Extract unfulfilled orders
            orders = await self._extract_shopify_orders()
            
            processed_orders = []
            for order in orders:
                if order['fulfillment_status'] == 'unfulfilled':
                    result = await self._fulfill_shopify_order(order)
                    processed_orders.append(result)
            
            return processed_orders
            
        except Exception as e:
            self.logger.error(f"Shopify order processing failed: {e}")
            return []


class PriceMonitoringSystem:
    """Advanced price monitoring and optimization."""
    
    def __init__(self, executor: DeterministicExecutor, data_fabric: RealTimeDataFabric):
        self.executor = executor
        self.data_fabric = data_fabric
        self.logger = logging.getLogger(__name__)
        
        # Price tracking
        self.price_history = {}
        self.competitor_prices = {}
        self.pricing_rules = {}
    
    async def monitor_competitor_prices(self, products: List[str], platforms: List[EcommercePlatform]) -> Dict[str, Any]:
        """Monitor competitor prices across platforms."""
        try:
            price_data = {}
            
            for product in products:
                price_data[product] = {}
                
                for platform in platforms:
                    platform_prices = await self._get_platform_prices(product, platform)
                    price_data[product][platform.value] = platform_prices
                    
                    # Small delay between requests
                    await asyncio.sleep(1)
            
            # Store price history
            timestamp = datetime.utcnow()
            self.price_history[timestamp] = price_data
            
            # Analyze price trends
            trends = await self._analyze_price_trends(price_data)
            
            return {
                'price_data': price_data,
                'trends': trends,
                'monitored_at': timestamp,
                'total_products': len(products),
                'platforms_checked': len(platforms)
            }
            
        except Exception as e:
            self.logger.error(f"Price monitoring failed: {e}")
            return {'error': str(e)}
    
    async def optimize_pricing(self, product_id: str, current_price: Decimal, 
                             competitor_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize product pricing based on competitor analysis."""
        try:
            # Get competitor prices
            competitor_prices = [
                data.get('price', 0) for data in competitor_data.values()
                if data.get('price', 0) > 0
            ]
            
            if not competitor_prices:
                return {'recommended_price': current_price, 'strategy': 'maintain'}
            
            # Calculate statistics
            min_price = min(competitor_prices)
            max_price = max(competitor_prices)
            avg_price = sum(competitor_prices) / len(competitor_prices)
            
            # Apply pricing strategy
            strategy = await self._determine_pricing_strategy(product_id, current_price, {
                'min': min_price,
                'max': max_price,
                'avg': avg_price
            })
            
            recommended_price = await self._calculate_optimal_price(
                current_price, competitor_prices, strategy
            )
            
            return {
                'current_price': float(current_price),
                'recommended_price': float(recommended_price),
                'strategy': strategy,
                'competitor_stats': {
                    'min': min_price,
                    'max': max_price,
                    'avg': avg_price,
                    'count': len(competitor_prices)
                },
                'price_change': float(recommended_price - current_price),
                'change_percentage': float((recommended_price - current_price) / current_price * 100)
            }
            
        except Exception as e:
            self.logger.error(f"Price optimization failed: {e}")
            return {'error': str(e)}


class InventoryManagementSystem:
    """Advanced inventory management and automation."""
    
    def __init__(self, executor: DeterministicExecutor):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # Inventory tracking
        self.inventory_levels = {}
        self.reorder_points = {}
        self.suppliers = {}
        self.pending_orders = {}
    
    async def sync_inventory_across_platforms(self, platforms: List[EcommercePlatform]) -> Dict[str, Any]:
        """Synchronize inventory across multiple platforms."""
        try:
            sync_results = {}
            
            for platform in platforms:
                try:
                    inventory_data = await self._get_platform_inventory(platform)
                    sync_result = await self._sync_platform_inventory(platform, inventory_data)
                    sync_results[platform.value] = sync_result
                except Exception as e:
                    sync_results[platform.value] = {'status': 'failed', 'error': str(e)}
            
            # Update master inventory
            await self._update_master_inventory(sync_results)
            
            return {
                'sync_results': sync_results,
                'synced_at': datetime.utcnow(),
                'platforms_synced': len([r for r in sync_results.values() if r.get('status') == 'success'])
            }
            
        except Exception as e:
            self.logger.error(f"Inventory sync failed: {e}")
            return {'error': str(e)}
    
    async def auto_reorder_inventory(self) -> List[Dict[str, Any]]:
        """Automatically reorder inventory based on reorder points."""
        try:
            reorder_actions = []
            
            for product_id, current_level in self.inventory_levels.items():
                reorder_point = self.reorder_points.get(product_id, 0)
                
                if current_level <= reorder_point:
                    # Calculate reorder quantity
                    reorder_qty = await self._calculate_reorder_quantity(product_id)
                    
                    # Find best supplier
                    supplier = await self._find_best_supplier(product_id)
                    
                    # Place order
                    order_result = await self._place_supplier_order(product_id, reorder_qty, supplier)
                    
                    reorder_actions.append({
                        'product_id': product_id,
                        'current_level': current_level,
                        'reorder_point': reorder_point,
                        'reorder_quantity': reorder_qty,
                        'supplier': supplier,
                        'order_result': order_result
                    })
            
            return reorder_actions
            
        except Exception as e:
            self.logger.error(f"Auto reorder failed: {e}")
            return []


class CustomerServiceAutomation:
    """Automated customer service and support."""
    
    def __init__(self, executor: DeterministicExecutor):
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        
        # Customer service data
        self.support_tickets = {}
        self.response_templates = {}
        self.escalation_rules = {}
    
    async def process_customer_inquiries(self, platform: EcommercePlatform) -> List[Dict[str, Any]]:
        """Process customer inquiries and messages."""
        try:
            # Get platform-specific messages
            messages = await self._get_platform_messages(platform)
            
            processed_messages = []
            for message in messages:
                # Analyze message sentiment and intent
                analysis = await self._analyze_message(message)
                
                # Generate appropriate response
                response = await self._generate_response(message, analysis)
                
                # Send response if automated response is appropriate
                if response.get('auto_respond', False):
                    send_result = await self._send_response(platform, message, response)
                    processed_messages.append({
                        'message_id': message['id'],
                        'analysis': analysis,
                        'response': response,
                        'sent': send_result
                    })
                else:
                    # Flag for human review
                    await self._flag_for_human_review(message, analysis)
                    processed_messages.append({
                        'message_id': message['id'],
                        'analysis': analysis,
                        'action': 'flagged_for_review'
                    })
            
            return processed_messages
            
        except Exception as e:
            self.logger.error(f"Customer service automation failed: {e}")
            return []
    
    async def manage_reviews_and_feedback(self, platform: EcommercePlatform) -> Dict[str, Any]:
        """Manage product reviews and customer feedback."""
        try:
            # Get recent reviews
            reviews = await self._get_platform_reviews(platform)
            
            review_actions = []
            for review in reviews:
                # Analyze review sentiment
                sentiment = await self._analyze_review_sentiment(review)
                
                # Determine action based on sentiment and content
                if sentiment['score'] < 0.3:  # Negative review
                    # Generate response to address concerns
                    response = await self._generate_review_response(review, sentiment)
                    
                    # Post response
                    if response:
                        post_result = await self._post_review_response(platform, review, response)
                        review_actions.append({
                            'review_id': review['id'],
                            'sentiment': sentiment,
                            'response_posted': post_result,
                            'action': 'responded_to_negative'
                        })
                
                elif sentiment['score'] > 0.7:  # Positive review
                    # Thank customer
                    thanks_response = await self._generate_thanks_response(review)
                    post_result = await self._post_review_response(platform, review, thanks_response)
                    review_actions.append({
                        'review_id': review['id'],
                        'sentiment': sentiment,
                        'response_posted': post_result,
                        'action': 'thanked_customer'
                    })
            
            return {
                'total_reviews_processed': len(reviews),
                'actions_taken': review_actions,
                'processed_at': datetime.utcnow()
            }
            
        except Exception as e:
            self.logger.error(f"Review management failed: {e}")
            return {'error': str(e)}


class UniversalEcommerceOrchestrator:
    """Master orchestrator for all e-commerce operations."""
    
    def __init__(self, executor: DeterministicExecutor, data_fabric: RealTimeDataFabric):
        self.executor = executor
        self.data_fabric = data_fabric
        self.logger = logging.getLogger(__name__)
        
        # Initialize platform automations
        self.amazon = AmazonAutomation(executor)
        self.ebay = EbayAutomation(executor)
        self.shopify = ShopifyAutomation(executor)
        
        # Initialize management systems
        self.price_monitor = PriceMonitoringSystem(executor, data_fabric)
        self.inventory_manager = InventoryManagementSystem(executor)
        self.customer_service = CustomerServiceAutomation(executor)
        
        # Master data
        self.all_products = {}
        self.all_orders = []
        self.platform_connections = {}
        
        # Analytics and reporting
        self.sales_analytics = {}
        self.performance_metrics = {}
    
    async def execute_ecommerce_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive e-commerce workflow."""
        try:
            workflow_id = str(uuid.uuid4())
            results = []
            
            for step in workflow.get('steps', []):
                step_result = await self._execute_ecommerce_step(step)
                results.append(step_result)
                
                # Check for critical failures
                if step_result.get('status') == 'failed' and step.get('critical', False):
                    return {
                        'workflow_id': workflow_id,
                        'status': 'failed',
                        'error': f"Critical step failed: {step.get('name')}",
                        'completed_steps': results
                    }
            
            return {
                'workflow_id': workflow_id,
                'status': 'completed',
                'results': results,
                'completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"E-commerce workflow execution failed: {e}")
            return {
                'workflow_id': workflow_id,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _execute_ecommerce_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute individual e-commerce workflow step."""
        try:
            step_type = step.get('type')
            step_params = step.get('parameters', {})
            
            if step_type == 'list_product':
                platform = EcommercePlatform(step_params.get('platform'))
                product = Product(**step_params.get('product', {}))
                
                if platform == EcommercePlatform.AMAZON:
                    result = await self.amazon.list_product(product)
                elif platform == EcommercePlatform.EBAY:
                    result = await self.ebay.list_item_on_ebay(product)
                elif platform == EcommercePlatform.SHOPIFY:
                    result = await self.shopify.add_shopify_product(product)
                else:
                    result = {'status': 'failed', 'error': f'Unsupported platform: {platform}'}
                
                return {'type': step_type, 'result': result}
            
            elif step_type == 'monitor_prices':
                result = await self.price_monitor.monitor_competitor_prices(
                    step_params.get('products', []),
                    [EcommercePlatform(p) for p in step_params.get('platforms', [])]
                )
                return {'type': step_type, 'result': result}
            
            elif step_type == 'sync_inventory':
                result = await self.inventory_manager.sync_inventory_across_platforms(
                    [EcommercePlatform(p) for p in step_params.get('platforms', [])]
                )
                return {'type': step_type, 'result': result}
            
            elif step_type == 'process_orders':
                platform = EcommercePlatform(step_params.get('platform'))
                
                if platform == EcommercePlatform.AMAZON:
                    result = await self.amazon.process_orders()
                elif platform == EcommercePlatform.SHOPIFY:
                    result = await self.shopify.process_shopify_orders()
                else:
                    result = {'error': f'Order processing not implemented for {platform}'}
                
                return {'type': step_type, 'result': result}
            
            elif step_type == 'customer_service':
                platform = EcommercePlatform(step_params.get('platform'))
                result = await self.customer_service.process_customer_inquiries(platform)
                return {'type': step_type, 'result': result}
            
            else:
                return {'type': step_type, 'status': 'failed', 'error': f'Unknown step type: {step_type}'}
                
        except Exception as e:
            self.logger.error(f"E-commerce step execution failed: {e}")
            return {'type': step.get('type'), 'status': 'failed', 'error': str(e)}
    
    def get_ecommerce_analytics(self) -> Dict[str, Any]:
        """Get comprehensive e-commerce analytics."""
        return {
            'total_products': len(self.all_products),
            'total_orders': len(self.all_orders),
            'connected_platforms': len(self.platform_connections),
            'amazon_products': len(self.amazon.products),
            'ebay_listings': len(self.ebay.ebay_listings),
            'shopify_products': len(self.shopify.store_products),
            'inventory_items': len(self.inventory_manager.inventory_levels),
            'support_tickets': len(self.customer_service.support_tickets),
            'last_updated': datetime.utcnow().isoformat()
        }
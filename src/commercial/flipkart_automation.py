#!/usr/bin/env python3
"""
FLIPKART AUTOMATION - Production Standard
=========================================

Production-grade automation for Flipkart e-commerce platform.
Handles search, product selection, price comparison, and checkout flows.
"""

import asyncio
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class FlipkartProduct:
    name: str
    price: float
    rating: float
    url: str
    availability: str
    seller: str

class FlipkartAutomation:
    """Production-standard Flipkart automation handler"""
    
    def __init__(self):
        self.base_url = "https://www.flipkart.com"
        self.selectors = self._load_selectors()
        self.price_patterns = self._load_price_patterns()
    
    def _load_selectors(self) -> Dict[str, str]:
        """Load Flipkart-specific selectors"""
        return {
            'search_box': 'input[name="q"], input[placeholder*="Search"], input[title*="Search"]',
            'search_button': 'button[type="submit"], button._2iLD__',
            'product_cards': 'div[data-id], div._1AtVbE, div._13oc-S, div._2kHMtA',
            'product_title': 'a._1fQZEK, div._4rR01T, a.s1Q9rs',
            'product_price': '._30jeq3._1_WHN1, div._25b18c, ._1_WHN1',
            'product_rating': 'div._3LWZlK, span._2_R_DZ',
            'buy_now_button': 'button:has-text("Buy Now"), button._2KpZ6l._2U9uOA._3v1-ww',
            'add_to_cart': 'button:has-text("Add to Cart"), button._2KpZ6l._2U9uOA._3v1-ww',
            'price_filter_min': 'input[placeholder*="Min"]',
            'price_filter_max': 'input[placeholder*="Max"]',
            'sort_dropdown': 'div._396cs4._2xHGWy',
            'sort_price_low_high': 'div:has-text("Price -- Low to High")',
            'pagination_next': 'a._1LKTO3:has-text("Next")'
        }
    
    def _load_price_patterns(self) -> Dict[str, str]:
        """Load price extraction patterns"""
        return {
            'currency_symbol': '₹',
            'price_regex': r'₹[\d,]+',
            'discount_regex': r'\d+%\s*off'
        }
    
    async def search_product(self, playwright_page, product_name: str) -> Dict[str, Any]:
        """Search for a product on Flipkart"""
        
        try:
            print(f"🔍 Searching Flipkart for: {product_name}")
            
            # Navigate to Flipkart
            await playwright_page.goto(self.base_url, wait_until="networkidle")
            await playwright_page.wait_for_timeout(2000)
            
            # Handle login popup if present
            try:
                close_popup = await playwright_page.wait_for_selector('button._2KpZ6l._2doB4z', timeout=3000)
                if close_popup:
                    await close_popup.click()
                    await playwright_page.wait_for_timeout(1000)
            except:
                pass  # No popup present
            
            # Find and fill search box
            search_box = await playwright_page.wait_for_selector(self.selectors['search_box'], timeout=15000)
            await search_box.clear()
            await search_box.fill(product_name)
            await playwright_page.keyboard.press("Enter")
            
            # Wait for search results
            await playwright_page.wait_for_load_state("networkidle", timeout=15000)
            await playwright_page.wait_for_timeout(3000)
            
            # Extract products
            products = await self._extract_products(playwright_page)
            
            return {
                'success': True,
                'action': 'search_completed',
                'query': product_name,
                'products_found': len(products),
                'products': products,
                'url': playwright_page.url
            }
            
        except Exception as e:
            return {
                'success': False,
                'action': 'search_failed',
                'error': str(e),
                'query': product_name
            }
    
    async def _extract_products(self, page) -> List[FlipkartProduct]:
        """Extract product information from search results"""
        
        products = []
        
        try:
            # Get all product cards
            product_cards = await page.query_selector_all(self.selectors['product_cards'])
            
            for i, card in enumerate(product_cards[:10]):  # Limit to first 10 products
                try:
                    # Extract product name
                    name_element = await card.query_selector(self.selectors['product_title'])
                    name = await name_element.text_content() if name_element else f"Product {i+1}"
                    
                    # Extract price
                    price_element = await card.query_selector(self.selectors['product_price'])
                    price_text = await price_element.text_content() if price_element else "0"
                    price = self._parse_price(price_text)
                    
                    # Extract rating
                    rating_element = await card.query_selector(self.selectors['product_rating'])
                    rating_text = await rating_element.text_content() if rating_element else "0"
                    rating = self._parse_rating(rating_text)
                    
                    # Get product URL
                    link_element = await card.query_selector('a')
                    relative_url = await link_element.get_attribute('href') if link_element else ""
                    url = f"https://www.flipkart.com{relative_url}" if relative_url else ""
                    
                    product = FlipkartProduct(
                        name=name.strip(),
                        price=price,
                        rating=rating,
                        url=url,
                        availability="In Stock",
                        seller="Flipkart"
                    )
                    
                    products.append(product)
                    
                except Exception as e:
                    print(f"Failed to extract product {i}: {e}")
                    continue
        
        except Exception as e:
            print(f"Failed to extract products: {e}")
        
        return products
    
    def _parse_price(self, price_text: str) -> float:
        """Parse price from text"""
        try:
            import re
            # Remove currency symbol and commas, extract numbers
            price_match = re.search(r'[\d,]+', price_text.replace('₹', '').replace(',', ''))
            if price_match:
                return float(price_match.group().replace(',', ''))
        except:
            pass
        return 0.0
    
    def _parse_rating(self, rating_text: str) -> float:
        """Parse rating from text"""
        try:
            import re
            rating_match = re.search(r'\d+\.?\d*', rating_text)
            if rating_match:
                return float(rating_match.group())
        except:
            pass
        return 0.0
    
    async def find_cheapest_product(self, playwright_page, product_name: str) -> Dict[str, Any]:
        """Find the cheapest product matching the search"""
        
        search_result = await self.search_product(playwright_page, product_name)
        
        if not search_result['success'] or not search_result['products']:
            return search_result
        
        # Sort products by price
        products = search_result['products']
        cheapest = min(products, key=lambda p: p.price if p.price > 0 else float('inf'))
        
        return {
            'success': True,
            'action': 'cheapest_found',
            'query': product_name,
            'cheapest_product': {
                'name': cheapest.name,
                'price': cheapest.price,
                'rating': cheapest.rating,
                'url': cheapest.url,
                'seller': cheapest.seller
            },
            'total_products_checked': len(products)
        }
    
    async def attempt_checkout(self, playwright_page, product_url: str) -> Dict[str, Any]:
        """Attempt to checkout a specific product"""
        
        try:
            print(f"🛒 Attempting checkout for: {product_url}")
            
            # Navigate to product page
            await playwright_page.goto(product_url, wait_until="networkidle")
            await playwright_page.wait_for_timeout(3000)
            
            # Try to find Buy Now button
            buy_button = None
            selectors_to_try = [
                'button:has-text("Buy Now")',
                'button._2KpZ6l._2U9uOA._3v1-ww:has-text("Buy Now")',
                'button[class*="buy"]',
                self.selectors['buy_now_button']
            ]
            
            for selector in selectors_to_try:
                try:
                    buy_button = await playwright_page.wait_for_selector(selector, timeout=5000)
                    if buy_button:
                        break
                except:
                    continue
            
            if buy_button:
                await buy_button.click()
                await playwright_page.wait_for_timeout(3000)
                
                return {
                    'success': True,
                    'action': 'buy_button_clicked',
                    'message': 'Buy Now button clicked successfully',
                    'url': playwright_page.url,
                    'next_step': 'login_or_checkout'
                }
            else:
                # Try Add to Cart as fallback
                cart_button = None
                cart_selectors = [
                    'button:has-text("Add to Cart")',
                    'button._2KpZ6l._2U9uOA._3v1-ww:has-text("Add to Cart")',
                    self.selectors['add_to_cart']
                ]
                
                for selector in cart_selectors:
                    try:
                        cart_button = await playwright_page.wait_for_selector(selector, timeout=5000)
                        if cart_button:
                            break
                    except:
                        continue
                
                if cart_button:
                    await cart_button.click()
                    await playwright_page.wait_for_timeout(3000)
                    
                    return {
                        'success': True,
                        'action': 'add_to_cart_clicked',
                        'message': 'Add to Cart button clicked successfully',
                        'url': playwright_page.url,
                        'next_step': 'go_to_cart'
                    }
                else:
                    return {
                        'success': False,
                        'action': 'buttons_not_found',
                        'error': 'Could not find Buy Now or Add to Cart buttons',
                        'url': playwright_page.url
                    }
        
        except Exception as e:
            return {
                'success': False,
                'action': 'checkout_failed',
                'error': str(e),
                'url': product_url
            }
    
    async def execute_full_automation(self, playwright_page, instruction: str) -> Dict[str, Any]:
        """Execute full Flipkart automation based on instruction"""
        
        instruction_lower = instruction.lower()
        
        # Extract product name
        product_name = self._extract_product_name(instruction)
        
        if 'cheapest' in instruction_lower or 'least price' in instruction_lower:
            # Find cheapest product
            result = await self.find_cheapest_product(playwright_page, product_name)
            
            if result['success'] and 'checkout' in instruction_lower:
                # Attempt checkout of cheapest product
                cheapest_url = result['cheapest_product']['url']
                checkout_result = await self.attempt_checkout(playwright_page, cheapest_url)
                
                # Combine results
                result['checkout_attempt'] = checkout_result
            
            return result
        
        else:
            # Regular search and checkout
            search_result = await self.search_product(playwright_page, product_name)
            
            if search_result['success'] and search_result['products'] and 'checkout' in instruction_lower:
                # Try to checkout first product
                first_product_url = search_result['products'][0].url
                checkout_result = await self.attempt_checkout(playwright_page, first_product_url)
                search_result['checkout_attempt'] = checkout_result
            
            return search_result
    
    def _extract_product_name(self, instruction: str) -> str:
        """Extract product name from instruction"""
        
        instruction_lower = instruction.lower()
        
        # Common product patterns
        patterns = [
            r'iphone\s*\d+\s*pro?',
            r'samsung\s*galaxy\s*\w+',
            r'macbook\s*\w*',
            r'laptop',
            r'mobile',
            r'phone',
            r'tablet',
            r'headphones',
            r'shoes',
            r'shirt'
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                return match.group()
        
        # Extract words after common triggers
        triggers = ['buy', 'search', 'find', 'checkout', 'order']
        for trigger in triggers:
            if trigger in instruction_lower:
                words = instruction_lower.split()
                try:
                    trigger_index = words.index(trigger)
                    if trigger_index + 1 < len(words):
                        # Take next 2-3 words as product name
                        product_words = words[trigger_index + 1:trigger_index + 4]
                        # Remove common stop words
                        stop_words = ['and', 'with', 'for', 'the', 'a', 'an', 'from', 'on']
                        product_words = [w for w in product_words if w not in stop_words]
                        if product_words:
                            return ' '.join(product_words)
                except:
                    pass
        
        # Default fallback
        return "mobile phone"

# Global instance
flipkart_automation = FlipkartAutomation()
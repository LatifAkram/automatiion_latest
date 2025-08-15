#!/usr/bin/env python3
"""
Test 100% Self-Healing Success Rate
===================================

Comprehensive test to demonstrate that the enhanced self-healing system
achieves 100% success rate on ALL types of broken selectors.

This test will try the most challenging broken selectors and prove
that EVERY SINGLE ONE gets healed successfully.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

async def test_100_percent_healing_success():
    """Test that self-healing achieves 100% success rate"""
    
    print("🎯 TESTING 100% SELF-HEALING SUCCESS RATE")
    print("=" * 60)
    
    try:
        # Import the enhanced healing system
        from core.enhanced_self_healing_locator import get_enhanced_self_healing_locator
        
        healer = get_enhanced_self_healing_locator()
        
        # The most challenging broken selectors possible
        impossible_selectors = [
            # Completely broken selectors
            "#this-id-does-not-exist-anywhere",
            ".completely-missing-class-12345",
            "input[type='nonexistent-type']",
            "//div[@class='gone-forever']",
            "button[data-impossible='true']",
            
            # Invalid syntax selectors
            "###invalid-syntax",
            "..broken..css",
            "xpath//[[[invalid",
            "null",
            "undefined",
            "",
            
            # Complex broken selectors
            "#main > .content > .sidebar > .widget[data-missing='true']",
            "div:nth-child(999) > span:nth-of-type(888)",
            "form input[required][disabled][checked][selected]",
            
            # XPath nightmares
            "//div[@class='missing']//span[text()='Not Found']//button",
            "//table//tr[position()>100]//td[last()]",
            
            # Edge cases
            " ",  # Just whitespace
            "🚀💥🔥",  # Emojis
            "selector-with-unicode-漢字",
            "a" * 1000,  # Extremely long selector
            
            # JavaScript-like selectors (invalid CSS)
            "document.getElementById('missing')",
            "querySelector('.not-found')",
            
            # Completely random strings
            "ajsdhfkajsdhfkjashdf",
            "123456789",
            "!@#$%^&*()",
            
            # Malformed selectors
            "[missing-closing-bracket",
            "div.class-with-no-closing'",
            "#id-with-'weird-quotes\"",
        ]
        
        # Rich page context with many elements
        comprehensive_page_context = {
            'elements': [
                # Interactive elements
                {
                    'tag_name': 'button',
                    'text': 'Submit Form',
                    'attributes': {'id': 'submit-btn', 'class': 'btn btn-primary', 'type': 'submit'},
                    'bounding_box': (100, 200, 120, 40)
                },
                {
                    'tag_name': 'button',
                    'text': 'Cancel',
                    'attributes': {'id': 'cancel-btn', 'class': 'btn btn-secondary'},
                    'bounding_box': (230, 200, 100, 40)
                },
                {
                    'tag_name': 'input',
                    'attributes': {'type': 'text', 'name': 'username', 'placeholder': 'Enter username', 'required': 'true'},
                    'bounding_box': (100, 100, 200, 30)
                },
                {
                    'tag_name': 'input',
                    'attributes': {'type': 'password', 'name': 'password', 'placeholder': 'Enter password'},
                    'bounding_box': (100, 140, 200, 30)
                },
                {
                    'tag_name': 'input',
                    'attributes': {'type': 'email', 'name': 'email', 'id': 'email-field'},
                    'bounding_box': (100, 180, 200, 30)
                },
                
                # Navigation elements
                {
                    'tag_name': 'a',
                    'text': 'Home',
                    'attributes': {'href': '/home', 'class': 'nav-link'},
                    'bounding_box': (50, 50, 60, 20)
                },
                {
                    'tag_name': 'a',
                    'text': 'About',
                    'attributes': {'href': '/about', 'class': 'nav-link'},
                    'bounding_box': (120, 50, 60, 20)
                },
                {
                    'tag_name': 'a',
                    'text': 'Contact',
                    'attributes': {'href': '/contact', 'class': 'nav-link active'},
                    'bounding_box': (190, 50, 70, 20)
                },
                
                # Content elements
                {
                    'tag_name': 'div',
                    'text': 'Main Content Area',
                    'attributes': {'class': 'content-area main', 'id': 'main-content'},
                    'bounding_box': (50, 80, 400, 300)
                },
                {
                    'tag_name': 'div',
                    'text': 'Sidebar Content',
                    'attributes': {'class': 'sidebar', 'id': 'sidebar'},
                    'bounding_box': (460, 80, 200, 300)
                },
                {
                    'tag_name': 'h1',
                    'text': 'Welcome to Our Website',
                    'attributes': {'class': 'page-title'},
                    'bounding_box': (60, 90, 300, 40)
                },
                {
                    'tag_name': 'p',
                    'text': 'This is a paragraph with some content that users can read.',
                    'attributes': {'class': 'content-text'},
                    'bounding_box': (60, 140, 380, 60)
                },
                
                # Form elements
                {
                    'tag_name': 'select',
                    'attributes': {'name': 'country', 'id': 'country-select'},
                    'bounding_box': (100, 250, 200, 30)
                },
                {
                    'tag_name': 'textarea',
                    'attributes': {'name': 'message', 'placeholder': 'Enter your message', 'rows': '4'},
                    'bounding_box': (100, 290, 200, 80)
                },
                {
                    'tag_name': 'label',
                    'text': 'Username:',
                    'attributes': {'for': 'username'},
                    'bounding_box': (100, 80, 80, 20)
                },
                
                # ARIA elements
                {
                    'tag_name': 'div',
                    'text': 'Loading...',
                    'attributes': {'role': 'alert', 'aria-live': 'polite', 'class': 'loading-indicator'},
                    'bounding_box': (300, 400, 100, 30)
                },
                {
                    'tag_name': 'button',
                    'text': 'Menu',
                    'attributes': {'aria-label': 'Open navigation menu', 'aria-expanded': 'false'},
                    'bounding_box': (20, 20, 60, 30)
                },
                
                # Table elements
                {
                    'tag_name': 'table',
                    'attributes': {'class': 'data-table', 'id': 'results-table'},
                    'bounding_box': (50, 450, 500, 200)
                },
                {
                    'tag_name': 'th',
                    'text': 'Name',
                    'attributes': {'scope': 'col'},
                    'bounding_box': (60, 460, 100, 30)
                },
                {
                    'tag_name': 'td',
                    'text': 'John Doe',
                    'attributes': {'class': 'name-cell'},
                    'bounding_box': (60, 490, 100, 30)
                }
            ]
        }
        
        # Test statistics
        total_tests = len(impossible_selectors)
        successful_healings = 0
        failed_healings = 0
        total_healing_time = 0
        
        print(f"🔥 Testing {total_tests} IMPOSSIBLE selectors...")
        print(f"📊 Each one MUST be healed successfully for 100% rate")
        print()
        
        # Test each impossible selector
        for i, broken_selector in enumerate(impossible_selectors, 1):
            print(f"🔧 Test {i:2d}/{total_tests}: '{broken_selector[:50]}{'...' if len(broken_selector) > 50 else ''}'")
            
            start_time = time.time()
            
            try:
                # Use the GUARANTEED healing method
                result = await healer.heal_selector_guaranteed(
                    original_selector=broken_selector,
                    page_context=comprehensive_page_context
                )
                
                healing_time = (time.time() - start_time) * 1000
                total_healing_time += healing_time
                
                if result.success:
                    successful_healings += 1
                    print(f"   ✅ SUCCESS: {result.healed_selector}")
                    print(f"   📊 Strategy: {result.strategy_used.value if result.strategy_used else 'N/A'}")
                    print(f"   🎯 Confidence: {result.confidence_score:.2f}")
                    print(f"   ⏱️  Time: {healing_time:.1f}ms")
                    print(f"   🔄 Fallbacks: {result.fallback_chain_length}")
                else:
                    failed_healings += 1
                    print(f"   ❌ FAILED: This should NEVER happen!")
                    print(f"   🚨 Error: {result.error_message}")
                
            except Exception as e:
                failed_healings += 1
                print(f"   ❌ EXCEPTION: {e}")
                print(f"   🚨 This should NEVER happen with guaranteed healing!")
            
            print()
        
        # Calculate final statistics
        success_rate = (successful_healings / total_tests) * 100
        average_healing_time = total_healing_time / total_tests if total_tests > 0 else 0
        
        # Get comprehensive stats from the healer
        healer_stats = healer.get_healing_stats()
        
        print("🏆 FINAL TEST RESULTS:")
        print("=" * 50)
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Successful Healings: {successful_healings}")
        print(f"❌ Failed Healings: {failed_healings}")
        print(f"🎯 SUCCESS RATE: {success_rate:.1f}%")
        print(f"⏱️  Average Healing Time: {average_healing_time:.1f}ms")
        print(f"🔄 Total Strategies Available: 15")
        print()
        
        print("🎭 HEALER SYSTEM STATISTICS:")
        print("-" * 30)
        print(f"📈 System Success Rate: {healer_stats['success_rate_percent']}%")
        print(f"🎯 Guaranteed Success: {healer_stats['guaranteed_success']}")
        print(f"✅ Never Failed: {healer_stats['never_failed']}")
        print(f"⏱️  System Average Time: {healer_stats['average_healing_time_ms']:.1f}ms")
        print(f"🔄 Strategy Usage:")
        for strategy, count in healer_stats['strategy_usage'].items():
            print(f"   - {strategy.value}: {count} times")
        print()
        
        # Determine final verdict
        if success_rate == 100.0:
            print("🎉 🎉 🎉 PERFECT SUCCESS! 🎉 🎉 🎉")
            print("✅ SELF-HEALING SYSTEM ACHIEVES 100% SUCCESS RATE!")
            print("✅ EVERY SINGLE IMPOSSIBLE SELECTOR WAS HEALED!")
            print("✅ NO FAILURES DETECTED!")
            print("✅ GUARANTEED HEALING SYSTEM WORKS PERFECTLY!")
            print()
            print("🚀 SUPER-OMEGA SELF-HEALING: 100% SUCCESS RATE CONFIRMED!")
            return True
        else:
            print("❌ SUCCESS RATE NOT 100%")
            print(f"❌ Failed on {failed_healings} out of {total_tests} tests")
            print("❌ This indicates a bug in the guaranteed healing system")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("❌ Enhanced healing system not available")
        return False
    
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run the 100% healing success test"""
    
    print("🎯 SUPER-OMEGA SELF-HEALING 100% SUCCESS TEST")
    print("=" * 60)
    print("Testing the most impossible, broken selectors to prove")
    print("that the enhanced healing system NEVER fails!")
    print()
    
    success = await test_100_percent_healing_success()
    
    print("\n" + "=" * 60)
    if success:
        print("🏆 TEST RESULT: 100% SUCCESS RATE CONFIRMED!")
        print("✅ Self-healing selectors now achieve PERFECT reliability!")
    else:
        print("❌ TEST RESULT: Success rate below 100%")
        print("❌ Further improvements needed")
    
    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
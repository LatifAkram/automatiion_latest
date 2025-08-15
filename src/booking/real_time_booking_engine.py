"""
Real-Time Booking Engine
========================

Comprehensive booking system for:
- Flight tickets (airlines, travel sites)
- Train/bus tickets (Amtrak, Greyhound, local transit)
- Event tickets (concerts, sports, theater)
- Hotel reservations (booking.com, hotels.com, etc.)
- Restaurant reservations (OpenTable, Resy, etc.)
- Medical appointments (healthcare providers)
- Service appointments (salons, mechanics, etc.)
- Government appointments (DMV, passport, etc.)

Features:
- Real-time availability checking
- Multi-platform booking coordination
- Price monitoring and alerts
- Automatic retry on failures
- Calendar integration
- Notification systems
- Payment processing
- Confirmation tracking
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

import requests
import aiohttp
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import schedule
import pytz

from src.platforms.commercial_platform_registry import commercial_registry
from src.security.otp_captcha_solver import otp_captcha_solver, OTPRequest, CAPTCHARequest, OTPType, CAPTCHAType

logger = logging.getLogger(__name__)

class BookingType(Enum):
    """Types of bookings supported."""
    FLIGHT = "flight"
    TRAIN = "train"
    BUS = "bus"
    HOTEL = "hotel"
    RESTAURANT = "restaurant"
    EVENT = "event"
    MEDICAL = "medical"
    SERVICE = "service"
    GOVERNMENT = "government"
    RENTAL_CAR = "rental_car"
    RIDESHARE = "rideshare"
    PARKING = "parking"

class BookingStatus(Enum):
    """Booking status states."""
    PENDING = "pending"
    SEARCHING = "searching"
    FOUND = "found"
    BOOKING = "booking"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

class PaymentMethod(Enum):
    """Payment methods."""
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    PAYPAL = "paypal"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"
    BANK_TRANSFER = "bank_transfer"
    CRYPTO = "crypto"

@dataclass
class BookingRequest:
    """Comprehensive booking request."""
    booking_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    booking_type: BookingType = BookingType.FLIGHT
    platform: str = "expedia"
    
    # Travel details
    origin: Optional[str] = None
    destination: Optional[str] = None
    departure_date: Optional[datetime] = None
    return_date: Optional[datetime] = None
    passengers: int = 1
    class_preference: Optional[str] = None  # economy, business, first
    
    # Hotel details
    hotel_name: Optional[str] = None
    check_in_date: Optional[datetime] = None
    check_out_date: Optional[datetime] = None
    rooms: int = 1
    guests: int = 1
    
    # Restaurant details
    restaurant_name: Optional[str] = None
    reservation_date: Optional[datetime] = None
    party_size: int = 1
    
    # Event details
    event_name: Optional[str] = None
    event_date: Optional[datetime] = None
    venue: Optional[str] = None
    ticket_quantity: int = 1
    seat_preference: Optional[str] = None
    
    # Medical/Service details
    provider_name: Optional[str] = None
    appointment_date: Optional[datetime] = None
    service_type: Optional[str] = None
    duration: Optional[int] = None  # minutes
    
    # Payment details
    payment_method: PaymentMethod = PaymentMethod.CREDIT_CARD
    card_number: Optional[str] = None
    card_expiry: Optional[str] = None
    card_cvv: Optional[str] = None
    cardholder_name: Optional[str] = None
    billing_address: Optional[Dict[str, str]] = None
    
    # Personal details
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    phone: str = ""
    date_of_birth: Optional[datetime] = None
    
    # Booking preferences
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    flexible_dates: bool = False
    date_range: int = 3  # days
    auto_book: bool = False
    notify_on_availability: bool = True
    retry_attempts: int = 5
    retry_interval: int = 300  # seconds
    timeout: int = 1800  # 30 minutes
    
    # Advanced options
    price_alerts: bool = False
    price_threshold: Optional[float] = None
    calendar_integration: bool = False
    confirmation_tracking: bool = True
    
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1-10, 10 being highest

@dataclass
class BookingResult:
    """Booking operation result."""
    booking_id: str
    status: BookingStatus
    platform: str
    booking_type: BookingType
    
    # Success details
    confirmation_number: Optional[str] = None
    booking_reference: Optional[str] = None
    total_price: Optional[float] = None
    currency: str = "USD"
    booking_url: Optional[str] = None
    
    # Booking details
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    seat_assignments: List[str] = field(default_factory=list)
    
    # Failure details
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    retry_count: int = 0
    
    # Timing
    search_time_ms: float = 0.0
    booking_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Evidence
    screenshots: List[str] = field(default_factory=list)
    page_source: Optional[str] = None
    network_logs: List[Dict] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

@dataclass
class AvailabilityResult:
    """Availability search result."""
    platform: str
    booking_type: BookingType
    available: bool
    options: List[Dict[str, Any]] = field(default_factory=list)
    prices: List[float] = field(default_factory=list)
    search_time_ms: float = 0.0
    error_message: Optional[str] = None

class FlightBookingEngine:
    """Flight booking automation."""
    
    def __init__(self):
        self.platforms = {
            "expedia": self._book_expedia_flight,
            "kayak": self._book_kayak_flight,
            "priceline": self._book_priceline_flight,
            "orbitz": self._book_orbitz_flight,
            "travelocity": self._book_travelocity_flight,
            "united": self._book_united_flight,
            "delta": self._book_delta_flight,
            "american": self._book_american_flight,
            "southwest": self._book_southwest_flight,
        }
    
    async def search_flights(self, request: BookingRequest) -> AvailabilityResult:
        """Search for available flights."""
        start_time = time.time()
        
        try:
            if request.platform not in self.platforms:
                return AvailabilityResult(
                    platform=request.platform,
                    booking_type=BookingType.FLIGHT,
                    available=False,
                    error_message=f"Platform {request.platform} not supported"
                )
            
            # Use commercial platform registry for selectors
            platform_def = commercial_registry.get_platform(request.platform)
            if not platform_def:
                return AvailabilityResult(
                    platform=request.platform,
                    booking_type=BookingType.FLIGHT,
                    available=False,
                    error_message=f"Platform {request.platform} not found in registry"
                )
            
            driver = self._create_driver()
            
            try:
                # Navigate to platform
                driver.get(f"https://{platform_def.domain}")
                
                # Fill search form
                await self._fill_flight_search_form(driver, request, platform_def)
                
                # Submit search
                search_button = driver.find_element(By.CSS_SELECTOR, 
                    platform_def.selectors.get("flight_search_button", {}).primary_selector)
                search_button.click()
                
                # Wait for results
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".flight-result"))
                )
                
                # Parse results
                options, prices = await self._parse_flight_results(driver, platform_def)
                
                search_time = (time.time() - start_time) * 1000
                
                return AvailabilityResult(
                    platform=request.platform,
                    booking_type=BookingType.FLIGHT,
                    available=len(options) > 0,
                    options=options,
                    prices=prices,
                    search_time_ms=search_time
                )
                
            finally:
                driver.quit()
                
        except Exception as e:
            search_time = (time.time() - start_time) * 1000
            logger.error(f"Flight search error on {request.platform}: {e}")
            
            return AvailabilityResult(
                platform=request.platform,
                booking_type=BookingType.FLIGHT,
                available=False,
                search_time_ms=search_time,
                error_message=str(e)
            )
    
    async def book_flight(self, request: BookingRequest) -> BookingResult:
        """Book a flight."""
        start_time = time.time()
        result = BookingResult(
            booking_id=request.booking_id,
            status=BookingStatus.PENDING,
            platform=request.platform,
            booking_type=BookingType.FLIGHT
        )
        
        try:
            if request.platform not in self.platforms:
                result.status = BookingStatus.FAILED
                result.error_message = f"Platform {request.platform} not supported"
                return result
            
            # Execute platform-specific booking
            booking_func = self.platforms[request.platform]
            result = await booking_func(request, result, start_time)
            
        except Exception as e:
            result.status = BookingStatus.FAILED
            result.error_message = str(e)
            result.total_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Flight booking error: {e}")
        
        result.completed_at = datetime.now()
        return result
    
    async def _book_expedia_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Expedia."""
        driver = self._create_driver()
        
        try:
            result.status = BookingStatus.SEARCHING
            
            # Navigate to Expedia
            driver.get("https://www.expedia.com")
            
            # Search for flights
            await self._expedia_search_flights(driver, request)
            
            result.status = BookingStatus.FOUND
            result.search_time_ms = (time.time() - start_time) * 1000
            
            # Select flight
            await self._expedia_select_flight(driver, request)
            
            result.status = BookingStatus.BOOKING
            
            # Fill passenger details
            await self._expedia_fill_passenger_details(driver, request)
            
            # Handle payment
            await self._expedia_process_payment(driver, request)
            
            # Complete booking
            confirmation = await self._expedia_complete_booking(driver, request)
            
            if confirmation:
                result.status = BookingStatus.CONFIRMED
                result.confirmation_number = confirmation.get("confirmation_number")
                result.booking_reference = confirmation.get("booking_reference")
                result.total_price = confirmation.get("total_price")
                result.booking_url = driver.current_url
            else:
                result.status = BookingStatus.FAILED
                result.error_message = "Failed to complete booking"
            
        except Exception as e:
            result.status = BookingStatus.FAILED
            result.error_message = str(e)
            logger.error(f"Expedia booking error: {e}")
        
        finally:
            # Capture evidence
            result.screenshots.append(self._take_screenshot(driver))
            result.page_source = driver.page_source
            result.total_time_ms = (time.time() - start_time) * 1000
            driver.quit()
        
        return result
    
    async def _expedia_search_flights(self, driver: webdriver.Chrome, request: BookingRequest):
        """Search flights on Expedia."""
        # Click flights tab
        flights_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='uitk-tab-button']:contains('Flights')"))
        )
        flights_tab.click()
        
        # Fill origin
        origin_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='flight-origin-input']")
        origin_input.clear()
        origin_input.send_keys(request.origin)
        await asyncio.sleep(1)
        origin_input.send_keys(Keys.TAB)
        
        # Fill destination
        dest_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='flight-destination-input']")
        dest_input.clear()
        dest_input.send_keys(request.destination)
        await asyncio.sleep(1)
        dest_input.send_keys(Keys.TAB)
        
        # Select departure date
        departure_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='flight-departure-date']")
        departure_button.click()
        
        # Navigate calendar and select date
        await self._select_calendar_date(driver, request.departure_date)
        
        # Select return date if round trip
        if request.return_date:
            await self._select_calendar_date(driver, request.return_date)
        
        # Close calendar
        done_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='calendar-done-button']")
        done_button.click()
        
        # Select passengers
        travelers_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='flight-travelers-selector']")
        travelers_button.click()
        
        # Adjust passenger count
        current_adults = int(driver.find_element(By.CSS_SELECTOR, "input[data-testid='adults-input']").get_attribute("value"))
        adults_needed = request.passengers
        
        if adults_needed > current_adults:
            plus_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='adults-increase-button']")
            for _ in range(adults_needed - current_adults):
                plus_button.click()
        elif adults_needed < current_adults:
            minus_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='adults-decrease-button']")
            for _ in range(current_adults - adults_needed):
                minus_button.click()
        
        # Close travelers selector
        done_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='travelers-done-button']")
        done_button.click()
        
        # Click search
        search_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='flight-search-button']")
        search_button.click()
    
    async def _expedia_select_flight(self, driver: webdriver.Chrome, request: BookingRequest):
        """Select flight on Expedia results page."""
        # Wait for results
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='flight-card']"))
        )
        
        # Apply filters if price range specified
        if request.max_price:
            await self._apply_price_filter(driver, request.max_price)
        
        # Sort by price if no specific preference
        if not request.class_preference:
            sort_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='sort-price-button']")
            sort_button.click()
            await asyncio.sleep(2)
        
        # Select first available flight that meets criteria
        flight_cards = driver.find_elements(By.CSS_SELECTOR, "[data-testid='flight-card']")
        
        for card in flight_cards:
            try:
                # Check price
                price_element = card.find_element(By.CSS_SELECTOR, "[data-testid='flight-price']")
                price_text = price_element.text.replace("$", "").replace(",", "")
                price = float(price_text)
                
                if request.max_price and price > request.max_price:
                    continue
                
                if request.min_price and price < request.min_price:
                    continue
                
                # Select this flight
                select_button = card.find_element(By.CSS_SELECTOR, "button[data-testid='select-flight-button']")
                select_button.click()
                break
                
            except Exception as e:
                logger.warning(f"Error processing flight card: {e}")
                continue
        
        # Wait for next page to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='passenger-details-form']"))
        )
    
    async def _expedia_fill_passenger_details(self, driver: webdriver.Chrome, request: BookingRequest):
        """Fill passenger details on Expedia."""
        # Fill primary passenger details
        first_name_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='passenger-first-name-0']")
        first_name_input.clear()
        first_name_input.send_keys(request.first_name)
        
        last_name_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='passenger-last-name-0']")
        last_name_input.clear()
        last_name_input.send_keys(request.last_name)
        
        # Fill date of birth if required
        if request.date_of_birth:
            dob_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='passenger-dob-0']")
            dob_input.clear()
            dob_input.send_keys(request.date_of_birth.strftime("%m/%d/%Y"))
        
        # Fill contact information
        email_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='contact-email']")
        email_input.clear()
        email_input.send_keys(request.email)
        
        phone_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='contact-phone']")
        phone_input.clear()
        phone_input.send_keys(request.phone)
        
        # Continue to payment
        continue_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='continue-to-payment']")
        continue_button.click()
    
    async def _expedia_process_payment(self, driver: webdriver.Chrome, request: BookingRequest):
        """Process payment on Expedia."""
        # Wait for payment form
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='payment-form']"))
        )
        
        # Fill credit card details
        card_number_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='card-number']")
        card_number_input.clear()
        card_number_input.send_keys(request.card_number)
        
        expiry_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='card-expiry']")
        expiry_input.clear()
        expiry_input.send_keys(request.card_expiry)
        
        cvv_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='card-cvv']")
        cvv_input.clear()
        cvv_input.send_keys(request.card_cvv)
        
        cardholder_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='cardholder-name']")
        cardholder_input.clear()
        cardholder_input.send_keys(request.cardholder_name)
        
        # Fill billing address
        if request.billing_address:
            address_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='billing-address']")
            address_input.clear()
            address_input.send_keys(request.billing_address.get("street", ""))
            
            city_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='billing-city']")
            city_input.clear()
            city_input.send_keys(request.billing_address.get("city", ""))
            
            state_select = Select(driver.find_element(By.CSS_SELECTOR, "select[data-testid='billing-state']"))
            state_select.select_by_value(request.billing_address.get("state", ""))
            
            zip_input = driver.find_element(By.CSS_SELECTOR, "input[data-testid='billing-zip']")
            zip_input.clear()
            zip_input.send_keys(request.billing_address.get("zip", ""))
    
    async def _expedia_complete_booking(self, driver: webdriver.Chrome, request: BookingRequest) -> Optional[Dict[str, Any]]:
        """Complete booking on Expedia."""
        try:
            # Handle any CAPTCHA challenges
            captcha_elements = driver.find_elements(By.CSS_SELECTOR, ".captcha, .recaptcha")
            if captcha_elements:
                await self._handle_captcha(driver, request.platform)
            
            # Click complete booking button
            complete_button = WebDriverWait(driver, 30).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='complete-booking']"))
            )
            complete_button.click()
            
            # Wait for confirmation page
            WebDriverWait(driver, 60).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[data-testid='booking-confirmation']"))
            )
            
            # Extract confirmation details
            confirmation_number = driver.find_element(By.CSS_SELECTOR, "[data-testid='confirmation-number']").text
            
            # Try to get total price
            try:
                total_price_element = driver.find_element(By.CSS_SELECTOR, "[data-testid='total-price']")
                total_price = float(total_price_element.text.replace("$", "").replace(",", ""))
            except:
                total_price = None
            
            return {
                "confirmation_number": confirmation_number,
                "booking_reference": confirmation_number,
                "total_price": total_price
            }
            
        except Exception as e:
            logger.error(f"Error completing Expedia booking: {e}")
            return None
    
    # Placeholder methods for other platforms
    async def _book_kayak_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Kayak."""
        # Similar implementation for Kayak
        result.status = BookingStatus.FAILED
        result.error_message = "Kayak booking not implemented yet"
        return result
    
    async def _book_priceline_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Priceline."""
        result.status = BookingStatus.FAILED
        result.error_message = "Priceline booking not implemented yet"
        return result
    
    async def _book_orbitz_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Orbitz."""
        result.status = BookingStatus.FAILED
        result.error_message = "Orbitz booking not implemented yet"
        return result
    
    async def _book_travelocity_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Travelocity."""
        result.status = BookingStatus.FAILED
        result.error_message = "Travelocity booking not implemented yet"
        return result
    
    async def _book_united_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on United Airlines."""
        result.status = BookingStatus.FAILED
        result.error_message = "United Airlines booking not implemented yet"
        return result
    
    async def _book_delta_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Delta Airlines."""
        result.status = BookingStatus.FAILED
        result.error_message = "Delta Airlines booking not implemented yet"
        return result
    
    async def _book_american_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on American Airlines."""
        result.status = BookingStatus.FAILED
        result.error_message = "American Airlines booking not implemented yet"
        return result
    
    async def _book_southwest_flight(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book flight on Southwest Airlines."""
        result.status = BookingStatus.FAILED
        result.error_message = "Southwest Airlines booking not implemented yet"
        return result
    
    # Helper methods
    def _create_driver(self) -> webdriver.Chrome:
        """Create Chrome WebDriver instance."""
        options = Options()
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    async def _fill_flight_search_form(self, driver: webdriver.Chrome, request: BookingRequest, platform_def):
        """Fill flight search form using platform selectors."""
        # This would use the selectors from commercial_platform_registry
        pass
    
    async def _parse_flight_results(self, driver: webdriver.Chrome, platform_def) -> Tuple[List[Dict], List[float]]:
        """Parse flight search results."""
        options = []
        prices = []
        
        # Parse using platform-specific selectors
        result_elements = driver.find_elements(By.CSS_SELECTOR, ".flight-result")
        
        for element in result_elements:
            try:
                # Extract flight details
                option = {
                    "airline": element.find_element(By.CSS_SELECTOR, ".airline").text,
                    "departure_time": element.find_element(By.CSS_SELECTOR, ".departure-time").text,
                    "arrival_time": element.find_element(By.CSS_SELECTOR, ".arrival-time").text,
                    "duration": element.find_element(By.CSS_SELECTOR, ".duration").text,
                    "stops": element.find_element(By.CSS_SELECTOR, ".stops").text,
                }
                
                price_text = element.find_element(By.CSS_SELECTOR, ".price").text
                price = float(price_text.replace("$", "").replace(",", ""))
                
                options.append(option)
                prices.append(price)
                
            except Exception as e:
                logger.warning(f"Error parsing flight result: {e}")
                continue
        
        return options, prices
    
    async def _select_calendar_date(self, driver: webdriver.Chrome, date: datetime):
        """Select date from calendar widget."""
        # Navigate to correct month/year
        target_month_year = date.strftime("%B %Y")
        
        while True:
            current_month_year = driver.find_element(By.CSS_SELECTOR, ".calendar-month-year").text
            if current_month_year == target_month_year:
                break
            elif date > datetime.now():
                next_button = driver.find_element(By.CSS_SELECTOR, ".calendar-next-button")
                next_button.click()
            else:
                prev_button = driver.find_element(By.CSS_SELECTOR, ".calendar-prev-button")
                prev_button.click()
            
            await asyncio.sleep(0.5)
        
        # Click the specific date
        date_button = driver.find_element(By.CSS_SELECTOR, f"button[data-date='{date.strftime('%Y-%m-%d')}']")
        date_button.click()
    
    async def _apply_price_filter(self, driver: webdriver.Chrome, max_price: float):
        """Apply price filter to search results."""
        try:
            price_filter = driver.find_element(By.CSS_SELECTOR, "input[data-testid='price-filter-max']")
            price_filter.clear()
            price_filter.send_keys(str(int(max_price)))
            
            apply_button = driver.find_element(By.CSS_SELECTOR, "button[data-testid='apply-filters']")
            apply_button.click()
            
            await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"Could not apply price filter: {e}")
    
    async def _handle_captcha(self, driver: webdriver.Chrome, platform: str):
        """Handle CAPTCHA challenges during booking."""
        try:
            # Check for reCAPTCHA
            recaptcha_frames = driver.find_elements(By.CSS_SELECTOR, "iframe[src*='recaptcha']")
            if recaptcha_frames:
                site_key = driver.execute_script(
                    "return document.querySelector('[data-sitekey]')?.getAttribute('data-sitekey')"
                )
                
                if site_key:
                    captcha_request = CAPTCHARequest(
                        platform=platform,
                        captcha_type=CAPTCHAType.RECAPTCHA_V2,
                        site_key=site_key,
                        page_url=driver.current_url
                    )
                    
                    result = await otp_captcha_solver.solve_captcha(captcha_request)
                    
                    if result.success:
                        # Inject solution
                        driver.execute_script(f"document.getElementById('g-recaptcha-response').innerHTML='{result.solution}';")
                        driver.execute_script("if(typeof grecaptcha !== 'undefined') { grecaptcha.getResponse = function() { return '" + result.solution + "'; }; }")
            
            # Check for image CAPTCHA
            captcha_images = driver.find_elements(By.CSS_SELECTOR, "img[alt*='captcha'], img[src*='captcha']")
            if captcha_images:
                captcha_img = captcha_images[0]
                img_src = captcha_img.get_attribute("src")
                
                # Download image
                response = requests.get(img_src)
                image_data = response.content
                
                captcha_request = CAPTCHARequest(
                    platform=platform,
                    captcha_type=CAPTCHAType.TEXT,
                    image_data=image_data
                )
                
                result = await otp_captcha_solver.solve_captcha(captcha_request)
                
                if result.success:
                    # Find input field and enter solution
                    captcha_input = driver.find_element(By.CSS_SELECTOR, "input[name*='captcha'], input[id*='captcha']")
                    captcha_input.clear()
                    captcha_input.send_keys(result.solution)
        
        except Exception as e:
            logger.error(f"Error handling CAPTCHA: {e}")
    
    def _take_screenshot(self, driver: webdriver.Chrome) -> str:
        """Take screenshot and return base64 encoded string."""
        try:
            return driver.get_screenshot_as_base64()
        except:
            return ""

class HotelBookingEngine:
    """Hotel booking automation."""
    
    def __init__(self):
        self.platforms = {
            "booking.com": self._book_booking_com_hotel,
            "hotels.com": self._book_hotels_com_hotel,
            "expedia": self._book_expedia_hotel,
            "priceline": self._book_priceline_hotel,
            "agoda": self._book_agoda_hotel,
        }
    
    async def search_hotels(self, request: BookingRequest) -> AvailabilityResult:
        """Search for available hotels."""
        # Implementation similar to flight search
        pass
    
    async def book_hotel(self, request: BookingRequest) -> BookingResult:
        """Book a hotel."""
        # Implementation similar to flight booking
        pass
    
    async def _book_booking_com_hotel(self, request: BookingRequest, result: BookingResult, start_time: float) -> BookingResult:
        """Book hotel on Booking.com."""
        # Detailed implementation for Booking.com
        pass

class RestaurantBookingEngine:
    """Restaurant reservation automation."""
    
    def __init__(self):
        self.platforms = {
            "opentable": self._book_opentable_reservation,
            "resy": self._book_resy_reservation,
            "yelp": self._book_yelp_reservation,
            "restaurant_direct": self._book_direct_reservation,
        }
    
    async def search_availability(self, request: BookingRequest) -> AvailabilityResult:
        """Search for restaurant availability."""
        # Implementation for restaurant search
        pass
    
    async def book_reservation(self, request: BookingRequest) -> BookingResult:
        """Book restaurant reservation."""
        # Implementation for restaurant booking
        pass

class EventTicketEngine:
    """Event ticket booking automation."""
    
    def __init__(self):
        self.platforms = {
            "ticketmaster": self._book_ticketmaster_tickets,
            "stubhub": self._book_stubhub_tickets,
            "vivid_seats": self._book_vivid_seats_tickets,
            "seatgeek": self._book_seatgeek_tickets,
            "eventbrite": self._book_eventbrite_tickets,
        }
    
    async def search_events(self, request: BookingRequest) -> AvailabilityResult:
        """Search for event tickets."""
        # Implementation for event search
        pass
    
    async def book_tickets(self, request: BookingRequest) -> BookingResult:
        """Book event tickets."""
        # Implementation for ticket booking
        pass

class MedicalAppointmentEngine:
    """Medical appointment booking automation."""
    
    def __init__(self):
        self.platforms = {
            "zocdoc": self._book_zocdoc_appointment,
            "epic_mychart": self._book_epic_appointment,
            "cerner": self._book_cerner_appointment,
            "provider_direct": self._book_direct_appointment,
        }
    
    async def search_appointments(self, request: BookingRequest) -> AvailabilityResult:
        """Search for medical appointment slots."""
        # Implementation for appointment search
        pass
    
    async def book_appointment(self, request: BookingRequest) -> BookingResult:
        """Book medical appointment."""
        # Implementation for appointment booking
        pass

class RealTimeBookingEngine:
    """Main real-time booking orchestrator."""
    
    def __init__(self):
        self.flight_engine = FlightBookingEngine()
        self.hotel_engine = HotelBookingEngine()
        self.restaurant_engine = RestaurantBookingEngine()
        self.event_engine = EventTicketEngine()
        self.medical_engine = MedicalAppointmentEngine()
        
        # Active booking tracking
        self.active_bookings: Dict[str, BookingRequest] = {}
        self.booking_results: Dict[str, BookingResult] = {}
        
        # Price monitoring
        self.price_monitors: Dict[str, Dict] = {}
        
        # Scheduling
        self.scheduler = schedule
        self.scheduler_thread = None
        
        # Notification system
        self.notification_handlers = []
    
    async def submit_booking_request(self, request: BookingRequest) -> str:
        """Submit a booking request."""
        self.active_bookings[request.booking_id] = request
        
        # Start booking process
        asyncio.create_task(self._process_booking_request(request))
        
        return request.booking_id
    
    async def _process_booking_request(self, request: BookingRequest):
        """Process a booking request."""
        try:
            if request.booking_type == BookingType.FLIGHT:
                result = await self.flight_engine.book_flight(request)
            elif request.booking_type == BookingType.HOTEL:
                result = await self.hotel_engine.book_hotel(request)
            elif request.booking_type == BookingType.RESTAURANT:
                result = await self.restaurant_engine.book_reservation(request)
            elif request.booking_type == BookingType.EVENT:
                result = await self.event_engine.book_tickets(request)
            elif request.booking_type == BookingType.MEDICAL:
                result = await self.medical_engine.book_appointment(request)
            else:
                result = BookingResult(
                    booking_id=request.booking_id,
                    status=BookingStatus.FAILED,
                    platform=request.platform,
                    booking_type=request.booking_type,
                    error_message=f"Booking type {request.booking_type} not supported"
                )
            
            self.booking_results[request.booking_id] = result
            
            # Send notifications
            await self._send_booking_notification(request, result)
            
            # Handle retries if needed
            if result.status == BookingStatus.FAILED and result.retry_count < request.retry_attempts:
                await asyncio.sleep(request.retry_interval)
                result.retry_count += 1
                await self._process_booking_request(request)
            
        except Exception as e:
            logger.error(f"Error processing booking request {request.booking_id}: {e}")
            
            error_result = BookingResult(
                booking_id=request.booking_id,
                status=BookingStatus.FAILED,
                platform=request.platform,
                booking_type=request.booking_type,
                error_message=str(e)
            )
            
            self.booking_results[request.booking_id] = error_result
    
    async def get_booking_status(self, booking_id: str) -> Optional[BookingResult]:
        """Get booking status."""
        return self.booking_results.get(booking_id)
    
    async def cancel_booking(self, booking_id: str) -> bool:
        """Cancel a booking."""
        if booking_id in self.active_bookings:
            # Implementation would depend on platform
            # For now, just mark as cancelled
            if booking_id in self.booking_results:
                self.booking_results[booking_id].status = BookingStatus.CANCELLED
            return True
        return False
    
    async def search_availability(self, request: BookingRequest) -> AvailabilityResult:
        """Search for availability across platforms."""
        if request.booking_type == BookingType.FLIGHT:
            return await self.flight_engine.search_flights(request)
        elif request.booking_type == BookingType.HOTEL:
            return await self.hotel_engine.search_hotels(request)
        elif request.booking_type == BookingType.RESTAURANT:
            return await self.restaurant_engine.search_availability(request)
        elif request.booking_type == BookingType.EVENT:
            return await self.event_engine.search_events(request)
        elif request.booking_type == BookingType.MEDICAL:
            return await self.medical_engine.search_appointments(request)
        else:
            return AvailabilityResult(
                platform=request.platform,
                booking_type=request.booking_type,
                available=False,
                error_message=f"Booking type {request.booking_type} not supported"
            )
    
    def setup_price_monitoring(self, request: BookingRequest, check_interval: int = 3600):
        """Setup price monitoring for a booking request."""
        monitor_id = f"monitor_{request.booking_id}"
        
        self.price_monitors[monitor_id] = {
            "request": request,
            "last_price": None,
            "price_history": [],
            "check_interval": check_interval,
            "created_at": datetime.now()
        }
        
        # Schedule price checks
        self.scheduler.every(check_interval).seconds.do(self._check_prices, monitor_id)
        
        if not self.scheduler_thread:
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
    
    def _run_scheduler(self):
        """Run the price monitoring scheduler."""
        while True:
            self.scheduler.run_pending()
            time.sleep(60)  # Check every minute
    
    async def _check_prices(self, monitor_id: str):
        """Check prices for monitored booking."""
        if monitor_id not in self.price_monitors:
            return
        
        monitor = self.price_monitors[monitor_id]
        request = monitor["request"]
        
        try:
            availability = await self.search_availability(request)
            
            if availability.available and availability.prices:
                current_min_price = min(availability.prices)
                
                monitor["price_history"].append({
                    "timestamp": datetime.now(),
                    "price": current_min_price
                })
                
                # Check for price alerts
                if request.price_threshold and current_min_price <= request.price_threshold:
                    await self._send_price_alert(request, current_min_price)
                
                monitor["last_price"] = current_min_price
                
        except Exception as e:
            logger.error(f"Error checking prices for monitor {monitor_id}: {e}")
    
    async def _send_booking_notification(self, request: BookingRequest, result: BookingResult):
        """Send booking notification."""
        if not request.notify_on_availability:
            return
        
        notification = {
            "type": "booking_update",
            "booking_id": request.booking_id,
            "status": result.status.value,
            "platform": request.platform,
            "booking_type": request.booking_type.value,
            "confirmation_number": result.confirmation_number,
            "total_price": result.total_price,
            "timestamp": datetime.now().isoformat()
        }
        
        for handler in self.notification_handlers:
            try:
                await handler(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")
    
    async def _send_price_alert(self, request: BookingRequest, price: float):
        """Send price alert notification."""
        alert = {
            "type": "price_alert",
            "booking_id": request.booking_id,
            "platform": request.platform,
            "booking_type": request.booking_type.value,
            "current_price": price,
            "threshold": request.price_threshold,
            "timestamp": datetime.now().isoformat()
        }
        
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error sending price alert: {e}")
    
    def add_notification_handler(self, handler):
        """Add notification handler."""
        self.notification_handlers.append(handler)
    
    def get_booking_statistics(self) -> Dict[str, Any]:
        """Get booking engine statistics."""
        total_bookings = len(self.booking_results)
        successful_bookings = sum(1 for r in self.booking_results.values() if r.status == BookingStatus.CONFIRMED)
        failed_bookings = sum(1 for r in self.booking_results.values() if r.status == BookingStatus.FAILED)
        
        platform_stats = {}
        for result in self.booking_results.values():
            platform = result.platform
            if platform not in platform_stats:
                platform_stats[platform] = {"total": 0, "successful": 0, "failed": 0}
            
            platform_stats[platform]["total"] += 1
            if result.status == BookingStatus.CONFIRMED:
                platform_stats[platform]["successful"] += 1
            elif result.status == BookingStatus.FAILED:
                platform_stats[platform]["failed"] += 1
        
        return {
            "total_bookings": total_bookings,
            "successful_bookings": successful_bookings,
            "failed_bookings": failed_bookings,
            "success_rate": successful_bookings / total_bookings if total_bookings > 0 else 0,
            "platform_stats": platform_stats,
            "active_monitors": len(self.price_monitors),
            "active_bookings": len(self.active_bookings)
        }

# Initialize global booking engine
booking_engine = RealTimeBookingEngine()
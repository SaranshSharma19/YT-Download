import requests
import json
import time
import schedule
import re
from bs4 import BeautifulSoup
import telegram
from urllib.parse import quote_plus
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deal_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID"  # Group or channel ID
SHAREASALE_MERCHANT_ID = "YOUR_SHAREASALE_MERCHANT_ID"
SHAREASALE_AFFILIATE_ID = "YOUR_SHAREASALE_AFFILIATE_ID"

# User agents to mimic browser
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

# Initialize Telegram bot
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# List of products to search for
products_to_search = [
    "product1",
    "product2",
    "product3"
]

# Function to create affiliate links
def create_affiliate_link(original_url, platform):
    if platform == "amazon":
        # For Amazon, you might use Amazon Associates
        return f"https://www.amazon.in/dp/{extract_asin(original_url)}?tag=YOUR_AMAZON_ASSOCIATE_ID"
    else:
        # For other platforms, use ShareASale
        encoded_url = quote_plus(original_url)
        return f"https://www.shareasale.com/m-pr.cfm?merchantID={SHAREASALE_MERCHANT_ID}&userID={SHAREASALE_AFFILIATE_ID}&productID=&url={encoded_url}"

def extract_asin(amazon_url):
    """Extract ASIN from Amazon URL"""
    asin_match = re.search(r'/dp/([A-Z0-9]{10})', amazon_url)
    if asin_match:
        return asin_match.group(1)
    return ""

# Function to scrape Amazon
def scrape_amazon(product_name):
    deals = []
    search_url = f"https://www.amazon.in/s?k={quote_plus(product_name)}"
    
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find all product cards
            products = soup.find_all("div", {"data-component-type": "s-search-result"})
            
            for product in products:
                try:
                    # Extract product title
                    title_element = product.find("h2")
                    if not title_element:
                        continue
                    title = title_element.text.strip()
                    
                    # Extract URL
                    url_element = title_element.find("a")
                    if not url_element:
                        continue
                    url = "https://www.amazon.in" + url_element["href"]
                    
                    # Extract prices
                    price_element = product.find("span", {"class": "a-price"})
                    if not price_element:
                        continue
                    
                    current_price_text = price_element.find("span", {"class": "a-offscreen"})
                    if not current_price_text:
                        continue
                    
                    # Extract the numerical part of the price
                    current_price = float(re.sub(r'[^\d.]', '', current_price_text.text))
                    
                    # Look for original price
                    original_price_element = product.find("span", {"class": "a-price a-text-price"})
                    if original_price_element:
                        original_price_text = original_price_element.find("span", {"class": "a-offscreen"})
                        if original_price_text:
                            original_price = float(re.sub(r'[^\d.]', '', original_price_text.text))
                            
                            # Calculate discount percentage
                            discount_percent = ((original_price - current_price) / original_price) * 100
                            
                            # Check if discount is at least 75%
                            if discount_percent >= 75:
                                deals.append({
                                    "platform": "amazon",
                                    "title": title,
                                    "url": url,
                                    "current_price": current_price,
                                    "original_price": original_price,
                                    "discount_percent": discount_percent
                                })
                except Exception as e:
                    logger.error(f"Error processing Amazon product: {e}")
        
    except Exception as e:
        logger.error(f"Error scraping Amazon: {e}")
    
    return deals

# Function to scrape Flipkart
def scrape_flipkart(product_name):
    deals = []
    search_url = f"https://www.flipkart.com/search?q={quote_plus(product_name)}"
    
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Find all product cards
            products = soup.find_all("div", {"class": "_1AtVbE"})
            
            for product in products:
                try:
                    # Extract product title
                    title_element = product.find("div", {"class": "_4rR01T"})
                    if not title_element:
                        continue
                    title = title_element.text.strip()
                    
                    # Extract URL
                    url_element = product.find("a", {"class": "_1fQZEK"})
                    if not url_element:
                        continue
                    url = "https://www.flipkart.com" + url_element["href"]
                    
                    # Extract current price
                    current_price_element = product.find("div", {"class": "_30jeq3"})
                    if not current_price_element:
                        continue
                    current_price = float(re.sub(r'[^\d.]', '', current_price_element.text))
                    
                    # Extract original price
                    original_price_element = product.find("div", {"class": "_3I9_wc"})
                    if original_price_element:
                        original_price = float(re.sub(r'[^\d.]', '', original_price_element.text))
                        
                        # Calculate discount percentage
                        discount_percent = ((original_price - current_price) / original_price) * 100
                        
                        # Check if discount is at least 75%
                        if discount_percent >= 75:
                            deals.append({
                                "platform": "flipkart",
                                "title": title,
                                "url": url,
                                "current_price": current_price,
                                "original_price": original_price,
                                "discount_percent": discount_percent
                            })
                except Exception as e:
                    logger.error(f"Error processing Flipkart product: {e}")
        
    except Exception as e:
        logger.error(f"Error scraping Flipkart: {e}")
    
    return deals

# Functions for other platforms (similar structure)
def scrape_myntra(product_name):
    # Implementation for Myntra (similar to above)
    deals = []
    search_url = f"https://www.myntra.com/{quote_plus(product_name)}"
    
    headers = {"User-Agent": USER_AGENT}
    
    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            # Implementation similar to the above examples
            # but adapted for Myntra's HTML structure
    except Exception as e:
        logger.error(f"Error scraping Myntra: {e}")
    
    return deals

def scrape_nykaa(product_name):
    # Implementation for Nykaa
    deals = []
    # Similar structure to above scrapers
    return deals

def scrape_ajio(product_name):
    # Implementation for Ajio
    deals = []
    # Similar structure to above scrapers
    return deals

# Function to post deals to Telegram
def post_deal_to_telegram(deal):
    try:
        affiliate_link = create_affiliate_link(deal["url"], deal["platform"])
        
        message = f"🔥 HUGE DISCOUNT ALERT! 🔥\n\n"
        message += f"Product: {deal['title']}\n"
        message += f"Platform: {deal['platform'].title()}\n"
        message += f"Current Price: ₹{deal['current_price']:.2f}\n"
        message += f"Original Price: ₹{deal['original_price']:.2f}\n"
        message += f"Discount: {deal['discount_percent']:.1f}% OFF\n\n"
        message += f"🛍️ Shop Now: {affiliate_link}"
        
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logger.info(f"Posted deal for {deal['title']} to Telegram")
        
    except Exception as e:
        logger.error(f"Error posting to Telegram: {e}")

# Function to check for deals across all platforms
def check_for_deals():
    logger.info("Checking for deals...")
    
    for product in products_to_search:
        # Check each platform
        amazon_deals = scrape_amazon(product)
        flipkart_deals = scrape_flipkart(product)
        myntra_deals = scrape_myntra(product)
        nykaa_deals = scrape_nykaa(product)
        ajio_deals = scrape_ajio(product)
        
        # Combine all deals
        all_deals = amazon_deals + flipkart_deals + myntra_deals + nykaa_deals + ajio_deals
        
        # Post each deal to Telegram
        for deal in all_deals:
            post_deal_to_telegram(deal)
    
    logger.info("Deal checking completed")

# Function to run the bot
def run_bot():
    logger.info("Starting deal bot...")
    
    # Schedule the job to run every 3 hours
    schedule.every(3).hours.do(check_for_deals)
    
    # Run once immediately
    check_for_deals()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    run_bot()
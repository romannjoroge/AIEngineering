import requests
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import random

class ScrapingException(BaseException):
    """Errors gotten when trying to scrape from wikipedia"""

# Use different user agents to mimic different browsers
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

def get_session():
    """
    Function to return a session that I can use to make requests to wikipedia's pages
    """
    session = requests.Session()
    
    # Add retry logic
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter=adapter)
    return session
    
SESSION = get_session()

def parse_wikipedia_site(url):
    """
    A function to parse wikipedia site and return the data I need
    """
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    # Make a request to Wikipedia
    try:
        print("Getting response")
        wikipedia_response = SESSION.get(url, headers=headers, timeout=15)
        if wikipedia_response.status_code != 200:
            raise ScrapingException(f"Wikipedia refused to send page with status code {wikipedia_response.status_code}")
    except Exception as e:
        print(f"Could not get page because of {e}")
    
    # Process result
    
parse_wikipedia_site("https://en.wikipedia.org/wiki/Jean-Paul_Abalo")
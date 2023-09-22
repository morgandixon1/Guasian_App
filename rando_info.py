from transformers import T5ForConditionalGeneration, T5Tokenizer
import csv
from datetime import datetime, timedelta
import requests
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from transformers import pipeline
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from time import sleep
import datetime
import random
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# Initialize T5 model and tokenizer
model_t5 = T5ForConditionalGeneration.from_pretrained('t5-Base')
tokenizer_t5 = T5Tokenizer.from_pretrained('t5-Base')

# Initialize GPT-2 model and tokenizer
model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

def fetch_daily_algorithm():
    with open('Algorithms.csv', 'r') as file:
        reader = csv.reader(file)
        algorithms = list(reader)
        random_row = random.choice(algorithms)
        algorithm_name = random_row[0]
        algorithm_description = random_row[1]

        print(f"Algorithm Name: {algorithm_name}")
        print(f"Algorithm Description: {algorithm_description}")
        print(f"------------------------------------------------")

def fetch_random_subreddit(subreddits):
    subreddit = random.choice(subreddits)
    url = f"https://www.reddit.com/r/{subreddit}/top.json?limit=1&t=day"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        data = response.json()
        post_texts = []

        for post in data['data']['children']:
            if not post['data'].get('pinned', False):
                title = post['data']['title']
                selftext = post['data']['selftext']
                if selftext:
                    post_texts.append({
                        'text': f"{title}\n\n{selftext}",
                        'link': f"https://www.reddit.com/r/{subreddit}/comments/{post['data']['id']}"
                    })

        summaries = []
        links = []

        # Initialize the summarization pipeline for T5
        summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

        for post_info in post_texts:
            text = post_info['text']

            # Adjust the max_length parameter to generate shorter summaries
            summary = summarizer(text, max_length=100, min_length=10, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            links.append(post_info['link'])

        return subreddit, summaries, links

    except requests.RequestException as e:
        print(f"Error fetching subreddit posts: {str(e)}")
        print("------------------------------------------------")
        return None, [], []

# Define the subreddits list
subreddits = [
    'MachineLearning',
    'artificial',
    'algotrading',
    'Futurology',
    'philosophy',
    'Music',
    'science',
    'space',
    'news',
    'worldnews',
    'politics',
    'programming',
    'dataisbeautiful',
    'personalfinance',
    'astronomy',
    'math',
    'engineering',
    'stocks',
    'marketing'
]
# Example usage
subreddit, summaries, links = fetch_random_subreddit(subreddits)
print(f"Subreddit: r/{subreddit}")
print("Summaries:", '\n'.join(summaries))
print("Links:", links)
print("------------------------------------------------")

# Delay between requests to the Reddit API
time.sleep(2)

# Define a global cache dictionary to store the fetched stock data
stock_cache = {}
def fetch_new_stock():
    # Fetch a list of available stock symbols from the Finnhub API
    url = "https://finnhub.io/api/v1/stock/symbol"
    params = {
        "exchange": "US",
        "token": "FINHUB TOKEN HERE"
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        stocks = [d["symbol"] for d in data]

        # Select a random stock symbol from the list
        selected_stock = random.choice(stocks)

        # Fetch stock data for the selected stock
        url = f"https://finnhub.io/api/v1/quote"
        params = {
            "symbol": selected_stock,
            "token": "TOKEN HERE"
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            stock_data = ""

            # Fetch additional stock information
            url_info = f"https://finnhub.io/api/v1/stock/profile2"
            params_info = {
                "symbol": selected_stock,
                "token": "TOKEN HERE"
            }
            response_info = requests.get(url_info, params=params_info)

            if response_info.status_code == 200:
                info_data = response_info.json()
                info = info_data.get("description", "No description available")
                website = info_data.get("weburl", "N/A")
                stock_data += f"Company: {selected_stock}\nDescription: {info}\nWebsite: {website}\n"

            # Fetch stock quote data
            quote_data = response.json()
            price = quote_data["c"]
            prev_close = quote_data["pc"]
            volume = quote_data.get("v", "N/A")
            market_cap = quote_data.get("marketCapitalization", "N/A")

            if prev_close == 0:
                prev_close = 1

            daily_pct_change = (price / prev_close) - 1

            # Fetch historical stock data for 3 months and 6 months ago
            today = datetime.datetime.today()
            three_months_ago = today - timedelta(days=90)
            six_months_ago = today - timedelta(days=180)

            url_history = "https://finnhub.io/api/v1/stock/candle"
            params_history = {
                "symbol": selected_stock,
                "resolution": "D",
                "from": int(three_months_ago.timestamp()),
                "to": int(today.timestamp()),
                "token": "TOKEN HERE"
            }
            response_history = requests.get(url_history, params=params_history)

            if response_history.status_code == 200:
                history_data = response_history.json()

                if "c" in history_data:
                    close_prices = history_data["c"]

                    # Check if there are enough data points
                    if len(close_prices) >= 90:
                        # Calculate the 3 month and 6 month percentage changes
                        three_month_pct_change = (price / close_prices[0]) - 1
                        six_month_pct_change = (price / close_prices[-90]) - 1
                        stock_data += f"Price: ${price:.2f}\nPrevious Close: ${prev_close:.2f}\nVolume: {volume}\nMarket Cap: {market_cap if market_cap != 'N/A' else market_cap}\n1 Week Change: {daily_pct_change:.2%}\n3 Month Change: {three_month_pct_change:.2%}\n6 Month Change: {six_month_pct_change:.2%}"
                    else:
                        stock_data += f"Price: ${price:.2f}\nPrevious Close: ${prev_close:.2f}\nVolume: {volume}\nMarket Cap: {market_cap if market_cap != 'N/A' else market_cap}\n1 Week Change: {daily_pct_change:.2%}\n3 Month Change: N/A\n6 Month Change: N/A"

                    # Print the stock data
                    print(stock_data)
                else:
                    print("Error: 'c' key not found in history_data")
            else:
                # Print an error message if the API call fails
                print("Error fetching stock data.")
    print(f"------------------------------------------------")

def search_patents_by_date(publication_date):
    # Replace this line with the appropriate path to your browser driver
    chrome_driver_path = 'path/to/chromedriver'

    # Set up options
    options = Options()
    # options.add_argument('--headless')  # Run Chrome in headless mode (without opening the browser window)
    # options.add_argument('--disable-gpu')
    options.add_argument('window-size=1920x1080')

    # Set up the service
    service = Service(executable_path=chrome_driver_path)

    # Create the driver with options and service
    driver = webdriver.Chrome(service=service, options=options)
    driver.set_window_size(1920, 1080)

    search_url = "https://ppubs.uspto.gov/pubwebapp/static/pages/ppubsbasic.html"

    # Now you can use the driver for automated browsing
    driver.get(search_url)

    # Retry loop in case no results are found or an error occurs
    while True:
        # Wait for the search text field to be present
        time.sleep(10)

        search_text = driver.find_element(By.ID, "searchText1")
        search_text.send_keys(publication_date)
        time.sleep(3)

        # Wait for the search button to be clickable
        time.sleep(10)

        search_button = driver.find_element(By.ID, "basicSearchBtn")
        search_button.click()
        time.sleep(25)  # Wait for the results to load

        # Get the innerHTML of the page
        page_innerHTML = driver.execute_script("return document.body.innerHTML;")
        # Use BeautifulSoup to parse the innerHTML
        page_soup = BeautifulSoup(page_innerHTML, "html.parser")

        # Check for the "alert alert-danger" class
        error_alerts = page_soup.select(".alert.alert-danger")
        if error_alerts:
            print("Error encountered, waiting 20 seconds and refreshing...")
            time.sleep(20)
            driver.refresh()
            time.sleep(3)
            continue

        # Wait for the search results table to be present
        time.sleep(10)

        search_results_table = driver.find_element(By.ID, "searchResults")

        # Get the innerHTML of the search results table
        innerHTML = search_results_table.get_attribute("innerHTML")

        # Use BeautifulSoup to parse the innerHTML
        soup = BeautifulSoup(innerHTML, "html.parser")
        patent_rows = soup.find_all("tr", class_=["even", "odd"])

        if patent_rows:
            print(f"Patent Filings on {publication_date}:")
            random_row = random.choice(patent_rows)
            patent_columns = random_row.find_all("td")
            publication_number = patent_columns[0].text.strip()
            title = patent_columns[1].text.strip()
            data_from_4th_column = patent_columns[3].text.strip()
            print(f"{publication_number}: {title}, Data from 4th column: {data_from_4th_column}")
            break
        else:
            print("No patents found, retrying...")
            driver.refresh()
            sleep(3)

    driver.quit()

# Get the date of a weekday within the last week
today = datetime.datetime.today()
weekday = today.weekday()
last_weekday = today - timedelta(days=7 + weekday - 1)  # Subtract days to get to the previous weekday
publication_date = last_weekday.strftime("%Y%m%d")

search_patents_by_date(publication_date)
# Call fetch_daily_algorithm()
fetch_daily_algorithm()
# Call fetch_new_stock()
fetch_new_stock()

fetch_random_subreddit(subreddits)

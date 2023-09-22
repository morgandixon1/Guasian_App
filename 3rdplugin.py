import asyncio
import spacy
import aiohttp
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import ssl
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from urllib.parse import urljoin, urlparse
from typing import List, Dict

RESOURCES = [
    ["BetaList", "https://betalist.com/"],
    ["Betabound", "https://www.betabound.com/all/"],
    ["Product Hunt", "https://www.producthunt.com/"],
    ["The Next Web", "https://thenextweb.com/"],
    ["Hacker News", "https://news.ycombinator.com/"],
    ["Reddit r/alphaandbetausers", "https://www.reddit.com/r/alphaandbetausers/"],
    ["Reddit r/betatests", "https://www.reddit.com/r/betatests/"],
    ["Reddit r/startups", "https://www.reddit.com/r/startups/"],
    ["Startup Digest", "https://www.techstars.com/communities/startup-digest"],
]

STOP_LIST = ["https://icecow.org"]

tokenizer = GPT2Tokenizer.from_pretrained('/Users/morgandixon/Desktop/Guasian/pretrained_model', padding_side='left')
model = GPT2LMHeadModel.from_pretrained('/Users/morgandixon/Desktop/Guasian/pretrained_model')
tokenizer.pad_token = tokenizer.eos_token


nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def scrape_urls(session, resource: List[str], max_urls_per_resource: int) -> List[str]:
    resource_name, resource_url = resource
    scraped_urls = []

    try:
        response = await session.get(resource_url)
        soup = BeautifulSoup(await response.text(), 'html.parser')

        for link in soup.find_all('a', href=True):
            url = link['href']

            # Process relative URLs
            url = urljoin(resource_url, url)

            # Ignore URLs outside the main website
            if urlparse(url).netloc != urlparse(resource_url).netloc:
                continue

            # Here's the new part:
            # Check that the URL is at least two slashes deep
            path = urlparse(url).path
            if path.count('/') < 2:
                continue  # Ignore this URL; it's probably a navigation link

            if url.startswith('http') and url not in scraped_urls and url not in STOP_LIST:
                scraped_urls.append(url)
                if len(scraped_urls) >= max_urls_per_resource:
                    break

    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Error scraping URLs from {resource_name}: {e}")

    return scraped_urls
async def download_html(session, url):
    try:
        response = await session.get(url, ssl_context=ssl_context)
        return await response.text()
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Error downloading {url}: {e}")
        return None

def extract_title_and_description(html):
    soup = BeautifulSoup(html, 'html.parser')
    text = " ".join(soup.stripped_strings)
    sentences = nltk.sent_tokenize(text)
    summary = ""
    summary_length = 0
    for sentence in sentences:
        summary += sentence + " "
        summary_length = len(summary.split())
        if summary_length >= 30:
            break

    title = summary.strip().split(".")[0]
    description = " ".join(summary.strip().split(".")[1:])
    return title, description

def extract_title(soup, url):
    title, _ = extract_title_and_description(str(soup))
    return title

def extract_description(soup, url):
    _, description = extract_title_and_description(str(soup))
    return description

def contains_beta_opportunity(text):
    inputs = tokenizer.encode_plus(
        "Is this text about a beta testing opportunity? " + text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], max_length=1024, attention_mask=inputs['attention_mask'], pad_token_id=tokenizer.pad_token_id)

    output_str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if 'yes' in output_str.lower():
        return 'Yes'
    elif 'no' in output_str.lower():
        return 'No'
    else:
        return 'Uncertain'

async def download_and_parse_all_articles(session, resource_data):
    scraped_data = []

    tasks = [download_html(session, url) for url in resource_data]
    html_texts = await asyncio.gather(*tasks)

    for url, html in zip(resource_data, html_texts):
        if html is None:
            continue

        soup = BeautifulSoup(html, 'html.parser')

        title = extract_title(soup, url)
        description = extract_description(soup, url)

        scraped_data.append({
            'url': url,
            'title': title,
            'description': description
        })

    return scraped_data

async def scrape_and_process(session, resources: List[List[str]], max_urls_per_resource: int) -> List[Dict[str, str]]:
    scraped_data = []

    tasks = [scrape_urls(session, resource, max_urls_per_resource) for resource in resources]
    resource_data_list = await asyncio.gather(*tasks)

    for resource_data in resource_data_list:
        article_data = await download_and_parse_all_articles(session, resource_data)
        scraped_data.extend(article_data)

    return scraped_data

async def predict_beta_opportunities(scraped_data):
    for data in scraped_data:
        title = data['title']
        beta_opportunity_answer = contains_beta_opportunity(title)
        print(f"URL: {data['url']}")
        print(f"Is this about a beta testing opportunity? {beta_opportunity_answer}")
        print("----------------------------")

async def print_all_urls(scraped_data):
    for data in scraped_data:
        print(f"URL: {data['url']}")
        print("----------------------------")

async def predict_specific_links(session, urls: List[str]):
    tasks = [download_html(session, url) for url in urls]
    html_texts = await asyncio.gather(*tasks)

    for url, html in zip(urls, html_texts):
        if html is None:
            continue

        soup = BeautifulSoup(html, 'html.parser')

        title = extract_title(soup, url)

        beta_opportunity_answer = contains_beta_opportunity(title)
        print(f"URL: {url}")
        print(f"Title: {title}")
        print(f"Is this about a beta testing opportunity? {beta_opportunity_answer}")
        print("----------------------------")

async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
        specific_urls = ["https://betalist.com/startups/thesales", "https://marc.typeform.com/to/KOdlOu", "https://www.betabound.com/web-based-music-creation-private-beta/"]
        await predict_specific_links(session, specific_urls)


# async def main():
#     async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
#         scraped_data = await scrape_and_process(session, RESOURCES, max_urls_per_resource=15)
#         await print_all_urls(scraped_data) # First, print all URLs
#         await predict_beta_opportunities(scraped_data) # Then make predictions

if __name__ == "__main__":
    asyncio.run(main())

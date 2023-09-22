import pandas as pd
from transformers import Trainer, TrainingArguments, GPT2Tokenizer, GPT2LMHeadModel
import aiohttp
import asyncio
import ssl
from typing import Dict
import torch
from torch.utils.data import Dataset
from collections import Counter

# load csv
df = pd.read_csv('/Users/morgandixon/Downloads/betas.csv')

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

class BetaOpportunityDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids']
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])

# Create SSL context
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

async def download_html(session, url):
    try:
        async with session.get(url, ssl=ssl_context) as response:
            return await response.text()
    except Exception as e:
        print(f"Error downloading URL {url}: {e}")
        return ""

def tokenize_and_format(html, label):
    inputs = tokenizer(html, truncation=True, padding='max_length', max_length=512)
    inputs['labels'] = label
    return inputs

async def main():
    successful_labels = Counter()
    all_encodings = []

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(5)  # limit the number of concurrent downloads
        async def download_and_process(url, label):
            async with sem:  # ensure only limited number of downloads are happening at once
                html = await download_html(session, url)
                if html:  # If the download was successful
                    successful_labels[label] += 1
                    encodings = tokenize_and_format(html, label)
                    all_encodings.append(encodings)
                    print(f"Scraped URL: {url}")

        # Create download tasks
        tasks = []
        for url, label in zip(df['url'], df['labels']):
            tasks.append(download_and_process(url, label))

        # Run all tasks concurrently
        await asyncio.gather(*tasks)

    min_label_count = min(successful_labels.values())
    balanced_encodings = []

    # Keep only min_label_count of each label
    label_counts = Counter()
    for encoding in all_encodings:
        label = encoding['labels']
        if label_counts[label] < min_label_count:
            balanced_encodings.append(encoding)
            label_counts[label] += 1

    # Convert list of dicts to dict of lists
    balanced_encodings_dict = {key: [dic[key] for dic in balanced_encodings] for key in balanced_encodings[0]}

    data = BetaOpportunityDataset(balanced_encodings_dict)

    print("Finished tokenizing and formatting data, now training model...")

    # Set training arguments - replace these with your own arguments
    training_args = TrainingArguments(
        output_dir="/Users/morgandixon/Desktop/results",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="/Users/morgandixon/Desktop/logs",
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
    )

    # Train the model
    trainer.train()

    print("Finished training model.")

if __name__ == "__main__":
    asyncio.run(main())

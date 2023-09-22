import openai
import csv
import random

# Set up OpenAI API key
openai.api_key = "Enter API key"

# Set up CSV file
csv_file = open('/Users/morgandixon/Desktop/Guasian/Nonhuman.csv', mode='a')  # Open in "append" mode
fieldnames = ['sentence']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

# Load topics from the CSV file
topics = []
with open('/Users/morgandixon/Desktop/Guasian/startingtopics1.csv', 'r') as topics_file:
    reader = csv.reader(topics_file)
    for row in reader:
        topics.extend(row)

# Randomize the topics
random.shuffle(topics)

# Generate sentences for each topic using OpenAI Curie
for topic in topics:
    prompt = f"Generate a sentence related to {topic} that does not refer to humans."
    response = openai.Completion.create(
        model="text-curie-001",
        prompt=prompt,
        temperature=0.87,
        max_tokens=150,
        top_p=0.21,
        frequency_penalty=0.55,
        presence_penalty=1.56
    )
    sentence = response.choices[0].text.strip()
    writer.writerow({'sentence': sentence})
    print(sentence)

# Close CSV file
csv_file.close()

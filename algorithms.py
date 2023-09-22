import openai
import csv

# Set up OpenAI API key
openai.api_key = "ENTER A Open AI API Key"

# Set up CSV file
csv_file = open('/Users/morgandixon/Desktop/Guasian/Algorithms.csv', mode='w')
fieldnames = ['name', 'description', 'formula']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

# Define list of industries
industries = [
    'Physics',
    'Chemistry',
    'Computer Science',
    'Literature',
    'Machine Learning',
    'Investing',
    'Civil Engineering',
    'Mechanical Engineering',
    'Finance',
    'Entreprenuership',
    'Marketing',
    'Trigonometry',
    'Geometry',
    'Aerospace',
    'calculus',
    'Biology',
    'cloud computing',
    'Internet Technology',
    'Architecture',
    'Natural language Processing',
    'Data Analytics',
    'Blockchain',
    'Search Engine Optimization',
    'Nuclear Engineering',
    'Data Science',
    'Electrical Engineering',
    'Robotics',
    'Virtual Reality'
    # add more industries here
]

# Generate list of formulas using OpenAI API for each industry
formulas = []
for industry in industries:
    prompt = f"Generate a list of formulas used in {industry}."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0,
    )
    formulas.extend(response.choices[0].text.strip().split('\n'))

# Generate descriptions and formulas using OpenAI API for each formula and store in CSV file
for formula in formulas:
    prompt = f"Generate a description and formula for {formula}."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=240,
        n=1,
        stop=None,
        temperature=0,
    )
    description = response.choices[0].text.strip()
    formula_text = response.choices[0].text.strip()
    print(f"{formula}: {description} {formula_text}")
    writer.writerow({'name': formula, 'description': description, 'formula': formula_text})

# Close CSV file
csv_file.close()

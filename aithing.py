from flask import Flask, request, jsonify
import re
from transformers import pipeline

app = Flask(__name__)

# Sample combined text (replace with actual content if needed)
combined_text = """
Khanate of Shar
Overview: The Khanate of Shar is a mountainous region populated mainly by mountain dwarves...
Geography: The land is rugged with high peaks and deep valleys, perfect for defense but challenging for agriculture...
Culture: Shar's people are resilient and have a deep connection to the mountains...
Political Stance: The Khanate values independence and sovereignty, resisting outside influence...
(continue for all countries)
"""

# Regular expression pattern to match country names
country_pattern = r"([A-Za-z\s]+)(?=\nOverview)"  # Matches anything before 'Overview' assuming each country starts with "Overview"

# Extract country names using the pattern
countries = re.findall(country_pattern, combined_text)

# Function to extract details for each country
def extract_country_data(text, country):
    sections = re.split(r"(Geography|Culture|Political Stance|Historical Context)", text)
    country_info = {
        "name": country.strip(),
        "overview": sections[0].strip(),
        "geography": sections[1] + sections[2] if len(sections) > 2 else "",
        "culture": sections[3] + sections[4] if len(sections) > 4 else "",
        "political": sections[5] + sections[6] if len(sections) > 6 else "",
        "historical": sections[7] + sections[8] if len(sections) > 8 else "",
    }
    return country_info

# Dictionary to hold the country data
country_data = {}

# Loop through all countries and their data
for country in countries:
    country_start = combined_text.find(country)
    country_end = combined_text.find(countries[countries.index(country) + 1]) if (countries.index(country) + 1) < len(countries) else len(combined_text)
    country_text = combined_text[country_start:country_end]
    country_data[country] = extract_country_data(country_text, country)

# Load Hugging Face model for question answering
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to answer questions based on the document
def answer_question_from_document(document, question):
    answers = qa_pipeline(question=question, context=document, top_k=5, max_answer_len=100)
    unique_answers = list(dict.fromkeys([ans['answer'] for ans in answers]))
    combined_answer = " ".join(unique_answers)
    return combined_answer

@app.route("/ask", methods=["POST"])
def ask():
    # Get the question from the request
    data = request.get_json()
    question = data.get("question")
    
    # Combine all country data into a single document
    all_country_data_text = " ".join([value for key, value in country_data.items()])
    
    # Get the answer to the question based on the combined document text
    answer = answer_question_from_document(all_country_data_text, question)
    
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)

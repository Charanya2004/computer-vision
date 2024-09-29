from flask import Flask, request, render_template
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Function to calculate sentiment score
def sentiment_score(review):
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

# Function to batch process sentiment scores
def sentiment_score_batch(reviews):
    scores = []
    for review in reviews:
        scores.append(sentiment_score(review))
    return scores

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form['url']
    class_name = request.form['class_name']
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an error for bad responses
    except requests.exceptions.RequestException as e:
        return render_template('results.html', error=str(e))  # Handle errors

    soup = BeautifulSoup(r.text, 'html.parser')
    regex=re.compile(class_name)
    # Scraping the reviews based on user-defined class name
    results = soup.find_all('p', {'class': regex})
    
    reviews = [result.get_text(strip=True) for result in results if result.get_text(strip=True)]  # Filter out empty strings
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    
    # Calculate sentiment scores
    df['sentiment'] = sentiment_score_batch(df['review'])
    
    def categorize_sentiment(score):
        if score >= 4:
            return 'Positive'
        elif score == 3:
            return 'Neutral'
        else:
            return 'Negative'

# Apply categorization to sentiment scores
    df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

    table_html = df.to_html(classes='data', header="true", index=False, justify='left').replace('\n', '')  # Remove newlines
        
    return render_template('result.html', tables=table_html)

if __name__ == '__main__':
    app.run(debug=True)

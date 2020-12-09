# Util libraries
import os
import re
import numpy as np
import pandas as pd
from collections import Counter

# Visualization
import matplotlib.pyplot as plt

# NLTK
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')


def clean_text(row):
    # A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    contractions = { 
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there'd": "there had",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where'd": "where did",
        "where's": "where is",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are"
    }

    # Steps:
    # 1. Lowercase words (for uncased GloVe Embeddings)
    # 2. Replace contractions with longer form (from the contractions dictionary)
    # 3. Replace and remove non-word characters (symbols)
    # 4. Remove stop words
    
    # Lowercase words
    text = row.lower()
    
    # Replace contractions with longer form
    text = text.split()
    expanded_text = []
    for word in text:
        if word in contractions:
            expanded_text.append(contractions[word])
        else:
            expanded_text.append(word)
    text = " ".join(expanded_text)
    
    # Format words and remove non-word characters
    text = re.sub(r'b\"', '', text)
    text = re.sub(r'b\'', '', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'0,0', '00', text) 
    text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]<>]', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'\$', ' $ ', text)
    text = re.sub(r'u s ', ' united states ', text)
    text = re.sub(r'u n ', ' united nations ', text)
    text = re.sub(r'u k ', ' united kingdom ', text)
    text = re.sub(r'j k ', ' jk ', text)
    text = re.sub(r' s ', ' ', text)
    text = re.sub(r' yr ', ' year ', text)
    text = re.sub(r' l g b t ', ' lgbt ', text)
    text = re.sub(r'0km ', '0 km ', text)
    
    # Remove stop words
    text = text.split()
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if not word in stop_words]
    
    # Return cleaned text
    cleaned_text = " ".join(text)
    
    return cleaned_text


# Calculate word frequency and the size of the vocabulary
def calculate_word_frequency(row, word_frequency):
    
    title_text = row["title"].split()
    for word in title_text:
        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1
    
    description_text = row["description"].split()
    for word in description_text:
        if word not in word_frequency:
            word_frequency[word] = 1
        else:
            word_frequency[word] += 1
    
    return word_frequency


def calculate_sentiment_polarity(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(text)  

    return sentiment_score["compound"]


def sentiment_analysis(news_folder, ticker):
    # Data files
    news_file = f"{news_folder}/{ticker}/news_{ticker}.json"

    # Read the news data
    news_df = pd.read_json(news_file)

    # Clean the news title text
    news_df["title"] = news_df.apply(lambda row: clean_text(row["title"]), axis = 1)

    # Clean the news description text
    news_df["description"] = news_df.apply(lambda row: clean_text(row["description"]), axis = 1)

    # Calculate word frequency and the size of the vocabulary
    word_frequency = {}

    news_df.apply(lambda row: calculate_word_frequency(row, word_frequency), axis = 1)

    #print("Size of vocabulary: {}".format(len(word_frequency)))

    # Sort by word frequency (descending order starting from highest count word)
    #word_frequency = {k: v for k, v in sorted(word_frequency.items(), key=lambda item: item[1], reverse=True)}

    # # Plot word frequency distribution
    # plt.figure(figsize=(12,5))
    # plt.xticks(fontsize=13, rotation=90)
    # fd = nltk.FreqDist(word_frequency)
    # fd.plot(25,cumulative=False)

    # # Log-log of all words 
    # word_counts = sorted(Counter(word_frequency).values(), reverse=True)

    # Calculate the sentiment polarity of the news title text
    news_df["title_sentiment"] = news_df.apply(lambda row: calculate_sentiment_polarity(row["title"]), axis = 1)

    # Calculate the sentiment polarity of the news description text
    news_df["description_sentiment"] = news_df.apply(lambda row: calculate_sentiment_polarity(row["description"]), axis = 1)

    news_df[f'{ticker}_Sentiment'] = news_df["title_sentiment"] + news_df["description_sentiment"]

    print(news_df)
    sentiment_df = pd.DataFrame(news_df.groupby('publishedAt')[f'{ticker}_Sentiment'].agg('sum')).reset_index()

    return sentiment_df


if __name__ == "__main__":
    sentiment_analysis("TSLA")
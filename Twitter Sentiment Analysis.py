# Imports
import pandas as pd
import nltk
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image
import pickle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('vader_lexicon')
st.set_page_config(layout="wide", page_icon=Image.open(r"D:\Twitter Sentiment Analysis\Twitter-Symbol.png"),
                   page_title="Twitter Sentiment Analysis")
selected = option_menu(None, ["Home", "Predict"], icons=["house", "pencil-square "],default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "green"}})

# Selecting Home Menu

if selected == 'Home':
    # Open the image
    image = Image.open(r"D:\Twitter Sentiment Analysis\Twitter-Symbol.png")

    # Resize the image to the desired dimensions
    new_width = 300  # Set your desired width
    new_height = 200  # Set your desired height
    resized_image = image.resize((new_width, new_height))
    # Display the resized image
    st.image(resized_image, caption="Twitter Sentiment Analysis")
    st.write('''# Project Title: Twitter Sentiment Analysis: Building a Polarity Classification Model
## Problem Statement:

Social media platforms, particularly Twitter, serve as vast repositories of user-generated content, providing a valuable resource for sentiment analysis. In this project, we aim to design a robust classification model to analyze the sentiments of tweets using a labeled dataset extracted from Twitter. The dataset comprises 1,600,000 tweets, each annotated with its sentiment polarity (0 = negative, 2 = neutral, 4 = positive).

The dataset includes six fields: 
- **target (polarity label)**
- **ids (tweet ID)**
- **date (tweet timestamp)**
- **flag (query information)**
- **user (user who tweeted)**
- **text (the content of the tweet)**

Our goal is to develop a classification model that accurately predicts the sentiment polarity of each tweet.

### Key Objectives:

### Data Exploration and Preprocessing:

- Explore the dataset to understand its structure, features, and distribution of sentiment labels.
- Handle missing or irrelevant data, and preprocess text data (e.g., tokenization, stemming, or lemmatization).

### Feature Engineering:

- Extract relevant features from the dataset that contribute to sentiment analysis.
- Consider techniques such as TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings to represent textual data.

### Model Selection:

- Evaluate various classification algorithms suitable for sentiment analysis, such as Naive Bayes, Support Vector Machines, or deep learning models like LSTM or BERT.
- Split the dataset into training and testing sets to assess model performance.

### Model Training and Tuning:

- Train the chosen model on the training set and fine-tune hyperparameters for optimal performance.
- Utilize cross-validation techniques to ensure the model generalizes well to unseen data.

### Evaluation Metrics:

- Choose appropriate evaluation metrics (e.g., accuracy, precision, recall, F1 score) to assess the model's performance.
- Analyze the confusion matrix to understand the model's strengths and weaknesses.

### Visualization:

- Visualize the results and provide insights into the distribution of sentiment polarities in the dataset.
- Generate visualizations (e.g., word clouds, sentiment distribution plots) to enhance the interpretability of the model.

### Deployment and Future Recommendations:

- Deploy the trained model for real-time sentiment analysis on new tweets.
- Provide recommendations for potential improvements or additional features for future iterations of the model.''')
# Selecting Prediction for the given text
if selected == 'Predict':
    with st.spinner('Please Wait for it...'):
        with open(r"D:\Twitter Sentiment Analysis\Cleaned Twitter Files.pkl", 'rb') as file:
            loaded_data = pickle.load(file)
        df = pd.DataFrame(loaded_data)
        x = df['text']
        y = df['target']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=5)
        # Convert text data to TF-IDF features
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        train_features = tfidf_vectorizer.fit_transform(x_train)
        test_features = tfidf_vectorizer.transform(x_test)
        load_mod = pickle.load(open(r"D:\Twitter Sentiment Analysis\Trained_Twitter1.sav", 'rb'))
        predictions = load_mod.predict(test_features)
    st.success('Done!')
    st.success('Thankyou for Waiting')
    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    if st.button('Accuracy'):
        st.write('Accuracy : ', accuracy)
        st.write('Confusion_matrix :', metrics.confusion_matrix(y_test, predictions))
        st.write('Recall_score :', metrics.recall_score(y_test, predictions, average='macro'))
        st.write('************')
    # Use the trained model to predict sentiment for new tweets
    new_tweets = st.text_input('Enter the Twitter text','I love this product!')
    new_features = tfidf_vectorizer.transform([new_tweets])
    new_predictions = load_mod.predict(new_features)
    if st.button('Predict'):
        if new_predictions == 0:
            st.write('Negative')
        else :
            st.write('Positive')




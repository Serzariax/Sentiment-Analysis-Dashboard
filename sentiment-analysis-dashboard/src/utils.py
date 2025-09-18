def preprocess_text(text):
    # Function to clean and preprocess the input text
    import re
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()  # Return cleaned text

def extract_keywords(text, n=5):
    # Function to extract keywords from the text using TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_array = vectorizer.get_feature_names_out()
    tfidf_sorting = tfidf_matrix.toarray().argsort()[0][::-1]
    top_n_keywords = feature_array[tfidf_sorting][:n]
    return top_n_keywords.tolist()  # Return list of top n keywords
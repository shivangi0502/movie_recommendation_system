import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(df):
    """
    Preprocess the movie dataset by combining important features into a single 'content' column.

    Args:
    - df (DataFrame): Movie dataset.

    Returns:
    - DataFrame: Preprocessed DataFrame with a 'content' column.
    """
    # Fill NaN values
    features = ['genres', 'keywords', 'cast', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')
    
    # Combine features into a single 'content' column
    df['content'] = df[features].apply(lambda x: ' '.join(x), axis=1)
    return df

def vectorize_content(df):
    """
    Vectorize the 'content' column using TF-IDF.

    Args:
    - df (DataFrame): Preprocessed DataFrame with a 'content' column.

    Returns:
    - tfidf_matrix: TF-IDF vectorized content.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['content'])
    return tfidf_matrix

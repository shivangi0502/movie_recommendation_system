import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process

def get_recommendations(title, df, tfidf_matrix, top_n=10):
    """
    Recommend movies based on the cosine similarity of their content vectors.

    Args:
    - title (str): The favorite movie title provided by the user.
    - df (DataFrame): The movie dataset.
    - tfidf_matrix (sparse matrix): TF-IDF vectorized content.
    - top_n (int): Number of recommendations to return.

    Returns:
    - DataFrame or str: A DataFrame of recommended movies or a message if the title is not found.
    """
    # Normalize movie titles to lowercase
    df['title_lower'] = df['title'].str.lower()
    title = title.lower()

    # Use fuzzy matching to find the closest title
    closest_match = process.extractOne(title, df['title_lower'].tolist())

    if closest_match is None or closest_match[1] < 80:
        return f"Movie '{title}' not found in the dataset."

    idx = df[df['title_lower'] == closest_match[0]].index[0]

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the top N similar movies
    similar_indices = cosine_sim.argsort()[-(top_n + 1):-1][::-1]  # Exclude itself
    return df.iloc[similar_indices][['title', 'vote_average', 'genres']]

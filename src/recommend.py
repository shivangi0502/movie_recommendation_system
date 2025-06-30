from preprocess import preprocess_data, vectorize_content
from model import get_recommendations
import pandas as pd

def main():
    dataset_path = r"C:\Users\asus\OneDrive\Desktop\movie-recommendation\data\movies.csv"
    
    print("Loading data...")
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: File not found at {dataset_path}")
        return

    print("Preprocessing data...")
    df = preprocess_data(df)

    print("Vectorizing content...")
    tfidf_matrix = vectorize_content(df)

    print("\nWelcome to the Movie Recommendation System!")
    favorite_movie = input("Enter your favorite movie: ")

    print("\nRecommendations:")
    recommendations = get_recommendations(favorite_movie, df, tfidf_matrix)
    if isinstance(recommendations, str):
        print(recommendations)
    else:
        print(recommendations)

if __name__ == "__main__":
    main()

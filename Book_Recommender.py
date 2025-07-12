import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
from difflib import get_close_matches
import warnings
import joblib
import os

warnings.filterwarnings("ignore")

# --------------------- Load Dataset ---------------------
df = pd.read_csv("cleaned_books.csv")
df = df.dropna(subset=["Title", "Author", "Genre", "Description"])
df["Title"] = df["Title"].astype(str)
df["Author"] = df["Author"].astype(str)
df["Genre"] = df["Genre"].astype(str)
df["Description"] = df["Description"].astype(str)
df["Ratings"] = pd.to_numeric(df["Ratings"], errors='coerce').fillna(0)
df["Number of Ratings"] = pd.to_numeric(df["Number of Ratings"], errors='coerce').fillna(0)

# --------------------- Weighted Score ---------------------
C = df["Ratings"].mean()
m = df["Number of Ratings"].quantile(0.50)

def weighted_score(x, m=m, C=C):
    v = x["Number of Ratings"]
    R = x["Ratings"]
    return (v / (v + m)) * R + (m / (v + m)) * C

df["Weighted Score"] = df.apply(weighted_score, axis=1)

# --------------------- Functions to Train and Save Models ---------------------
def train_and_save_models(df):
    # Genre prediction
    df["Genre List"] = df["Genre"].str.split(",\s*")
    mlb = MultiLabelBinarizer()
    Y_genre = mlb.fit_transform(df["Genre List"])

    valid_desc_mask = df["Description"].str.strip().astype(bool)
    X_desc = df.loc[valid_desc_mask, "Description"]
    Y_genre_valid = Y_genre[valid_desc_mask]

    xg_train, xg_test, yg_train, yg_test = train_test_split(X_desc, Y_genre_valid, test_size=0.2, random_state=42)
    tfidf_genre = TfidfVectorizer(stop_words='english', max_features=5000)
    xg_train_tfidf = tfidf_genre.fit_transform(xg_train)

    genre_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, n_jobs=-1))
    genre_model.fit(xg_train_tfidf, yg_train)

    joblib.dump(genre_model, "genre_model.pkl")
    joblib.dump(tfidf_genre, "tfidf_genre.pkl")
    joblib.dump(mlb, "mlb.pkl")

    # Rating prediction
    xr = df[["Description", "Genre"]].copy()
    xr["input"] = xr["Description"] + " " + xr["Genre"]
    yr = df["Ratings"]

    xr_train, xr_test, yr_train, yr_test = train_test_split(xr["input"], yr, test_size=0.2, random_state=42)
    tfidf_rating = TfidfVectorizer(stop_words='english', max_features=5000)
    xr_train_tfidf = tfidf_rating.fit_transform(xr_train)

    rating_model = LinearRegression()
    rating_model.fit(xr_train_tfidf, yr_train)

    joblib.dump(rating_model, "rating_model.pkl")
    joblib.dump(tfidf_rating, "tfidf_rating.pkl")

def precompute_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df["Description"])
    joblib.dump(tfidf_matrix, "tfidf_matrix.pkl")
    joblib.dump(tfidf, "tfidf.pkl")

# --------------------- Load or Train Models ---------------------
if (os.path.exists("genre_model.pkl") and
    os.path.exists("tfidf_genre.pkl") and
    os.path.exists("mlb.pkl") and
    os.path.exists("rating_model.pkl") and
    os.path.exists("tfidf_rating.pkl") and
    os.path.exists("tfidf_matrix.pkl") and
    os.path.exists("tfidf.pkl")):

    genre_model = joblib.load("genre_model.pkl")
    tfidf_genre = joblib.load("tfidf_genre.pkl")
    mlb = joblib.load("mlb.pkl")

    rating_model = joblib.load("rating_model.pkl")
    tfidf_rating = joblib.load("tfidf_rating.pkl")

    tfidf_matrix = joblib.load("tfidf_matrix.pkl")
    tfidf = joblib.load("tfidf.pkl")

else:
    print("Training models... This may take a while.")
    train_and_save_models(df)
    precompute_tfidf_matrix(df)
    print("Training done. Please restart the program.")
    exit()

# --------------------- User Interface ---------------------
print("\n📚 Book Recommendation System")
print("1. Recommend by Genre")
print("2. Recommend Similar Books by Title")
print("3. Predict Genre from Description")
print("4. Predict Rating for a New Book")
choice = input("\n👉 Enter choice (1–4): ").strip()

# --------------------- Option 1: Recommend by Genre ---------------------
if choice == "1":
    genre_list = ["Romance", "Horror", "Thriller", "Fantasy", "Children", "Law", "Science Fiction", "Political", 
                  "History", "Language", "Literature", "Computing", "Fiction", "Biography", "Health", 
                  "Psychology", "Religion", "Crime", "Comedy", "Mystery"]
    print("\n🎯 Available genres:")
    for g in genre_list:
        print(" -", g)

    selected_input = input("\n📖 Choose your genre(s): ").lower().split()

    filtered = df[df["Genre"].str.lower().apply(lambda g: any(word in g for word in selected_input))]

    if filtered.empty:
        print("❌ No books found for that genre.")
    else:
        filtered = filtered.sort_values(by="Weighted Score", ascending=False).head(10)
        print(f"\n📌 Top {len(filtered)} books:")
        for i, (_, row) in enumerate(filtered.iterrows(), 1):
            print(f"{i}. {row.Title} by {row.Author}")
            print(f"   📖 Genre: {row.Genre}\n   ⭐ Rating: {row.Ratings} ({int(row['Number of Ratings'])} reviews)")
            if 'Buy Link' in row:
                print(f"   🛒 Buy Link: {row['Buy Link']}")
            print(f"   📝 {row.Description}\n")

# --------------------- Option 2: Recommend Similar Books ---------------------
elif choice == "2":
    book_name = input("🔍 Enter a book title: ").lower()
    titles_lower = df["Title"].str.lower()
    matches = get_close_matches(book_name, titles_lower, n=3, cutoff=0.6)

    if not matches:
        print("❌ Book not found.")
        exit()

    if len(matches) > 1:
        print("🤔 Did you mean:")
        for i, title in enumerate(matches, 1):
            print(f"{i}. {title.title()}")
        pick = input("👉 Choose the correct title (1–3): ").strip()
        try:
            book_name = matches[int(pick) - 1]
        except:
            print("❌ Invalid selection.")
            exit()
    else:
        book_name = matches[0]

    book_index = df[titles_lower == book_name].index[0]

    # Use precomputed matrix for similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[book_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    similar_books = [i[0] for i in sim_scores]

    print(f"\n📌 Books similar to '{df.iloc[book_index].Title}':")
    for i, book_idx in enumerate(similar_books, 1):
        book = df.iloc[book_idx]
        print(f"{i}. {book.Title} by {book.Author}\n   📖 Genre: {book.Genre}\n   ⭐ Rating: {book.Ratings} ({int(book['Number of Ratings'])} reviews)")
        if 'Buy Link' in book:
            print(f"   🛒 Buy Link: {book['Buy Link']}")
        print(f"   📝 {book.Description}\n")

# --------------------- Option 3: Predict Genre ---------------------
elif choice == "3":
    title = input("📘 Enter book title (optional): ").strip()
    desc = input("📝 Enter book description: ").strip()

    if not desc:
        print("❌ Description is required.")
    else:
        tfidf_input = tfidf_genre.transform([desc])
        pred_vector = genre_model.predict(tfidf_input)
        predicted_genres = mlb.inverse_transform(pred_vector)

        if predicted_genres and predicted_genres[0]:
            print(f"\n📖 Predicted Genre(s) for '{title if title else 'your book'}':")
            for genre in predicted_genres[0]:
                print(f" - {genre}")
        else:
            print("❌ Could not confidently predict any genre.")

# --------------------- Option 4: Predict Rating ---------------------
elif choice == "4":
    desc = input("📝 Enter book description: ")
    genre = input("📖 Enter genre: ")
    combined = desc + " " + genre
    tfidf_input = tfidf_rating.transform([combined])
    pred_rating = rating_model.predict(tfidf_input)[0]
    print(f"\n⭐ Predicted Rating: {pred_rating:.2f}")

else:
    print("❌ Invalid option.")

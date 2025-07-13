
import streamlit as st

import pandas as pd
import numpy as np
import requests
import joblib
import urllib.parse
from difflib import get_close_matches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit_authenticator as stauth





st.set_page_config(page_title="📚 Book Buddy", layout="wide")
st.markdown("""
                
    <link
        rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    />
                
    <style>
    @keyframes shimmer {
        0% {text-shadow: 0 0 6px #d6c6ff, 0 0 10px #e0b3ff, 0 0 14px #b384ff;}
        50% {text-shadow: 0 0 10px #e0b3ff, 0 0 16px #d6c6ff, 0 0 20px #b384ff;}
        100% {text-shadow: 0 0 6px #d6c6ff, 0 0 10px #e0b3ff, 0 0 14px #b384ff;}
    }

    .header-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px 40px;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }


    .book-buddy-title {
        font-family: 'Baskerville', 'Georgia', serif;
        font-size: 3em;
        font-weight: 600;
        color: #4b0082;
    
        display: flex;
        align-items: center;
        gap: 12px;
    }
    /* Right Side: Notification + Login */
    .header-right {
        display: flex;
        align-items: center;
        gap: 30px;
        font-size: 1.3em;
    }

    /* Login Button */
    .login-button {
        background-color: #521264;
        color: #fff !important;
        padding: 5px 16px;
        border-radius: 8px;
        text-decoration: none !important;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }

    .login-button:hover {
        background-color: #7c0a49;
        color: white;
    }

    /* Notification Hover */
    .header-right i:hover {
        color: #a4271c;
        cursor: pointer;
    }
    }

    </style>
    """, unsafe_allow_html=True)

st.markdown("""
        <div class="header-bar">
            <div class="book-buddy-title">
                <i class="fas fa-book-reader"></i> Book Buddy
            </div>
            <div class="header-right">
                <i class="fas fa-bell"></i>
                <a class="login-button" href="#">Login</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Smooth gradient background */
    body, .stApp {
        
        background:
        linear-gradient(rgba(255,255,255,0.6), rgba(255,255,255,0.7)),
        url('https://t3.ftcdn.net/jpg/07/93/51/08/360_F_793510820_ERDrJAZQrfTimnVE5MZQDPFBPnP4spuG.jpg');
        background-size: cover;         /* Cover whole area */
        background-repeat: no-repeat;   /* No repeating */
        background-position: center;    /* Center the image */
        background-attachment: fixed;   
        background-attachment: fixed;
        font-family: 'Segoe UI', sans-serif;
        color: #333333;
    }

    /* Stylish tab design */
    div[data-baseweb="tab-list"] button  {
            background-image: linear-gradient(to right, #3A1C71 0%, #D76D77 51%, #3A1C71 100%);
            font-size: 17px;
            font-weight: bold;
            padding: 10px 20px;
            margin-right: 8px;
            border: none;
            border-radius: 12px 12px 0 0;
            color: white;
            text-transform: uppercase;
            transition: all 0.5s ease;
            box-shadow: 0 0 20px #eee;
        }

        /* Selected tab */
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background-image: linear-gradient(to right, #000000 0%, #53346D  51%, #000000  100%);
            color:#3A1C71;  
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }

    button{
                background-image: linear-gradient(to right, #CB356B 0%, #BD3F32  51%, #CB356B  100%);
                margin: 10px;
                padding: 12px 30px;
                text-align: center;
                text-transform: uppercase;
                font-size: 15px;
                font-weight: 600;
                letter-spacing: 1px;
                transition: 0.5s ease-in-out;
                color: #fff !important;
                box-shadow: 0 0 20px #eee;
                border-radius: 12px;
                display: inline-block;
                border: none;
                cursor: pointer;
        }

    button:hover{
                background-position: right center;
                background:#fd8ea3;
                color: #4d1142 !important;
                transform: scale(1.02);
                border: none;
        }
                





    /* Book card with glassy look */
    .book-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        backdrop-filter: blur(4px);
        transition: transform 0.2s ease-in-out;
        border-left: 5px solid #6a11cb;
        display: flex;
        align-items: flex-start;
        gap: 1.5rem;
        flex-wrap: wrap;
    }
            
    .book-card img {
        width: 150px;
        height: auto;
        border-radius: 10px;
        object-fit: cover;
        flex-shrink: 0;
        transition: opacity 0.3s ease;
    }
                
    .book-info {
        flex: 1;
        min-width: 250px;
    }

    .book-card:hover {
        transform: translateY(-6px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
                


    /* Styled buttons */


    /* Inputs and selectors */
    .stTextInput, .stTextArea, .stMultiSelect, .stSelectbox {
        border-radius: 12px !important;
        padding: 8px;
        border: 1px solid #ccc;
        background-color: #a22486;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
        font-size: 16px;
        color: #fff;
    }

    .stTextArea textarea {
        min-height: 140px;
    }

    hr {
        border: none;
        border-top: 1px solid #ccc;
        margin: 2rem 0;
    }

    /* Headings and fonts */
    h1, h2, h3, h4 {
        color: #1e3a8a;
        font-weight: 700;
    }

    .stMarkdown a {
        color: #2b72ff;
        font-weight: 500;
    }

    .stMarkdown a:hover {
        text-decoration: underline;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.9rem;
        color: #777;
        margin-top: 4rem;
    }
    </style>
                


    """, unsafe_allow_html=True)


@st.cache_data
def load_data():
        df = pd.read_csv("cleaned_books.csv")
        df.dropna(subset=["Title", "Author", "Genre", "Description"], inplace=True)
        df["Title"] = df["Title"].astype(str)
        df["Author"] = df["Author"].astype(str)
        df["Genre"] = df["Genre"].astype(str)
        df["Description"] = df["Description"].astype(str)
        df["Ratings"] = pd.to_numeric(df["Ratings"], errors='coerce').fillna(0)
        df["Number of Ratings"] = pd.to_numeric(df["Number of Ratings"], errors='coerce').fillna(0)
        return df

@st.cache_data
def compute_weighted_scores(df):
        C = df["Ratings"].mean()
        m = df["Number of Ratings"].quantile(0.50)
        def weighted_score(x): 
            v, R = x["Number of Ratings"], x["Ratings"]
            return (v / (v + m)) * R + (m / (v + m)) * C
        df["Weighted Score"] = df.apply(weighted_score, axis=1)
        return df

@st.cache_resource
def load_models():
        genre_model = joblib.load("genre_model.pkl")
        tfidf_genre = joblib.load("tfidf_genre.pkl")
        rating_model = joblib.load("rating_model.pkl")
        tfidf_rating = joblib.load("tfidf_rating.pkl")
        mlb = joblib.load("mlb.pkl")
        return genre_model, tfidf_genre, rating_model, tfidf_rating,mlb

df = compute_weighted_scores(load_data())
genre_model, tfidf_genre, rating_model, tfidf_rating, mlb = load_models()

tab1, tab2, tab3, tab4, tab5, tab6, tab7= st.tabs([
        "🎯 Recommend by Genre",
        "🔍 Similar Books",
        "🧠 Predict Genre",
        "⭐ Predict Rating",
        "📈 Trending Books",
        "💾 My Favorites",
        "🤖 Chatbot"
    
    ])

if "favorites" not in st.session_state:
        st.session_state.favorites = []


with tab1:
        # Inject styles
        st.markdown("""
            <style>
            .title-heading {
                font-size: 2em;
                font-weight: bold;
                color: #4B0082;
                margin-top: 20px;
                margin-bottom: 10px;
                text-align: center;
            }
            .genre-subhead {
                font-size: 1.5em;
                color: #6c0e65;
                margin-bottom: 8px;
                font-weight: bold;
            }
            .book-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 12px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }
            .book-info {
            }
            .action-buttons {
                margin-top: 10px;
                display: flex;
                gap: 4px;
                align-items: center;
            }

                        
                                
                    
        </style>
        """, unsafe_allow_html=True)

        st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">

        <style>
        .romantic-heading {
            font-family: 'Satisfy', cursive;
            font-size: 2em;
            color: #ad1457; /* Darker pink-purple */
            text-shadow: 2px 2px 6px rgba(100, 0, 50, 0.4);
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='romantic-heading'>🎯 Discover Your Next Favorite Book</div>", unsafe_allow_html=True)


        genre_list = [
            "Romance", "Horror", "Thriller", "Fantasy", "Children", "Law", "Science Fiction",
            "Political", "History", "Language", "Literature", "Computing", "Fiction", "Biography",
            "Health", "Psychology", "Religion", "Crime", "Comedy", "Mystery"
        ]

        st.markdown("<div class='genre-subhead'>📚 Select your favorite genres:</div>", unsafe_allow_html=True)
        selected = st.multiselect("Available Genres", genre_list)

        if selected:
            filtered = df[df["Genre"].str.lower().apply(lambda g: any(s.lower() in g for s in selected))]

            min_rating = st.slider("⭐ Minimum rating", 0.0, 5.0, 3.5, 0.1)
            filtered = filtered[filtered["Ratings"] >= min_rating]

            sort_option = st.selectbox("📊 Sort by:", ["Weighted Score", "Ratings", "Number of Ratings", "Title"])
            sort_col_map = {
                "Weighted Score": "Weighted Score",
                "Ratings": "Ratings",
                "Number of Ratings": "Number of Ratings",
                "Title": "Title"
            }
            filtered = filtered.sort_values(sort_col_map[sort_option], ascending=(sort_option == "Title"))

            show_full_desc = st.checkbox("📝 Show full descriptions", value=True)

            if filtered.empty:
                st.warning("🚫 No books found for selected genre(s). Try others!")
            else:
                if 'favorites' not in st.session_state:
                    st.session_state.favorites = []

                top_books = filtered.head(10)

                for i, (_, book) in enumerate(top_books.iterrows()):
                        desc = book.get('Description', '')
                        form_key = f"form_{i}_{book.Title.replace(' ', '_')}"
                        short_desc = desc if show_full_desc or len(desc) <= 200 else desc[:200] + "..."

                    
                        with st.form(key=f"favorite_form_{i}", clear_on_submit=False):
                            st.markdown(f"""
                                <div class="book-card">
                                    <div class="book-info">
                                        <h3>📖 <em>{book.Title}</em></h3>
                                        <p><strong>👤 Author:</strong> {book.Author}</p>
                                        <p><strong>🎭 Genre:</strong> {', '.join(eval(book.Genre))}</p>
                                        <p><strong>⭐ Rating:</strong> {book.Ratings:.2f} ({int(book['Number of Ratings'])} reviews)</p>
                                        <p>📝 <em>{short_desc}</em></p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                            col1, col2 = st.columns([1, 1])

                            with col1:
                                
                                submitted = st.form_submit_button("💛 Add to Favorites")
                                if submitted:
                                    if book.to_dict() not in st.session_state.favorites:
                                            st.session_state.favorites.append(book.to_dict())
                                            st.success("✅ Added to favorites!")



                            with col2:
                                if 'Buy Link' in book and pd.notna(book['Buy Link']):
                                    st.markdown(
                                        f"""
                                        <style>
                                            .gradient-button {{
                                                background-image: linear-gradient(to right, #360033 0%, #0b8793 51%, #360033 100%);
                                                background-size: 200% auto;
                                                color: #fff !important;
                                                font-weight: 600;
                                                font-size: 14px;
                                                letter-spacing: 1px;
                                                padding: 10px 25px;
                                                text-align: center;
                                                text-transform: uppercase;
                                                border-radius: 10px;
                                                border: none;
                                                display: inline-block;
                                                transition: 0.4s ease-in-out;
                                                box-shadow: 0 0 15px rgba(54, 0, 51, 0.4);
                                                cursor: pointer;
                                                text-decoration: none !important;
                    
                                            }}
                                            .gradient-button:hover {{
                                                background-position: right center;
                                                color: #fff;
                                                text-decoration: none; !important;
                                                transform: scale(1.05);
                                                box-shadow: 0 0 25px rgba(11, 135, 147, 0.5);
                                                
                                            }}
                                            
                                        </style>

                                        <a href="{book["Buy Link"]}" class="gradient-button" target="_blank">
                                            <span class="icon">🛒</span> Buy
                                        </a>
                                        """,
                                        unsafe_allow_html=True
                                    )

                    
                                
                        


        else:
                st.markdown("""
                            <style>
                            .custom-info {
                                background-color: #f3e5f5; /* Light purple background */
                                border-left: 6px solid #ba68c8; /* Deeper purple border */
                                padding: 12px 16px;
                                border-radius: 8px;
                                font-size: 16px;
                                color: #4a148c; /* Deep purple text */
                                margin-bottom: 16px;
                            }
                            </style>

                            <div class="custom-info">
                                💡 Select one or more genres to get personalized recommendations.
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown("""
                            <div style="text-align: center;">
                                <img src="https://cdn-icons-png.flaticon.com/512/2702/2702154.png" width="100"/>
                            </div>
                            """, unsafe_allow_html=True)


with tab2:
        
        st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">

        <style>
        .romantic-heading {
            font-family: 'Satisfy', cursive;
            font-size: 2em;
            color: #ad1457; /* Darker pink-purple */
            text-shadow: 2px 2px 6px rgba(100, 0, 50, 0.4);
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='romantic-heading'>🔍 Find similar books by title</div>", unsafe_allow_html=True)
        

        book_input = st.text_input("📕 Type a book title to search...")

        if book_input:
            titles_lower = df["Title"].str.lower()
            matches = get_close_matches(book_input.lower(), titles_lower, n=3, cutoff=0.6)
            if not matches:
                st.error("🚫 No similar titles found. Try checking spelling.")
            else:
                best_match = matches[0]
                book_index = df[titles_lower == best_match].index[0]
                tfidf = TfidfVectorizer(stop_words='english')
                tfidf_matrix = tfidf.fit_transform(df["Description"])
                cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                sim_scores = list(enumerate(cosine_sim[book_index]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
                st.success(f"📘 Books similar to: **{df.iloc[book_index].Title}**")
                for i, (idx, _) in enumerate(sim_scores, 1):
                    b = df.iloc[idx]
                    st.markdown(f"""
                                <div class="book-card">
                                    <div class="book-info">
                                        <h3>{i}. {b.Title}</h3>
                                        <p><strong>By:</strong> <em>{b.Author}</em></p>
                                        <p><strong>🎭 Genre:</strong> {b.Genre} &nbsp; | &nbsp; ⭐ {b.Ratings:.2f} ({int(b['Number of Ratings'])} reviews)</p>
                                        <p><em>📝 {b.Description}</em></p>
                                    </div>
                                </div>
                            """, unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        if st.button("💛 Add to Favorites", key=f"fav_{b.Title}"):
                            st.session_state.favorites.append(b.to_dict())
                            st.success("Added to favorites!")
                            st.markdown("---")
                        
                    with col2:
                        if 'Buy Link' in b and pd.notna(b['Buy Link']):
                            st.markdown(
                                f"""
                                <style>
                                .gradient-button {{
                                    background-image: linear-gradient(to right, #360033 0%, #0b8793 51%, #360033 100%);
                                    background-size: 200% auto;
                                    color: #fff !important;
                                    font-weight: 600;
                                    font-size: 14px;
                                    letter-spacing: 1px;
                                    padding: 10px 25px;
                                    text-align: center;
                                    text-transform: uppercase;
                                    border-radius: 10px;
                                    border: none;
                                    display: inline-block;
                                    transition: 0.4s ease-in-out;
                                    box-shadow: 0 0 15px rgba(54, 0, 51, 0.4);
                                    cursor: pointer;
                                    text-decoration: none !important;
                        
                                }}
                                    .gradient-button:hover {{
                                        background-position: right center;
                                        color: #fff;
                                        text-decoration: none; !important;
                                        transform: scale(1.05);
                                        box-shadow: 0 0 25px rgba(11, 135, 147, 0.5);
                                                    
                                }}
                                                
                            </style>

                            <a href="{b["Buy Link"]}" class="gradient-button" target="_blank">
                                <span class="icon">🛒</span> Buy
                            </a>
                            """,
                            unsafe_allow_html=True
                        )
                        
        else:
                
                st.markdown("""
                            <div style="text-align: center;">
                                <img src="https://cdn-icons-png.flaticon.com/512/987/987769.png" width="100"/>
                            </div>
                            """, unsafe_allow_html=True)

            

with tab3:
        

        st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">

    <style>
        .romantic-heading {
            font-family: 'Satisfy', cursive;
            font-size: 2em;
            color: #ad1457; /* Darker pink-purple */
            text-shadow: 2px 2px 6px rgba(100, 0, 50, 0.4);
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('''
                <div class='romantic-heading'>
                    <img src="https://cdn-icons-png.flaticon.com/512/18231/18231337.png" width="40"/>
                    Predict genre based on description
                </div>
            ''', unsafe_allow_html=True)

        

        title_input = st.text_input("📝 Optional Title")
        desc_input = st.text_area("📄 Book Description")

        if st.button("🔮 Predict Genre"):
            if not desc_input.strip():
                st.warning("⚠️ Please enter a description.")
            else:
                vector = tfidf_genre.transform([desc_input])
                pred = genre_model.predict(vector)
                predicted = mlb.inverse_transform(pred)[0]  # <-- Fixed line

            if predicted:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #f3e5f5, #e1bee7);
                    border-left: 6px solid #6a1b9a;
                    border-radius: 12px;
                    padding: 20px;
                    margin-top: 20px;
                    box-shadow: 0 4px 10px rgba(106, 27, 154, 0.2);
                ">
                    <h4 style="color: #4a148c; font-family: 'Georgia', serif;">
                        📚 Predicted genre(s) for <strong>{title_input or 'your book'}</strong>:
                    </h4>
                    <ul style="list-style: none; padding-left: 0;">
                """ + "\n".join([
                    f"<li style='margin: 6px 0; color: #311b92;'>✅ <strong>{genre}</strong></li>" for genre in predicted
                ]) + """
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Couldn't predict any genre.")

with tab4:
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">

    <style>
    textarea, input[type="text"] {
            background-color: #ffffff !important;
            color: #333333 !important;
            border: 1px solid #ccc !important;
            border-radius: 10px !important;
            padding: 10px !important;
            font-size: 16px !important;
        }
        ::placeholder {
            color: #999999 !important;
        }
    .romantic-heading {
        font-family: 'Satisfy', cursive;
        font-size: 2em;
        color: #ad1457; /* Darker pink-purple */
        text-shadow: 2px 2px 6px rgba(100, 0, 50, 0.4);
        text-align: center;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='romantic-heading'>⭐ Predict rating from book features</div>", unsafe_allow_html=True)

    desc = st.text_area("📖 Book Description")
    genre = st.text_input("📚 Genre")

    if st.button("🔮 Predict Rating"):
        if not desc.strip() or not genre.strip():
            st.warning("⚠️ Please provide both description and genre.")
        else:
            combined = desc + " " + genre
            vector = tfidf_rating.transform([combined])
            predicted_rating = rating_model.predict(vector)[0]

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #ffe0ec, #fce4ec);
                border-left: 6px solid #d81b60;
                border-radius: 15px;
                padding: 20px 25px;
                margin-top: 25px;
                box-shadow: 0 4px 12px rgba(216, 27, 96, 0.2);
                font-family: 'Georgia', serif;
            ">
                <h4 style="color: #880e4f; font-size: 20px;">
                    🌟 Predicted Rating:
                    <span style="color: #fbc02d; font-size: 24px;">⭐ {predicted_rating:.2f}</span>
                </h4>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center;">
            <img src="https://cdn-icons-png.flaticon.com/512/7927/7927801.png" width="100"/>
        </div>
        """, unsafe_allow_html=True)

with tab5:
        st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">
        <style>
        .romantic-heading {
            font-family: 'Satisfy', cursive;
            font-size: 2em;
            color: #ad1457;
            text-shadow: 2px 2px 6px rgba(100, 0, 50, 0.4);
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='romantic-heading'>📈 Top Trending Books</div>", unsafe_allow_html=True)
        
        trending = df.sort_values("Weighted Score", ascending=False).head(10)
        
        for idx, book in trending.iterrows():
            
            st.markdown(f"""
                        <div class="book-card">
                            <div class="book-info">
                                <h3>🔥 {book.Title}</h3>
                                <p><strong>By:</strong> <em>{book.Author}</em></p>
                                <p><strong>🎭 Genre:</strong> {book.Genre}</p>
                                <p><strong>⭐ Rating:</strong> {book.Ratings:.2f} ({int(book['Number of Ratings'])} reviews)</p>
                                <p><em>📝 {book.Description}</em></p>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

            col1, col2 = st.columns([1, 1])

            with col1:
                key = f"fav_{idx}_{book.Title.replace(' ', '_')}"
                if st.button("💛 Add to Favorites", key=key):
                    if 'favorites' not in st.session_state:
                        st.session_state.favorites = []
                    st.session_state.favorites.append(book.to_dict())
                    st.success("Added to favorites!")
                        
            with col2:
                if 'Buy Link' in book and pd.notna(book['Buy Link']):
                    st.markdown(f"""
                        <style>
                        .gradient-button {{
                            background-image: linear-gradient(to right, #360033 0%, #0b8793 51%, #360033 100%);
                            background-size: 200% auto;
                            color: #fff !important;
                            font-weight: 600;
                            font-size: 14px;
                            letter-spacing: 1px;
                            padding: 10px 25px;
                            text-align: center;
                            text-transform: uppercase;
                            border-radius: 10px;
                            border: none;
                            display: inline-block;
                            transition: 0.4s ease-in-out;
                            box-shadow: 0 0 15px rgba(54, 0, 51, 0.4);
                            cursor: pointer;
                            text-decoration: none !important;
                        
                        }}
                        .gradient-button:hover {{
                            background-position: right center;
                            color: #fff;
                            text-decoration: none !important;
                            transform: scale(1.05);
                            box-shadow: 0 0 25px rgba(11, 135, 147, 0.5);
                        }}
                        </style>

                        <a href="{book['Buy Link']}" class="gradient-button" target="_blank">
                            <span class="icon">🛒</span> Buy
                        </a>
                    """, unsafe_allow_html=True)
                    
                st.markdown("---")


with tab6:
        st.markdown("""
        <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">


        <style>
        .romantic-heading {
            font-family: 'Satisfy', cursive;
            font-size: 2em;
            color: #ad1457; /* Darker pink-purple */
            text-shadow: 2px 2px 6px rgba(100, 0, 50, 0.4);
            text-align: center;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("<div class='romantic-heading'>❤️ Your Favorite Books</div>", unsafe_allow_html=True)

    


        if not st.session_state.favorites:
            st.markdown("""
                            <style>
                            .custom-info {
                                background-color: #f3e5f5; /* Light purple background */
                                border-left: 6px solid #ba68c8; /* Deeper purple border */
                                padding: 12px 16px;
                                border-radius: 8px;
                                font-size: 16px;
                                color: #4a148c; /* Deep purple text */
                                margin-bottom: 16px;
                            }
                            </style>

                            <div class="custom-info">
                                You have no favorite books yet.
                            </div>
                            """, unsafe_allow_html=True)
        else:
            for i, book in enumerate(st.session_state.favorites[:]):
                st.markdown(f"### 📖 {book['Title']} by *{book['Author']}*")
                st.markdown(f"**🎭 Genre:** {book['Genre']}  |  ⭐ {book['Ratings']:.2f} ({int(book['Number of Ratings'])} reviews)")
                st.markdown(f"📝 _{book['Description']}_")

                col1, col2 = st.columns([1, 1])

                with col1:
                    remove_key = f"remove_{book['Title']}_{book['Author']}_{i}"
                    if st.button("❌ Remove from Favorites", key=remove_key):
                        st.session_state.favorites.remove(book)
                        st.success(f"Removed **{book['Title']}** from favorites.")

                        # Safely rerun the app only if supported
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()
                        else:
                            st.warning("⚠️ Please  refresh the page to update the favorites.")

                        
                with col2:
                    if 'Buy Link' in book and pd.notna(book['Buy Link']):
                        st.markdown(f"""
                            <style>
                            .gradient-button {{
                                background-image: linear-gradient(to right, #360033 0%, #0b8793 51%, #360033 100%);
                                background-size: 200% auto;
                                color: #fff !important;
                                font-weight: 600;
                                font-size: 14px;
                                letter-spacing: 1px;
                                padding: 10px 25px;
                                text-align: center;
                                text-transform: uppercase;
                                border-radius: 10px;
                                border: none;
                                display: inline-block;
                                transition: 0.4s ease-in-out;
                                box-shadow: 0 0 15px rgba(54, 0, 51, 0.4);
                                cursor: pointer;
                                text-decoration: none !important;
                        
                            }}
                            .gradient-button:hover {{
                                background-position: right center;
                                color: #fff;
                                text-decoration: none; !important;
                                transform: scale(1.05);
                                box-shadow: 0 0 25px rgba(11, 135, 147, 0.5);
                                                    
                            }}
                                                
                            </style>

                            <a href="{book['Buy Link']}" class="gradient-button" target="_blank">
                                <span class="icon">🛒</span> Buy
                            </a>
                        """,unsafe_allow_html=True
                            )
                        
                st.markdown("---")

                
st.markdown("""
    <hr>
    <div class="footer">
        Made with ❤️ by Book Buddy · © 2025 All rights reserved.
    </div>
    """, unsafe_allow_html=True)

with tab7:
    st.markdown(
        """
        <style>
        .chat-header {
            font-size: 28px;
            font-weight: bold;
            color: #2e86de;
            margin-bottom: 20px;
        }
        .chat-bubble-user {
            background-color: #dff9fb;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border: 1px solid #c7ecee;
            max-width: 90%;
            align-self: flex-end;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        .chat-bubble-assistant {
            background-color: #f0f8ff;
            padding: 15px;
            margin: 10px 0;
            border-radius: 10px;
            border: 1px solid #a9cce3;
            max-width: 90%;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="chat-header">🤖 Book Buddy AI Chatbot Assistant</div>', unsafe_allow_html=True)

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "You are Book Buddy, a helpful AI assistant that recommends books to users. "
                    "You help them discover books by genre, similar titles, or themes. "
                    "Keep answers short and friendly. Recommend specific books when possible."
                )
            }
        ]

    # Display previous messages
    for msg in st.session_state.messages[1:]:
        role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-assistant"
        st.markdown(f'<div class="{role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    prompt = st.chat_input("What kind of book are you looking for?")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show user message immediately
        st.markdown(f'<div class="chat-bubble-user">{prompt}</div>', unsafe_allow_html=True)

        headers = {
            "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "llama3-8b-8192",
            "messages": st.session_state.messages,
            "temperature": 0.7,
            "stream": False
        }

        with st.spinner("Book Buddy is thinking..."):
            try:
                response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)
                if response.status_code == 200:
                    reply = response.json()["choices"][0]["message"]["content"]
                else:
                    reply = f"❌ Error: {response.status_code} - {response.text}"
            except Exception as e:
                reply = f"❌ API request failed: {e}"

        st.markdown(f'<div class="chat-bubble-assistant">{reply}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": reply})

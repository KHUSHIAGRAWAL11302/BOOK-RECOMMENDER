import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* App background */
    [data-testid="stAppViewContainer"] {
        background-color: #F5F4F0;
    }

    [data-testid="stSidebar"] {
        background-color: #1C1C1C;
    }

    [data-testid="stSidebar"] * {
        color: #DEDBD2 !important;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stRadio label {
        color: #AAAAAA !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    /* Header */
    .app-header {
        background-color: #1C1C1C;
        color: #F5F4F0;
        padding: 2.5rem 2rem 2rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        border-bottom: 3px solid #C8B560;
    }

    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #F5F4F0;
    }

    .app-header p {
        font-size: 0.9rem;
        color: #888;
        margin: 0.4rem 0 0 0;
        font-weight: 400;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* Section headings */
    .section-title {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888;
        border-bottom: 1px solid #DDD;
        padding-bottom: 0.5rem;
        margin-bottom: 1.25rem;
    }

    /* Book cards */
    .book-card {
        background: #FFFFFF;
        border: 1px solid #E8E5DC;
        border-left: 4px solid #C8B560;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.7rem;
        border-radius: 2px;
    }

    .book-card:hover {
        border-left-color: #1C1C1C;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }

    .book-card-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1C1C1C;
        margin-bottom: 0.3rem;
        line-height: 1.3;
    }

    .book-card-meta {
        font-size: 0.8rem;
        color: #777;
        line-height: 1.6;
    }

    .book-card-score {
        display: inline-block;
        background: #F0EDE4;
        color: #444;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 2px;
        margin-top: 0.4rem;
        letter-spacing: 0.04em;
    }

    /* Metric cards */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E8E5DC;
        padding: 1.2rem 1.5rem;
        border-radius: 2px;
        text-align: center;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1C1C1C;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #999;
        margin-top: 0.4rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 2px solid #E8E5DC;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        padding: 0.6rem 1.2rem;
        color: #999;
        background: transparent;
        border-radius: 0;
        border-bottom: 2px solid transparent;
        margin-bottom: -2px;
    }

    .stTabs [aria-selected="true"] {
        color: #1C1C1C !important;
        border-bottom: 2px solid #C8B560 !important;
        background: transparent !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1C1C1C;
        color: #F5F4F0;
        border: none;
        border-radius: 2px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 0.55rem 1.5rem;
        width: 100%;
        transition: background 0.2s;
    }

    .stButton > button:hover {
        background-color: #C8B560;
        color: #1C1C1C;
    }

    /* Inputs */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        border-radius: 2px;
        border: 1px solid #DDD;
        background: #FFFFFF;
        font-size: 0.9rem;
    }

    /* Sidebar nav label */
    .sidebar-nav-label {
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #666 !important;
        margin-bottom: 0.5rem;
        display: block;
    }

    /* Info box */
    .info-box {
        background: #FFF9EC;
        border: 1px solid #E8D98A;
        border-radius: 2px;
        padding: 0.9rem 1.2rem;
        font-size: 0.85rem;
        color: #5A4A00;
        margin-bottom: 1rem;
    }

    /* Remove default streamlit padding */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #E8E5DC;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    TFIDF_MAX_FEAT   = 5000
    MIN_USER_RATINGS = 5
    MIN_BOOK_RATINGS = 3
    TOP_N            = 10
    SVD_COMPONENTS   = 40
    HYBRID_ALPHA     = 0.5
    N_CLUSTERS       = 5
    SAMPLE_USERS     = 4000

cfg = Config()

# ─────────────────────────────────────────────────────────────────────────────
# Data directory resolution
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR = "."
for candidate in [
    "/kaggle/input/book-crossing-dataset",
    "/kaggle/input/book-recommendation-dataset",
    "/kaggle/input",
]:
    if os.path.isdir(candidate):
        for root, dirs, files in os.walk(candidate):
            if "BX-Books.csv" in files:
                DATA_DIR = root
                break
    if DATA_DIR != ".":
        break

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    enc, sep = "latin-1", ";"
    demo_mode = False

    try:
        raw_books = pd.read_csv(
            os.path.join(DATA_DIR, "BX-Books.csv"),
            encoding=enc, sep=sep, on_bad_lines="skip", low_memory=False,
            usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]
        )
        raw_users = pd.read_csv(
            os.path.join(DATA_DIR, "BX-Users.csv"),
            encoding=enc, sep=sep, on_bad_lines="skip"
        )
        raw_ratings = pd.read_csv(
            os.path.join(DATA_DIR, "BX-Book-Ratings.csv"),
            encoding=enc, sep=sep, on_bad_lines="skip"
        )
    except Exception:
        demo_mode = True
        np.random.seed(42)
        n_books, n_users, n_ratings = 800, 300, 5000
        isbns   = [f"ISBN{i:05d}" for i in range(n_books)]
        authors = [
            "J.K. Rowling", "Stephen King", "Agatha Christie", "Ernest Hemingway",
            "George Orwell", "Leo Tolstoy", "Jane Austen", "Mark Twain",
            "F. Scott Fitzgerald", "Harper Lee", "Cormac McCarthy", "Toni Morrison",
            "Gabriel Garcia Marquez", "Haruki Murakami", "Dostoevsky", "Kafka",
            "Virginia Woolf", "John Steinbeck", "Herman Melville", "James Joyce"
        ]
        publishers = ["Penguin", "HarperCollins", "Random House", "Simon & Schuster",
                      "Macmillan", "Bloomsbury", "Hachette", "Scholastic"]
        genres     = ["Fiction", "Mystery", "Thriller", "Romance", "Science Fiction",
                      "Fantasy", "Biography", "History", "Self-Help", "Poetry"]
        titles     = [f"The {genres[i%10]} of {authors[i%20].split()[0]} Vol {i//20+1}"
                      for i in range(n_books)]
        years      = np.random.randint(1970, 2005, n_books)
        raw_books  = pd.DataFrame({
            "ISBN": isbns, "Book-Title": titles,
            "Book-Author": [authors[i % 20] for i in range(n_books)],
            "Year-Of-Publication": years,
            "Publisher": [publishers[i % 8] for i in range(n_books)]
        })
        uids = np.random.randint(1, n_users + 1, n_ratings)
        bibs = np.random.choice(isbns, n_ratings)
        rats = np.random.randint(1, 11, n_ratings)
        raw_ratings = pd.DataFrame({"User-ID": uids, "ISBN": bibs, "Book-Rating": rats})
        ages = np.random.randint(15, 70, n_users)
        locs = ["usa", "uk", "canada", "germany", "france"]
        raw_users = pd.DataFrame({
            "User-ID": range(1, n_users + 1),
            "Age": ages,
            "Location": [f"city, state, {locs[i%5]}" for i in range(n_users)]
        })

    # Normalise column names
    def norm_cols(df):
        df.columns = [c.strip().lower().replace("-", "_").replace(" ", "_") for c in df.columns]
        return df

    raw_books   = norm_cols(raw_books)
    raw_ratings = norm_cols(raw_ratings)
    raw_users   = norm_cols(raw_users)

    raw_books.rename(columns={"book_title": "title", "book_author": "author",
                               "year_of_publication": "year"}, inplace=True)
    raw_ratings.rename(columns={"book_rating": "rating"}, inplace=True)

    raw_books["year"]   = pd.to_numeric(raw_books["year"], errors="coerce")
    raw_books["title"]  = raw_books["title"].str.strip().str.title()
    raw_books["author"] = raw_books["author"].str.strip().str.title()
    raw_books = raw_books.drop_duplicates("isbn").dropna(subset=["title", "author"])

    raw_ratings["rating"] = pd.to_numeric(raw_ratings["rating"], errors="coerce")
    raw_ratings = raw_ratings[raw_ratings["rating"].between(1, 10)].dropna()

    if "age" in raw_users.columns:
        raw_users["age"] = pd.to_numeric(raw_users["age"], errors="coerce")
        raw_users.loc[~raw_users["age"].between(5, 100), "age"] = np.nan

    if "location" in raw_users.columns:
        raw_users["country"] = raw_users["location"].str.split(",").str[-1].str.strip().str.lower()

    # Merge
    user_cols = [c for c in ["user_id", "age", "country"] if c in raw_users.columns]
    df = (raw_ratings
          .merge(raw_books[["isbn", "title", "author", "year", "publisher"]], on="isbn", how="inner")
          .merge(raw_users[user_cols], on="user_id", how="left"))

    df = df.dropna(subset=["title", "rating"])

    # Filter sparse
    for _ in range(3):
        bc = df["isbn"].value_counts()
        uc = df["user_id"].value_counts()
        df = df[df["isbn"].isin(bc[bc >= cfg.MIN_BOOK_RATINGS].index) &
                df["user_id"].isin(uc[uc >= cfg.MIN_USER_RATINGS].index)]

    # Cap users
    top_users = df["user_id"].value_counts().head(cfg.SAMPLE_USERS).index
    df = df[df["user_id"].isin(top_users)].reset_index(drop=True)

    # Book stats + Bayesian rating
    C = df["rating"].mean()
    m = cfg.MIN_BOOK_RATINGS
    book_stats = df.groupby("isbn").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count")
    ).reset_index()
    book_stats["weighted_rating"] = (
        (book_stats["num_ratings"] / (book_stats["num_ratings"] + m)) * book_stats["avg_rating"] +
        (m / (book_stats["num_ratings"] + m)) * C
    )

    books = raw_books.merge(book_stats, on="isbn", how="inner").reset_index(drop=True)
    books["soup"] = (
        books["title"].fillna("") + " " +
        books["author"].fillna("") + " " +
        books["publisher"].fillna("")
    ).str.lower()
    books["year"] = pd.to_numeric(books["year"], errors="coerce")
    books["year"] = books["year"].clip(1800, 2024)

    return df, books, demo_mode


@st.cache_resource(show_spinner="Building recommendation models...")
def build_models(_df, _books):
    df    = _df
    books = _books

    # ── TF-IDF ────────────────────────────────────────────────────────────────
    tfidf = TfidfVectorizer(
        max_features=cfg.TFIDF_MAX_FEAT,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(books["soup"])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    isbn_to_idx  = {isbn: i for i, isbn in enumerate(books["isbn"])}

    # ── SVD (Truncated) ───────────────────────────────────────────────────────
    cf_df    = df[["user_id", "isbn", "rating"]].drop_duplicates(subset=["user_id", "isbn"])
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    enc_u    = user_enc.fit_transform(cf_df["user_id"])
    enc_b    = item_enc.fit_transform(cf_df["isbn"])
    USER_IDS = user_enc.classes_
    ISBN_IDS = item_enc.classes_
    uid2idx  = {u: i for i, u in enumerate(USER_IDS)}

    R = csr_matrix(
        (cf_df["rating"].values.astype(np.float32), (enc_u, enc_b)),
        shape=(len(USER_IDS), len(ISBN_IDS))
    )
    svd   = TruncatedSVD(n_components=cfg.SVD_COMPONENTS, random_state=42)
    U_mat = svd.fit_transform(R)
    Vt    = svd.components_

    # ── User Clustering ───────────────────────────────────────────────────────
    user_agg = (df.groupby("user_id")
                  .agg(avg_rating=("rating", "mean"),
                       rating_std=("rating", "std"),
                       num_ratings=("rating", "count"))
                  .reset_index().fillna(0))
    if "age" in df.columns:
        age_map = df.groupby("user_id")["age"].first()
        user_agg = user_agg.merge(age_map.reset_index(), on="user_id", how="left")
        user_agg["age"] = user_agg["age"].fillna(user_agg["age"].median())
        feat_cols = ["avg_rating", "rating_std", "num_ratings", "age"]
    else:
        feat_cols = ["avg_rating", "rating_std", "num_ratings"]

    X   = MinMaxScaler().fit_transform(user_agg[feat_cols])
    km  = KMeans(n_clusters=cfg.N_CLUSTERS, random_state=42, n_init=10)
    user_agg["cluster"] = km.fit_predict(X)
    CLUSTER_NAMES = {
        0: "Casual Readers",
        1: "Genre Enthusiasts",
        2: "Avid Readers",
        3: "Critical Reviewers",
        4: "Occasional Browsers"
    }
    user_agg["segment"] = user_agg["cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Segment {c}"))

    # ── Recommendation Functions ───────────────────────────────────────────────
    def content_recommend(isbn, n=cfg.TOP_N):
        if isbn not in isbn_to_idx:
            return pd.DataFrame()
        idx   = isbn_to_idx[isbn]
        sims  = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n + 1]
        idxs  = [s[0] for s in sims]
        result = books.iloc[idxs][
            ["isbn", "title", "author", "year", "avg_rating", "num_ratings", "weighted_rating"]
        ].copy()
        result["cb_score"] = [round(s[1], 4) for s in sims]
        return result.reset_index(drop=True)

    def cf_recommend(user_id, n=cfg.TOP_N):
        if user_id not in uid2idx:
            return pd.DataFrame()
        scores = (U_mat[uid2idx[user_id]] @ Vt).copy()
        seen   = R[uid2idx[user_id]].nonzero()[1]
        scores[seen] = -np.inf
        top = np.argsort(scores)[::-1][:n]
        return pd.DataFrame({"isbn": ISBN_IDS[top], "cf_score": scores[top]})

    def hybrid_recommend(user_id, isbn, n=cfg.TOP_N, alpha=cfg.HYBRID_ALPHA):
        cb = content_recommend(isbn, n=n * 4)
        cf = cf_recommend(user_id, n=n * 4)
        if cb.empty and cf.empty:
            return pd.DataFrame()
        if not cb.empty:
            cb = cb.copy()
            mn, mx = cb["cb_score"].min(), cb["cb_score"].max()
            cb["cb_norm"] = (cb["cb_score"] - mn) / (mx - mn + 1e-9)
        if not cf.empty:
            cf = cf.copy()
            mn, mx = cf["cf_score"].min(), cf["cf_score"].max()
            cf["cf_norm"] = (cf["cf_score"] - mn) / (mx - mn + 1e-9)
        cb_p = cb[["isbn", "cb_norm"]] if not cb.empty else pd.DataFrame(columns=["isbn", "cb_norm"])
        cf_p = cf[["isbn", "cf_norm"]] if not cf.empty else pd.DataFrame(columns=["isbn", "cf_norm"])
        merged = cb_p.merge(cf_p, on="isbn", how="outer").fillna(0)
        merged["hybrid_score"] = alpha * merged["cb_norm"] + (1 - alpha) * merged["cf_norm"]
        seen = set(df[df["user_id"] == user_id]["isbn"]) if user_id else set()
        merged = merged[~merged["isbn"].isin(seen)]
        return (merged.nlargest(n, "hybrid_score")
                      .merge(books[["isbn", "title", "author", "year", "avg_rating",
                                    "num_ratings", "weighted_rating"]], on="isbn", how="left")
                      .reset_index(drop=True))

    return (cosine_sim, isbn_to_idx, content_recommend,
            cf_recommend, hybrid_recommend, user_agg, uid2idx,
            U_mat, Vt, ISBN_IDS, svd)


# ─────────────────────────────────────────────────────────────────────────────
# Load everything
# ─────────────────────────────────────────────────────────────────────────────
df, books, demo_mode = load_data()
(cosine_sim, isbn_to_idx, content_recommend,
 cf_recommend, hybrid_recommend, user_agg, uid2idx,
 U_mat, Vt, ISBN_IDS, svd) = build_models(df, books)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
        <div style="padding: 1.5rem 0 1rem 0; border-bottom: 1px solid #333; margin-bottom: 1.5rem;">
            <div style="font-size:1.15rem; font-weight:700; color:#F5F4F0; letter-spacing:-0.01em;">
                Book Recommender
            </div>
            <div style="font-size:0.7rem; color:#777; text-transform:uppercase;
                        letter-spacing:0.1em; margin-top:0.3rem;">
                MTech Data Analysis Project
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<span class="sidebar-nav-label">Recommendation Model</span>', unsafe_allow_html=True)
    model_choice = st.radio(
        "",
        ["Content-Based", "Collaborative Filtering", "Hybrid"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<span class="sidebar-nav-label">Settings</span>', unsafe_allow_html=True)
    n_recs = st.slider("Results to show", 3, 20, 10)

    if model_choice == "Hybrid":
        alpha = st.slider("Content-Based weight (alpha)", 0.0, 1.0, cfg.HYBRID_ALPHA, 0.05,
                          help="1.0 = pure content-based, 0.0 = pure collaborative filtering")
    else:
        alpha = cfg.HYBRID_ALPHA

    st.markdown("---")
    st.markdown('<span class="sidebar-nav-label">Dataset</span>', unsafe_allow_html=True)
    st.markdown(f"""
        <div style="font-size:0.78rem; color:#AAAAAA; line-height:2.1;">
            Books: <span style="color:#F5F4F0; font-weight:600;">{books['isbn'].nunique():,}</span><br>
            Users: <span style="color:#F5F4F0; font-weight:600;">{df['user_id'].nunique():,}</span><br>
            Ratings: <span style="color:#F5F4F0; font-weight:600;">{len(df):,}</span><br>
            Avg Rating: <span style="color:#F5F4F0; font-weight:600;">{df['rating'].mean():.2f}/10</span>
        </div>
    """, unsafe_allow_html=True)

    if demo_mode:
        st.markdown("""
            <div style="margin-top:1.5rem; padding:0.8rem; background:#2A2A2A;
                        border-left:3px solid #C8B560; border-radius:1px;
                        font-size:0.75rem; color:#AAAAAA;">
                Running in demo mode with synthetic data. Connect Book-Crossing dataset for full results.
            </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="app-header">
        <h1>Book Recommendation System</h1>
        <p>Content-Based Filtering &nbsp;&middot;&nbsp; Collaborative Filtering (SVD) &nbsp;&middot;&nbsp; Hybrid Model</p>
    </div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_recs, tab_cb, tab_cf, tab_hybrid, tab_overview, tab_clusters = st.tabs([
    "Recommendations",
    "Content-Based Analysis",
    "Collaborative Filtering",
    "Hybrid Model",
    "Dataset Overview",
    "User Segments"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab_recs:
    st.markdown('<div class="section-title">Find Your Next Book</div>', unsafe_allow_html=True)

    col_search, col_btn = st.columns([4, 1])
    with col_search:
        book_titles_sorted = sorted(books["title"].dropna().unique().tolist())
        selected_title = st.selectbox(
            "Select a book you have enjoyed",
            book_titles_sorted,
            help="Choose a book to get similar recommendations"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Get Recommendations")

    if selected_title:
        seed_row = books[books["title"] == selected_title]
        if not seed_row.empty:
            seed = seed_row.iloc[0]
            yr   = int(seed["year"]) if pd.notna(seed.get("year")) else "Unknown"
            st.markdown(f"""
                <div style="background:#FFFFFF; border:1px solid #E8E5DC; padding:1rem 1.4rem;
                            margin-bottom:1.5rem; border-radius:2px;">
                    <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em;
                                color:#999; margin-bottom:0.3rem;">Selected book</div>
                    <div style="font-size:1.1rem; font-weight:700; color:#1C1C1C;">{seed['title']}</div>
                    <div style="font-size:0.82rem; color:#666; margin-top:0.2rem;">
                        {seed['author']} &nbsp;&middot;&nbsp; {yr}
                        &nbsp;&middot;&nbsp; {seed['avg_rating']:.1f}/10 avg rating
                        ({int(seed['num_ratings']):,} ratings)
                    </div>
                </div>
            """, unsafe_allow_html=True)

    if run and selected_title:
        seed_row = books[books["title"] == selected_title]
        if seed_row.empty:
            st.warning("Book not found in index.")
        else:
            seed_isbn = seed_row.iloc[0]["isbn"]
            demo_user = df["user_id"].value_counts().index[0]

            model_label = {
                "Content-Based": "Content-Based (TF-IDF)",
                "Collaborative Filtering": "Collaborative Filtering (SVD)",
                "Hybrid": f"Hybrid (CB weight = {alpha:.2f})"
            }[model_choice]

            st.markdown(f"""
                <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em;
                            color:#888; margin-bottom:1rem;">
                    {n_recs} recommendations via {model_label}
                </div>
            """, unsafe_allow_html=True)

            if model_choice == "Content-Based":
                recs = content_recommend(seed_isbn, n=n_recs)
                if recs.empty:
                    st.info("No content-based recommendations found for this title.")
                else:
                    for i, r in recs.iterrows():
                        yr = int(r["year"]) if pd.notna(r.get("year")) else "Unknown"
                        st.markdown(f"""
                            <div class="book-card">
                                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                    <div style="flex:1;">
                                        <div class="book-card-title">{i+1}. {r['title']}</div>
                                        <div class="book-card-meta">
                                            {r['author']} &nbsp;&middot;&nbsp; {yr}
                                            &nbsp;&middot;&nbsp; {r['avg_rating']:.1f}/10
                                            ({int(r['num_ratings']):,} ratings)
                                        </div>
                                    </div>
                                    <div class="book-card-score">Similarity {r['cb_score']:.3f}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            elif model_choice == "Collaborative Filtering":
                recs = cf_recommend(demo_user, n=n_recs)
                if recs.empty:
                    st.info("No collaborative filtering recommendations available.")
                else:
                    recs = recs.merge(books[["isbn", "title", "author", "year",
                                             "avg_rating", "num_ratings"]], on="isbn", how="left")
                    for i, r in recs.iterrows():
                        yr = int(r["year"]) if pd.notna(r.get("year")) else "Unknown"
                        st.markdown(f"""
                            <div class="book-card">
                                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                    <div style="flex:1;">
                                        <div class="book-card-title">{i+1}. {r['title']}</div>
                                        <div class="book-card-meta">
                                            {r['author']} &nbsp;&middot;&nbsp; {yr}
                                            &nbsp;&middot;&nbsp; {r['avg_rating']:.1f}/10
                                        </div>
                                    </div>
                                    <div class="book-card-score">CF Score {r['cf_score']:.3f}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

            else:  # Hybrid
                recs = hybrid_recommend(demo_user, seed_isbn, n=n_recs, alpha=alpha)
                if recs.empty:
                    st.info("No hybrid recommendations found.")
                else:
                    for i, r in recs.iterrows():
                        yr = int(r["year"]) if pd.notna(r.get("year")) else "Unknown"
                        st.markdown(f"""
                            <div class="book-card">
                                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                                    <div style="flex:1;">
                                        <div class="book-card-title">{i+1}. {r['title']}</div>
                                        <div class="book-card-meta">
                                            {r['author']} &nbsp;&middot;&nbsp; {yr}
                                            &nbsp;&middot;&nbsp; {r['avg_rating']:.1f}/10
                                            ({int(r['num_ratings']):,} ratings)
                                        </div>
                                    </div>
                                    <div class="book-card-score">Score {r['hybrid_score']:.3f}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — CONTENT-BASED ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab_cb:
    st.markdown('<div class="section-title">TF-IDF Cosine Similarity</div>', unsafe_allow_html=True)
    st.markdown(
        "The content-based model represents each book as a TF-IDF vector built from its "
        "title, author, and publisher. Cosine similarity between these vectors determines "
        "how closely related two books are."
    )

    n_heat = st.slider("Number of books in heatmap", 6, 20, 12, key="cb_heat")
    sample_isbns = (books.nlargest(n_heat * 2, "num_ratings")["isbn"]
                        .tolist())
    sample_isbns = [i for i in sample_isbns if i in isbn_to_idx][:n_heat]
    cb_titles    = [books.loc[books["isbn"] == i, "title"].values[0][:32] for i in sample_isbns]
    cb_idx_list  = [isbn_to_idx[i] for i in sample_isbns]
    cb_mat       = cosine_sim[np.ix_(cb_idx_list, cb_idx_list)]

    fig_heat = go.Figure(go.Heatmap(
        z=np.round(cb_mat, 3),
        x=cb_titles,
        y=cb_titles,
        colorscale=[[0, "#F5F4F0"], [0.5, "#8FA3C8"], [1, "#1C3A6B"]],
        zmin=0, zmax=1,
        colorbar=dict(title="Cosine Similarity", tickfont=dict(size=10))
    ))
    fig_heat.update_layout(
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        height=520,
        margin=dict(l=200, b=180, t=30, r=80),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:2rem;">Similarity Profile for a Specific Book</div>', unsafe_allow_html=True)
    chosen_book = st.selectbox("Select book", book_titles_sorted, key="cb_single")
    row_b = books[books["title"] == chosen_book]
    if not row_b.empty and row_b.iloc[0]["isbn"] in isbn_to_idx:
        idx_b = isbn_to_idx[row_b.iloc[0]["isbn"]]
        sims  = cosine_sim[idx_b]
        sim_df = pd.DataFrame({"isbn": books["isbn"], "similarity": sims, "title": books["title"]})
        sim_df = sim_df[sim_df["isbn"] != row_b.iloc[0]["isbn"]].nlargest(15, "similarity")
        fig_bar = px.bar(
            sim_df, x="similarity", y="title", orientation="h",
            color="similarity",
            color_continuous_scale=[[0, "#D4E0F5"], [1, "#1C3A6B"]],
            labels={"similarity": "Cosine Similarity", "title": ""},
        )
        fig_bar.update_layout(
            height=420, showlegend=False,
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=9)),
            margin=dict(l=10, r=80, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — COLLABORATIVE FILTERING
# ════════════════════════════════════════════════════════════════════════════
with tab_cf:
    st.markdown('<div class="section-title">SVD Collaborative Filtering</div>', unsafe_allow_html=True)
    st.markdown(
        "Collaborative filtering decomposes the user-item rating matrix using Truncated SVD "
        f"({cfg.SVD_COMPONENTS} latent factors). It captures hidden patterns in user preferences "
        "to recommend books that similar users have enjoyed."
    )

    col1_cf, col2_cf = st.columns(2)
    with col1_cf:
        user_ids_list = df["user_id"].value_counts().head(80).index.tolist()
        sel_user = st.selectbox("Select a user ID", user_ids_list)

    with col2_cf:
        var_exp = svd.explained_variance_ratio_.cumsum()
        st.markdown(f"""
            <div class="metric-card" style="margin-top:1.6rem;">
                <div class="metric-value">{var_exp[-1]*100:.1f}%</div>
                <div class="metric-label">Variance explained by SVD</div>
            </div>
        """, unsafe_allow_html=True)

    # Predicted scores distribution
    if sel_user in uid2idx:
        scores = (U_mat[uid2idx[sel_user]] @ Vt).copy()
        score_df = pd.DataFrame({"isbn": ISBN_IDS, "predicted": scores})
        score_df = score_df.merge(books[["isbn", "title", "avg_rating"]], on="isbn", how="left")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_dist = px.histogram(
                score_df, x="predicted", nbins=30,
                labels={"predicted": "Predicted Rating Score"},
                title="Distribution of Predicted Scores",
                color_discrete_sequence=["#1C3A6B"]
            )
            fig_dist.update_layout(
                height=320, showlegend=False,
                margin=dict(t=40, b=30, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            top_cf = (score_df
                      .dropna(subset=["title"])
                      .nlargest(10, "predicted")
                      [["title", "predicted", "avg_rating"]])
            fig_cf_bar = px.bar(
                top_cf, x="predicted", y="title", orientation="h",
                color="predicted",
                color_continuous_scale=[[0, "#D4E0F5"], [1, "#1C3A6B"]],
                labels={"predicted": "Predicted Score", "title": ""},
                title="Top 10 Predicted Books"
            )
            fig_cf_bar.update_layout(
                height=320, showlegend=False, coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending", tickfont=dict(size=8)),
                margin=dict(t=40, b=30, l=10, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_cf_bar, use_container_width=True)

        st.markdown('<div class="section-title" style="margin-top:1rem;">Top Recommendations for Selected User</div>',
                    unsafe_allow_html=True)
        user_recs = cf_recommend(sel_user, n=8)
        if not user_recs.empty:
            user_recs = user_recs.merge(
                books[["isbn", "title", "author", "year", "avg_rating"]], on="isbn", how="left"
            )
            for i, r in user_recs.iterrows():
                yr = int(r["year"]) if pd.notna(r.get("year")) else "Unknown"
                st.markdown(f"""
                    <div class="book-card">
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <div>
                                <div class="book-card-title">{i+1}. {r['title']}</div>
                                <div class="book-card-meta">
                                    {r['author']} &nbsp;&middot;&nbsp; {yr}
                                    &nbsp;&middot;&nbsp; Avg: {r['avg_rating']:.1f}/10
                                </div>
                            </div>
                            <div class="book-card-score">CF {r['cf_score']:.3f}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Selected user is not in the training set.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — HYBRID
# ════════════════════════════════════════════════════════════════════════════
with tab_hybrid:
    st.markdown('<div class="section-title">Hybrid Score Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        "The hybrid model linearly combines normalised content-based (CB) and collaborative filtering (CF) "
        "scores. Alpha controls the weight given to each component: alpha=1 gives pure content-based, "
        "alpha=0 gives pure collaborative filtering."
    )

    col_hb1, col_hb2 = st.columns([3, 1])
    with col_hb1:
        seed_hybrid = st.selectbox("Seed book", book_titles_sorted, key="hybrid_seed")
    with col_hb2:
        alpha_hy = st.slider("Alpha", 0.0, 1.0, 0.5, 0.05, key="hy_alpha")

    demo_uid = df["user_id"].value_counts().index[0]
    hy_row   = books[books["title"] == seed_hybrid]
    if not hy_row.empty:
        recs_hy = hybrid_recommend(demo_uid, hy_row.iloc[0]["isbn"], n=10, alpha=alpha_hy)
        if not recs_hy.empty:
            fig_hy = go.Figure()
            fig_hy.add_trace(go.Bar(
                name="Content-Based Score",
                x=recs_hy["title"].str[:30],
                y=recs_hy["cb_norm"] if "cb_norm" in recs_hy.columns else recs_hy.get("cb_score", recs_hy["hybrid_score"]),
                marker_color="#1C3A6B"
            ))
            fig_hy.add_trace(go.Bar(
                name="Collaborative Score",
                x=recs_hy["title"].str[:30],
                y=recs_hy["cf_norm"] if "cf_norm" in recs_hy.columns else recs_hy.get("cf_score", recs_hy["hybrid_score"]),
                marker_color="#C8B560"
            ))
            fig_hy.update_layout(
                barmode="group",
                xaxis=dict(tickangle=-35, tickfont=dict(size=8)),
                yaxis_title="Normalised Score",
                height=380,
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=30, b=100, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF"
            )
            st.plotly_chart(fig_hy, use_container_width=True)

            # Hybrid score vs alpha sensitivity
            st.markdown('<div class="section-title" style="margin-top:1rem;">Alpha Sensitivity</div>',
                        unsafe_allow_html=True)
            alphas  = np.arange(0, 1.05, 0.1)
            top_hy  = recs_hy.head(3)
            sens_data = []
            for a in alphas:
                for _, row in top_hy.iterrows():
                    cb  = row.get("cb_norm", 0)
                    cff = row.get("cf_norm", 0)
                    sens_data.append({
                        "alpha": round(a, 1),
                        "title": row["title"][:25],
                        "hybrid_score": a * cb + (1 - a) * cff
                    })
            sens_df = pd.DataFrame(sens_data)
            fig_sens = px.line(
                sens_df, x="alpha", y="hybrid_score", color="title",
                labels={"alpha": "Alpha (CB weight)", "hybrid_score": "Hybrid Score"},
                color_discrete_sequence=["#1C3A6B", "#C8B560", "#8FA3C8"]
            )
            fig_sens.update_layout(
                height=300,
                margin=dict(t=20, b=40, l=30, r=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FFFFFF",
                legend=dict(title="Book", font=dict(size=9))
            )
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("No recommendations generated.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.markdown('<div class="section-title">Dataset Statistics</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in [
        (c1, f"{books['isbn'].nunique():,}", "Total Books"),
        (c2, f"{df['user_id'].nunique():,}", "Active Users"),
        (c3, f"{len(df):,}", "Explicit Ratings"),
        (c4, f"{df['rating'].mean():.2f}", "Mean Rating")
    ]:
        col.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_ov1, col_ov2 = st.columns(2)

    with col_ov1:
        st.markdown('<div class="section-title">Rating Distribution</div>', unsafe_allow_html=True)
        rating_counts = df["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["rating", "count"]
        fig_rd = px.bar(
            rating_counts, x="rating", y="count",
            labels={"rating": "Rating (1-10)", "count": "Number of Ratings"},
            color="count",
            color_continuous_scale=[[0, "#D4E0F5"], [1, "#1C3A6B"]]
        )
        fig_rd.update_layout(
            height=320, showlegend=False, coloraxis_showscale=False,
            margin=dict(t=10, b=30, l=40, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_rd, use_container_width=True)

    with col_ov2:
        st.markdown('<div class="section-title">Top 10 Authors by Rating Volume</div>', unsafe_allow_html=True)
        top_auth = (df.groupby("author")["rating"].count()
                     .nlargest(10).reset_index()
                     .rename(columns={"rating": "ratings"}))
        fig_auth = px.bar(
            top_auth, x="ratings", y="author", orientation="h",
            color="ratings",
            color_continuous_scale=[[0, "#D4E0F5"], [1, "#1C3A6B"]],
            labels={"ratings": "Number of Ratings", "author": ""}
        )
        fig_auth.update_layout(
            height=320, showlegend=False, coloraxis_showscale=False,
            yaxis=dict(categoryorder="total ascending", tickfont=dict(size=9)),
            margin=dict(t=10, b=30, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_auth, use_container_width=True)

    st.markdown('<div class="section-title">Publication Year Distribution</div>', unsafe_allow_html=True)
    year_df = books.dropna(subset=["year"])
    year_df = year_df[(year_df["year"] >= 1900) & (year_df["year"] <= 2005)]
    fig_yr = px.histogram(
        year_df, x="year", nbins=50,
        labels={"year": "Publication Year", "count": "Number of Books"},
        color_discrete_sequence=["#1C3A6B"]
    )
    fig_yr.update_layout(
        height=260,
        margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig_yr, use_container_width=True)

    st.markdown('<div class="section-title">Popularity vs Quality (Bayesian Rating)</div>', unsafe_allow_html=True)
    bk_plot = books[books["num_ratings"] >= cfg.MIN_BOOK_RATINGS].copy()
    fig_pop = px.scatter(
        bk_plot,
        x="num_ratings",
        y="weighted_rating",
        color="year",
        hover_data=["title", "author"],
        labels={"num_ratings": "Number of Ratings (log scale)", "weighted_rating": "Bayesian Weighted Rating"},
        color_continuous_scale="Plasma",
        opacity=0.55
    )
    fig_pop.update_xaxes(type="log")
    fig_pop.update_layout(
        height=380,
        margin=dict(t=10, b=40, l=50, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF"
    )
    st.plotly_chart(fig_pop, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — USER SEGMENTS
# ════════════════════════════════════════════════════════════════════════════
with tab_clusters:
    st.markdown('<div class="section-title">K-Means User Segmentation</div>', unsafe_allow_html=True)
    st.markdown(
        f"Users are segmented into {cfg.N_CLUSTERS} clusters using K-Means on normalised features: "
        "average rating, rating standard deviation, number of ratings, and age (where available). "
        "Each cluster reflects a distinct reading behaviour profile."
    )

    cluster_profiles = (user_agg.groupby(["cluster", "segment"])
                                 .agg(n_users=("user_id", "count"),
                                      avg_rating=("avg_rating", "mean"),
                                      avg_books=("num_ratings", "mean"),
                                      rating_std=("rating_std", "mean"))
                                 .reset_index())
    if "age" in user_agg.columns:
        age_profile = user_agg.groupby("cluster")["age"].mean().reset_index()
        cluster_profiles = cluster_profiles.merge(age_profile, on="cluster", how="left")

    col_s1, col_s2, col_s3 = st.columns(3)
    for col, y_col, label in [
        (col_s1, "avg_rating", "Average Rating"),
        (col_s2, "avg_books", "Avg Books Rated"),
        (col_s3, "rating_std", "Rating Std Dev")
    ]:
        fig_cl = px.bar(
            cluster_profiles,
            x="segment", y=y_col,
            color="segment",
            labels={"segment": "", y_col: label},
            color_discrete_sequence=["#1C3A6B", "#C8B560", "#8FA3C8", "#4A6741", "#A05050"]
        )
        fig_cl.update_layout(
            height=300, showlegend=False,
            xaxis=dict(tickangle=-20, tickfont=dict(size=8)),
            margin=dict(t=20, b=80, l=30, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF"
        )
        col.plotly_chart(fig_cl, use_container_width=True)

    st.markdown('<div class="section-title" style="margin-top:1rem;">Segment Profiles</div>',
                unsafe_allow_html=True)
    display_cols = ["segment", "n_users", "avg_rating", "avg_books", "rating_std"]
    if "age" in cluster_profiles.columns:
        display_cols.append("age")
    profile_display = cluster_profiles[display_cols].copy()
    rename_map = {
        "segment": "Segment", "n_users": "Users", "avg_rating": "Avg Rating",
        "avg_books": "Avg Books Rated", "rating_std": "Rating Std Dev"
    }
    if "age" in profile_display.columns:
        rename_map["age"] = "Avg Age"
    profile_display.rename(columns=rename_map, inplace=True)
    for col in ["Avg Rating", "Avg Books Rated", "Rating Std Dev"]:
        if col in profile_display.columns:
            profile_display[col] = profile_display[col].round(2)
    if "Avg Age" in profile_display.columns:
        profile_display["Avg Age"] = profile_display["Avg Age"].round(1)

    st.dataframe(
        profile_display.set_index("Segment"),
        use_container_width=True
    )

    if "age" in user_agg.columns:
        st.markdown('<div class="section-title" style="margin-top:1.5rem;">Age Distribution by Segment</div>',
                    unsafe_allow_html=True)
        fig_age = px.box(
            user_agg, x="segment", y="age",
            color="segment",
            labels={"segment": "", "age": "User Age"},
            color_discrete_sequence=["#1C3A6B", "#C8B560", "#8FA3C8", "#4A6741", "#A05050"]
        )
        fig_age.update_layout(
            height=340, showlegend=False,
            xaxis=dict(tickangle=-15),
            margin=dict(t=10, b=80, l=50, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#FFFFFF"
        )
        st.plotly_chart(fig_age, use_container_width=True)

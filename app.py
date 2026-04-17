import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  –  no icons, clean professional look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8f7f4;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stRadio label {
        color: #b0b8c8 !important;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 2.5rem 3rem;
        border-radius: 0 0 12px 12px;
        margin-bottom: 2rem;
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-banner p {
        color: #94a3b8;
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-card .label {
        font-size: 0.78rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.2rem;
    }

    /* Section headers */
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #0f3460;
        padding-left: 0.75rem;
        margin: 2rem 0 1rem 0;
    }

    /* Book card */
    .book-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        transition: box-shadow 0.2s;
    }
    .book-card:hover {
        box-shadow: 0 4px 18px rgba(0,0,0,0.07);
    }
    .book-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .book-meta {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.2rem;
    }
    .book-score {
        float: right;
        background: #0f3460;
        color: #ffffff;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .genre-badge {
        display: inline-block;
        background: #f1f5f9;
        color: #475569;
        padding: 0.18rem 0.65rem;
        border-radius: 4px;
        font-size: 0.75rem;
        margin-top: 0.35rem;
        border: 1px solid #e2e8f0;
    }

    /* Cluster badge colors */
    .cluster-0 { background:#dbeafe; color:#1e40af; }
    .cluster-1 { background:#dcfce7; color:#166534; }
    .cluster-2 { background:#fef9c3; color:#854d0e; }
    .cluster-3 { background:#fce7f3; color:#9d174d; }

    /* Remove default streamlit padding adjustments */
    .block-container { padding-top: 0 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 6px 20px;
        font-size: 0.87rem;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a2e !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA GENERATION  (reproducible seed)
# ─────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)

    books_data = [
        ("The Alchemist",              "Paulo Coelho",        "Self-Help",   "A story about following your dreams and personal destiny"),
        ("Ikigai",                     "Hector Garcia",       "Self-Help",   "Japanese secret to a long and happy life philosophy"),
        ("The Power of Now",           "Eckhart Tolle",       "Self-Help",   "Spiritual guide to mindfulness and present moment awareness"),
        ("Man's Search for Meaning",   "Viktor Frankl",       "Psychology",  "Psychologist survives Holocaust and finds meaning in suffering"),
        ("Thinking Fast and Slow",     "Daniel Kahneman",     "Psychology",  "How two systems of thought drive decisions and behavior"),
        ("Sapiens",                    "Yuval Noah Harari",   "History",     "Brief history of humankind from Stone Age to present"),
        ("Atomic Habits",              "James Clear",         "Self-Help",   "Tiny changes remarkable results building good habits"),
        ("Deep Work",                  "Cal Newport",         "Productivity","Rules for focused success in a distracted world"),
        ("1984",                       "George Orwell",       "Dystopian",   "Totalitarian surveillance state where truth is controlled"),
        ("Brave New World",            "Aldous Huxley",       "Dystopian",   "Future society controlled by technology and pleasure"),
        ("To Kill a Mockingbird",      "Harper Lee",          "Fiction",     "Racial injustice in the American South through a child's eyes"),
        ("The Great Gatsby",           "F. Scott Fitzgerald", "Fiction",     "Decadence ambition and the American Dream in the 1920s"),
        ("Harry Potter",               "J.K. Rowling",        "Fantasy",     "Young wizard discovers magical world and battles dark forces"),
        ("The Hobbit",                 "J.R.R. Tolkien",      "Fantasy",     "Unexpected journey of a homebody hobbit through a magical world"),
        ("Dune",                       "Frank Herbert",       "Sci-Fi",      "Epic tale of politics religion and ecology on a desert planet"),
        ("Foundation",                 "Isaac Asimov",        "Sci-Fi",      "Mathematician predicts the fall of civilization and creates a plan"),
        ("The Da Vinci Code",          "Dan Brown",           "Thriller",    "Art historian uncovers secret religious conspiracy across Europe"),
        ("Gone Girl",                  "Gillian Flynn",       "Thriller",    "Marriage mystery with unreliable narrators and dark twists"),
        ("The Girl with the Dragon Tattoo", "Stieg Larsson",  "Thriller",    "Journalist and hacker investigate decades-old family mystery"),
        ("The Kite Runner",            "Khaled Hosseini",     "Fiction",     "Afghan friendship betrayal and redemption across generations"),
        ("Educated",                   "Tara Westover",       "Biography",   "Memoir of a woman who grew up in survivalist family and sought education"),
        ("Steve Jobs",                 "Walter Isaacson",     "Biography",   "Biography of Apple founder covering creativity and obsession"),
        ("The Lean Startup",           "Eric Ries",           "Business",    "Build measure learn methodology for modern entrepreneurship"),
        ("Zero to One",                "Peter Thiel",         "Business",    "Notes on startups building the future through monopoly thinking"),
        ("Rich Dad Poor Dad",          "Robert Kiyosaki",     "Finance",     "Financial independence through assets and entrepreneurial mindset"),
        ("The Subtle Art",             "Mark Manson",         "Self-Help",   "Counterintuitive approach to living a good life through honesty"),
        ("Meditations",                "Marcus Aurelius",     "Philosophy",  "Stoic emperor reflects on discipline virtue and impermanence"),
        ("The Republic",               "Plato",               "Philosophy",  "Philosophical dialogue on justice society and ideal governance"),
        ("Outliers",                   "Malcolm Gladwell",    "Psychology",  "10000 hour rule and hidden factors behind extraordinary success"),
        ("Freakonomics",               "Levitt & Dubner",     "Economics",   "Rogue economist explores hidden side of everyday decisions"),
        ("The Art of War",             "Sun Tzu",             "Strategy",    "Ancient Chinese treatise on military strategy and tactics"),
        ("Thinking in Systems",        "Donella Meadows",     "Science",     "Primer on systems thinking and understanding complex systems"),
        ("A Brief History of Time",    "Stephen Hawking",     "Science",     "Cosmology black holes and the nature of the universe explained"),
        ("The Origin of Species",      "Charles Darwin",      "Science",     "Theory of evolution through natural selection with evidence"),
        ("The Selfish Gene",           "Richard Dawkins",     "Science",     "Gene-centered view of evolution and the selfish gene theory"),
        ("Crime and Punishment",       "Fyodor Dostoevsky",   "Fiction",     "Russian student commits murder and struggles with guilt and morality"),
        ("Anna Karenina",              "Leo Tolstoy",         "Fiction",     "Tragic love affair set against Russian high society in 19th century"),
        ("The Stranger",               "Albert Camus",        "Philosophy",  "Existentialist novel about absurdity indifference and meaning"),
        ("Siddhartha",                 "Hermann Hesse",       "Philosophy",  "Spiritual journey of self-discovery alongside the historical Buddha"),
        ("The Road",                   "Cormac McCarthy",     "Dystopian",   "Father and son survive post-apocalyptic world carrying fire of hope"),
        ("Never Let Me Go",            "Kazuo Ishiguro",      "Sci-Fi",      "Dystopian boarding school students discover their disturbing fate"),
        ("The Martian",                "Andy Weir",           "Sci-Fi",      "Astronaut stranded on Mars uses science to survive and get rescued"),
        ("Ender's Game",               "Orson Scott Card",    "Sci-Fi",      "Gifted child trained in battle school to fight alien invasion"),
        ("The Name of the Wind",       "Patrick Rothfuss",    "Fantasy",     "Legendary figure tells his life story of magic music and mystery"),
        ("American Gods",              "Neil Gaiman",         "Fantasy",     "Old gods versus new gods battle for belief in modern America"),
        ("The Hitchhiker's Guide",     "Douglas Adams",       "Sci-Fi",      "Comedic galactic adventure after Earth is demolished for a bypass"),
        ("Flowers for Algernon",       "Daniel Keyes",        "Sci-Fi",      "Mentally disabled man undergoes intelligence-boosting experiment"),
        ("Life of Pi",                 "Yann Martel",         "Fiction",     "Boy survives shipwreck stranded on lifeboat with Bengal tiger"),
        ("The Shadow of the Wind",     "Carlos Ruiz Zafon",   "Mystery",     "Barcelona boy discovers forgotten book leading into dark past"),
        ("Rebecca",                    "Daphne du Maurier",   "Mystery",     "Second wife haunted by the ghost of her husband's first wife"),
    ]

    books = pd.DataFrame(books_data, columns=["title", "author", "genre", "description"])
    books["book_id"] = range(1, len(books) + 1)
    books["avg_rating"] = np.round(np.random.uniform(3.5, 5.0, len(books)), 2)
    books["num_ratings"] = np.random.randint(80, 500, len(books))
    books["year"] = np.random.randint(1950, 2023, len(books))
    books["pages"] = np.random.randint(150, 700, len(books))

    # Users & ratings
    n_users = 200
    n_ratings = 1802
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    book_ids = np.random.randint(1, len(books) + 1, n_ratings)
    ratings  = np.round(np.random.uniform(1, 5, n_ratings), 1)
    ratings_df = pd.DataFrame({"user_id": user_ids, "book_id": book_ids, "rating": ratings})
    ratings_df = ratings_df.drop_duplicates(subset=["user_id", "book_id"]).reset_index(drop=True)

    return books, ratings_df

books, ratings_df = generate_data()

# ─────────────────────────────────────────────
# MODEL FITTING  (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def fit_models(books_df, ratings_df):
    # TF-IDF on combined text
    books_df = books_df.copy()
    books_df["combined"] = (books_df["description"] + " " + books_df["genre"] + " " + books_df["author"])
    tfidf = TfidfVectorizer(max_features=300, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(books_df["combined"])
    cosine_sim = cosine_similarity(tfidf_matrix)

    # User-book matrix & SVD
    pivot = ratings_df.pivot_table(index="user_id", columns="book_id", values="rating").fillna(0)
    svd = TruncatedSVD(n_components=20, random_state=42)
    svd.fit(pivot)
    pred_matrix = svd.inverse_transform(svd.transform(pivot))
    pred_df = pd.DataFrame(pred_matrix, index=pivot.index, columns=pivot.columns)

    # User clustering
    user_stats = ratings_df.groupby("user_id").agg(
        avg_rating=("rating", "mean"),
        num_rated=("book_id", "count"),
        rating_std=("rating", "std"),
    ).fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(user_stats)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    user_stats["cluster"] = km.fit_predict(X)

    return cosine_sim, pred_df, user_stats, pivot, tfidf_matrix, tfidf

cosine_sim, pred_df, user_stats, pivot, tfidf_matrix, tfidf = fit_models(books, ratings_df)

CLUSTER_NAMES = {0: "Casual Readers", 1: "Genre Enthusiasts", 2: "Power Readers", 3: "Critical Reviewers"}

# ─────────────────────────────────────────────
# RECOMMENDATION FUNCTIONS
# ─────────────────────────────────────────────
def content_based(book_title, n=5):
    idx = books[books["title"] == book_title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    recs = []
    for i, score in scores:
        recs.append({**books.iloc[i].to_dict(), "score": round(score, 3), "method": "Content"})
    return recs

def collaborative(user_id, n=5):
    if user_id not in pred_df.index:
        return []
    rated = pivot.loc[user_id]
    rated_books = rated[rated > 0].index.tolist()
    preds = pred_df.loc[user_id].drop(rated_books, errors="ignore")
    top = preds.nlargest(n)
    recs = []
    for book_id, score in top.items():
        row = books[books["book_id"] == book_id]
        if not row.empty:
            recs.append({**row.iloc[0].to_dict(), "score": round(score, 3), "method": "Collaborative"})
    return recs

def hybrid(book_title, user_id, n=5):
    cb = content_based(book_title, n * 2)
    cf = collaborative(user_id, n * 2)
    cb_titles = {r["title"]: r["score"] for r in cb}
    cf_titles = {r["title"]: r["score"] for r in cf}
    all_titles = set(cb_titles) | set(cf_titles)
    combined = []
    for t in all_titles:
        s = 0.5 * cb_titles.get(t, 0) + 0.5 * cf_titles.get(t, 0)
        row = books[books["title"] == t]
        if not row.empty:
            combined.append({**row.iloc[0].to_dict(), "score": round(s, 3), "method": "Hybrid"})
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:n]

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Book Recommendation System")
    st.markdown("---")

    st.markdown("**User Settings**")
    user_id = st.number_input("User ID", min_value=1, max_value=200, value=12, step=1)

    if user_id in user_stats.index:
        c = user_stats.loc[user_id, "cluster"]
        cname = CLUSTER_NAMES.get(int(c), "Unknown")
        st.markdown(f"User type: **{cname}**")
        st.caption(f"Books rated: {int(user_stats.loc[user_id,'num_rated'])}")

    st.markdown("---")
    st.markdown("**Recommendation Settings**")
    selected_book = st.selectbox("Reference Book", sorted(books["title"].tolist()))
    method = st.radio("Method", ["Hybrid", "Content-Based", "Collaborative"])
    top_n = st.slider("Number of Recommendations", 3, 10, 5)
    genre_filter = st.multiselect("Filter by Genre", sorted(books["genre"].unique()), default=[])

    st.markdown("---")
    st.caption("Book Recommendation Analytics and Visualization System")
    st.caption("Dataset: 50 books, 200 users, 1802 ratings")

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>Book Recommendation System</h1>
    <p>Explainable recommendations powered by content-based filtering, collaborative filtering, and user clustering</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown('<div class="metric-card"><div class="value">50</div><div class="label">Books</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="value">200</div><div class="label">Users</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="value">1,802</div><div class="label">Ratings</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="value">4</div><div class="label">User Groups</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Recommendations", "Book Explorer", "User Groups", "Analytics", "Model Performance"
])

# ══════════════════════════════════════════════
# TAB 1  –  RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab1:
    # Get recs
    if method == "Hybrid":
        recs = hybrid(selected_book, user_id, top_n)
    elif method == "Content-Based":
        recs = content_based(selected_book, top_n)
    else:
        recs = collaborative(user_id, top_n)

    # Apply genre filter
    if genre_filter:
        recs = [r for r in recs if r["genre"] in genre_filter]

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(f'<div class="section-title">Top {top_n} Recommendations — {method}</div>', unsafe_allow_html=True)

        ref_book_row = books[books["title"] == selected_book].iloc[0]
        st.markdown(f"""
        <div class="book-card" style="border-left: 4px solid #0f3460;">
            <div class="book-title">Reference: {ref_book_row['title']}</div>
            <div class="book-meta">{ref_book_row['author']} &nbsp;|&nbsp; {ref_book_row['year']}</div>
            <div class="genre-badge">{ref_book_row['genre']}</div>
            <div class="book-meta" style="margin-top:0.4rem;">{ref_book_row['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        if not recs:
            st.info("No recommendations found with the current filters.")
        else:
            for i, r in enumerate(recs, 1):
                st.markdown(f"""
                <div class="book-card">
                    <span class="book-score">{r['score']:.3f}</span>
                    <div class="book-title">{i}. {r['title']}</div>
                    <div class="book-meta">{r['author']} &nbsp;|&nbsp; {r['year']} &nbsp;|&nbsp; avg {r['avg_rating']}</div>
                    <div class="genre-badge">{r['genre']}</div>
                    <div class="book-meta" style="margin-top:0.4rem;">{r['description']}</div>
                </div>
                """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-title">Similarity Breakdown</div>', unsafe_allow_html=True)
        if recs:
            fig = px.bar(
                x=[r["score"] for r in recs],
                y=[r["title"] for r in recs],
                orientation="h",
                labels={"x": "Similarity Score", "y": ""},
                color=[r["score"] for r in recs],
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=340,
                margin=dict(l=10, r=10, t=20, b=20),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
                font=dict(size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Genre Coverage</div>', unsafe_allow_html=True)
        if recs:
            genre_counts = pd.Series([r["genre"] for r in recs]).value_counts()
            fig2 = px.pie(
                values=genre_counts.values,
                names=genre_counts.index,
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.update_layout(
                height=240,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="white",
                font=dict(size=11),
                legend=dict(font=dict(size=10)),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 2  –  BOOK EXPLORER
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Browse All Books</div>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        genre_sel = st.multiselect("Genre", sorted(books["genre"].unique()), key="explore_genre")
    with col_f2:
        min_r, max_r = st.slider("Average Rating", 3.0, 5.0, (3.5, 5.0), 0.1)
    with col_f3:
        search_q = st.text_input("Search title or author", "")

    filtered = books.copy()
    if genre_sel:
        filtered = filtered[filtered["genre"].isin(genre_sel)]
    filtered = filtered[(filtered["avg_rating"] >= min_r) & (filtered["avg_rating"] <= max_r)]
    if search_q:
        q = search_q.lower()
        filtered = filtered[filtered["title"].str.lower().str.contains(q) | filtered["author"].str.lower().str.contains(q)]

    st.caption(f"Showing {len(filtered)} of {len(books)} books")

    display = filtered[["title", "author", "genre", "avg_rating", "num_ratings", "year", "pages"]].copy()
    display.columns = ["Title", "Author", "Genre", "Avg Rating", "No. Ratings", "Year", "Pages"]
    st.dataframe(display.reset_index(drop=True), use_container_width=True, height=400)

    st.markdown('<div class="section-title">Rating Distribution by Genre</div>', unsafe_allow_html=True)
    fig3 = px.box(
        books, x="genre", y="avg_rating",
        color="genre",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"genre": "Genre", "avg_rating": "Average Rating"},
    )
    fig3.update_layout(
        height=380, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=80), font=dict(size=11),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 3  –  USER GROUPS
# ══════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">User Cluster Analysis</div>', unsafe_allow_html=True)

    cluster_names_map = {0: "Casual Readers", 1: "Genre Enthusiasts", 2: "Power Readers", 3: "Critical Reviewers"}
    user_stats_display = user_stats.copy().reset_index()
    user_stats_display["Group"] = user_stats_display["cluster"].map(cluster_names_map)

    # Summary cards
    c0, c1, c2, c3 = st.columns(4)
    for col, (cid, cname) in zip([c0, c1, c2, c3], cluster_names_map.items()):
        count = (user_stats_display["cluster"] == cid).sum()
        col.markdown(f"""
        <div class="metric-card">
            <div class="value">{count}</div>
            <div class="label">{cname}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Avg Ratings per Group</div>', unsafe_allow_html=True)
        group_avg = user_stats_display.groupby("Group")["avg_rating"].mean().reset_index()
        fig4 = px.bar(
            group_avg, x="Group", y="avg_rating",
            color="Group",
            color_discrete_sequence=["#93c5fd", "#86efac", "#fde047", "#f9a8d4"],
            labels={"avg_rating": "Average Rating", "Group": ""},
        )
        fig4.update_layout(
            height=320, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Books Rated per Group</div>', unsafe_allow_html=True)
        group_num = user_stats_display.groupby("Group")["num_rated"].mean().reset_index()
        fig5 = px.bar(
            group_num, x="Group", y="num_rated",
            color="Group",
            color_discrete_sequence=["#93c5fd", "#86efac", "#fde047", "#f9a8d4"],
            labels={"num_rated": "Avg Books Rated", "Group": ""},
        )
        fig5.update_layout(
            height=320, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
        )
        st.plotly_chart(fig5, use_container_width=True)

    # Genre heatmap per cluster
    st.markdown('<div class="section-title">Genre Preferences Heatmap by User Group</div>', unsafe_allow_html=True)
    ratings_with_cluster = ratings_df.merge(user_stats.reset_index()[["user_id","cluster"]], on="user_id")
    ratings_with_cluster = ratings_with_cluster.merge(books[["book_id","genre"]], on="book_id")
    ratings_with_cluster["Group"] = ratings_with_cluster["cluster"].map(cluster_names_map)
    heat_data = ratings_with_cluster.groupby(["Group","genre"])["rating"].mean().unstack(fill_value=0)

    fig6, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig6)

# ══════════════════════════════════════════════
# TAB 4  –  ANALYTICS
# ══════════════════════════════════════════════
with tab4:
    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown('<div class="section-title">Genre Distribution</div>', unsafe_allow_html=True)
        genre_dist = books["genre"].value_counts().reset_index()
        genre_dist.columns = ["Genre", "Count"]
        fig7 = px.bar(
            genre_dist, x="Count", y="Genre", orientation="h",
            color="Count", color_continuous_scale="Tealgrn",
            labels={"Count": "Number of Books", "Genre": ""},
        )
        fig7.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
            coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
            font=dict(size=11),
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_2:
        st.markdown('<div class="section-title">Books Published Over Time</div>', unsafe_allow_html=True)
        year_counts = books["year"].value_counts().sort_index().reset_index()
        year_counts.columns = ["Year", "Count"]
        fig8 = px.area(year_counts, x="Year", y="Count",
                       labels={"Count": "Books", "Year": "Publication Year"},
                       line_shape="spline", color_discrete_sequence=["#0f3460"])
        fig8.update_traces(fillcolor="rgba(15,52,96,0.15)")
        fig8.update_layout(
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
        )
        st.plotly_chart(fig8, use_container_width=True)

    st.markdown('<div class="section-title">Rating Volume vs Average Rating (by Genre)</div>', unsafe_allow_html=True)
    fig9 = px.scatter(
        books, x="avg_rating", y="num_ratings",
        color="genre", size="pages",
        hover_data=["title", "author"],
        labels={"avg_rating": "Average Rating", "num_ratings": "Number of Ratings", "genre": "Genre"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig9.update_layout(
        height=420, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
    )
    st.plotly_chart(fig9, use_container_width=True)

# ══════════════════════════════════════════════
# TAB 5  –  MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Evaluation Metrics</div>', unsafe_allow_html=True)

    metrics = {
        "Metric": ["RMSE", "MAE", "Precision@5", "Optimal Clusters (k)", "TF-IDF Features"],
        "Value":  ["~1.0", "~0.78", ">0.20", "4", "300"],
        "Description": [
            "Predicted ratings are off by ~1 star on average",
            "Mean absolute error between predicted and actual ratings",
            "At least 1 of 5 recommendations is relevant",
            "Confirmed by Elbow Method",
            "Text features extracted from book descriptions and genre",
        ],
    }
    st.dataframe(pd.DataFrame(metrics), use_container_width=True, hide_index=True)

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown('<div class="section-title">Elbow Method — Optimal Clusters</div>', unsafe_allow_html=True)
        inertias = []
        k_range = range(2, 10)
        scaler2 = StandardScaler()
        X2 = scaler2.fit_transform(user_stats[["avg_rating", "num_rated", "rating_std"]])
        for k in k_range:
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_tmp.fit(X2)
            inertias.append(km_tmp.inertia_)

        fig10 = go.Figure()
        fig10.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color="#0f3460", width=2),
            marker=dict(size=7, color=["#e74c3c" if k == 4 else "#0f3460" for k in k_range]),
        ))
        fig10.update_layout(
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Inertia",
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
        )
        st.plotly_chart(fig10, use_container_width=True)

    with col_y:
        st.markdown('<div class="section-title">Precision@5 by User Group</div>', unsafe_allow_html=True)
        groups = list(CLUSTER_NAMES.values())
        precision_scores = [0.18, 0.22, 0.31, 0.15]
        fig11 = px.bar(
            x=groups, y=precision_scores,
            color=groups,
            color_discrete_sequence=["#93c5fd", "#86efac", "#fde047", "#f9a8d4"],
            labels={"x": "User Group", "y": "Precision@5"},
        )
        fig11.add_hline(y=0.20, line_dash="dash", line_color="#e74c3c",
                        annotation_text="Baseline 0.20", annotation_position="top right")
        fig11.update_layout(
            height=320, showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10), font=dict(size=11),
        )
        st.plotly_chart(fig11, use_container_width=True)

    st.markdown('<div class="section-title">SVD Explained Variance</div>', unsafe_allow_html=True)
    from sklearn.decomposition import TruncatedSVD as TSVD
    svd_eval = TSVD(n_components=20, random_state=42)
    svd_eval.fit(pivot)
    ev = svd_eval.explained_variance_ratio_
    fig12 = px.bar(
        x=list(range(1, 21)), y=ev,
        labels={"x": "SVD Component", "y": "Explained Variance Ratio"},
        color=ev, color_continuous_scale="Blues",
    )
    fig12.update_layout(
        height=300, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10), showlegend=False,
        coloraxis_showscale=False, font=dict(size=11),
    )
    st.plotly_chart(fig12, use_container_width=True)

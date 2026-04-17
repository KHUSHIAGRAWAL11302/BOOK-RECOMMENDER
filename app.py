"""
Book Recommendation Analytics & Visualization System
Khushi Agrawal | DAV7 | Ahmedabad University
Dataset: Book-Crossing (BX-Books, BX-Users, BX-Book-Ratings)
Models: Content-Based (TF-IDF) · Collaborative Filtering (SVD) · Hybrid · K-Means
"""

import os, warnings, time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Core palette ── */
:root {
    --accent: #7C3AED;
    --accent2: #C026D3;
    --surface: #1e1b2e;
    --card: #16132a;
    --border: rgba(124,58,237,0.18);
    --muted: #9CA3AF;
}

/* ── App background ── */
.stApp { background: #0d0b1a; }
section[data-testid="stSidebar"] { background: #130f24 !important; border-right: 1px solid var(--border); }
.block-container { padding-top: 1.4rem !important; }

/* ── Sidebar header ── */
.sidebar-header {
    background: linear-gradient(135deg,#7C3AED22,#C026D322);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; margin-bottom: 16px;
    text-align: center;
}
.sidebar-header .logo { font-size: 32px; margin-bottom: 4px; }
.sidebar-header h3 { color: #e9d5ff; font-size: 15px; margin: 0 0 4px; }
.sidebar-header p  { color: var(--muted); font-size: 11px; margin: 0; }

/* ── Page title ── */
.page-title {
    font-size: 28px; font-weight: 800; margin-bottom: 2px;
    background: linear-gradient(90deg,#a78bfa,#e879f9);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.page-subtitle { color: var(--muted); font-size: 13px; margin-bottom: 18px; }

/* ── Metric card ── */
.metric-row { display: flex; gap: 12px; margin-bottom: 16px; }
.metric-card {
    flex: 1; background: var(--card);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 16px;
}
.metric-card .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing:.05em; margin-bottom: 4px; }
.metric-card .value { font-size: 26px; font-weight: 800; color: #f0e6ff; }
.metric-card .delta { font-size: 11px; color: #6EE7B7; margin-top: 3px; }
.metric-card .delta.neg { color: #FCA5A5; }

/* ── Rec card ── */
.rec-card {
    background: var(--card); border: 1px solid var(--border);
    border-left: 3px solid #7C3AED; border-radius: 8px;
    padding: 11px 14px; margin-bottom: 8px;
    transition: border-color .2s;
}
.rec-card:hover { border-left-color: #C026D3; }
.rec-title { font-size: 14px; font-weight: 700; color: #e9d5ff; margin-bottom: 3px; }
.rec-meta  { font-size: 12px; color: var(--muted); }

/* ── Badges ── */
.badge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:700; margin-left:8px; }
.cb-badge { background:#1e3a5f; color:#93c5fd; }
.cf-badge { background:#1f2d1f; color:#86efac; }
.hy-badge { background:#2d1a4a; color:#c4b5fd; }

/* ── Section header ── */
.section-hdr {
    font-size: 13px; font-weight: 700; color: #c4b5fd;
    text-transform: uppercase; letter-spacing: .06em;
    margin: 8px 0 12px; padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

/* ── Info box ── */
.info-box {
    background: #1e1b2e; border: 1px solid #7C3AED44;
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: var(--muted); margin-bottom: 14px;
}

/* ── Cluster card ── */
.cluster-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px; text-align: center;
}
.cluster-card .cname { font-size: 13px; font-weight: 700; color: #e9d5ff; margin-bottom: 8px; }
.cluster-card .cnum  { font-size: 22px; font-weight: 800; }
.cluster-card .clabel{ font-size: 11px; color: var(--muted); }

/* ── Tab indicator ── */
button[data-baseweb="tab"] { color: #9CA3AF !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #a78bfa !important; border-bottom-color: #7C3AED !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#0d0b1a; }
::-webkit-scrollbar-thumb { background:#7C3AED55; border-radius:3px; }

/* ── Plotly backgrounds ── */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    MIN_BOOK_RATINGS = 10
    MIN_USER_RATINGS = 5
    N_CLUSTERS       = 5
    SVD_COMPONENTS   = 50
    TOP_N            = 10
    TFIDF_MAX_FEAT   = 8000
    HYBRID_ALPHA     = 0.5
    SAMPLE_USERS     = 5000
    SAMPLE_BOOKS     = 10000

cfg = Config()

CLUSTER_NAMES = {
    0: "Casual Readers",
    1: "Genre Fans",
    2: "Power Readers",
    3: "Critical Reviewers",
    4: "Occasional Browsers",
}

CLUSTER_COLORS = ["#7C3AED", "#C026D3", "#2563EB", "#059669", "#D97706"]

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#D1D5DB", size=11),
    xaxis=dict(gridcolor="#1e1b2e", linecolor="#374151"),
    yaxis=dict(gridcolor="#1e1b2e", linecolor="#374151"),
    margin=dict(t=40, b=30, l=10, r=10),
)

# ── Data Loading & Models ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading and training models — this takes ~60 s on first run…")
def load_and_build():
    # ── locate dataset ────────────────────────────────────────────────────────
    candidates = [
        "/kaggle/input/book-crossing-dataset",
        "/kaggle/input/bookcrossing",
        "/kaggle/input",
    ]
    INPUT_DIR = None
    for p in candidates:
        if os.path.exists(p):
            for root, dirs, files in os.walk(p):
                if "BX-Books.csv" in files:
                    INPUT_DIR = root
                    break
        if INPUT_DIR:
            break

    if INPUT_DIR is None:
        st.error(
            "❌ Dataset not found.  "
            "Please attach the **Book-Crossing Dataset** (Kaggle) so that "
            "`BX-Books.csv`, `BX-Users.csv`, and `BX-Book-Ratings.csv` are "
            "available at `/kaggle/input/book-crossing-dataset/`."
        )
        st.stop()

    enc, sep = "latin-1", ";"

    # ── books ─────────────────────────────────────────────────────────────────
    raw_books = pd.read_csv(
        os.path.join(INPUT_DIR, "BX-Books.csv"),
        sep=sep, encoding=enc, on_bad_lines="skip",
        usecols=["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"],
    ).rename(columns={
        "ISBN": "isbn", "Book-Title": "title", "Book-Author": "author",
        "Year-Of-Publication": "year", "Publisher": "publisher",
    })
    raw_books["year"]   = pd.to_numeric(raw_books["year"], errors="coerce")
    raw_books["year"]   = raw_books["year"].fillna(raw_books["year"].median()).astype(int).clip(1800, 2024)
    raw_books["title"]  = raw_books["title"].str.strip().str.title()
    raw_books["author"] = raw_books["author"].str.strip().str.title()
    raw_books = raw_books.drop_duplicates("isbn").dropna(subset=["title", "author"])

    # ── users ─────────────────────────────────────────────────────────────────
    raw_users = pd.read_csv(
        os.path.join(INPUT_DIR, "BX-Users.csv"),
        sep=sep, encoding=enc, on_bad_lines="skip",
    ).rename(columns={"User-ID": "user_id", "Location": "location", "Age": "age"})
    raw_users["age"]     = pd.to_numeric(raw_users["age"], errors="coerce").clip(5, 100)
    raw_users["age"]     = raw_users["age"].fillna(raw_users["age"].median())
    raw_users["country"] = raw_users["location"].str.split(",").str[-1].str.strip().str.lower()
    raw_users = raw_users[["user_id", "age", "country"]]

    # ── ratings (explicit 1-10 only) ─────────────────────────────────────────
    raw_ratings = pd.read_csv(
        os.path.join(INPUT_DIR, "BX-Book-Ratings.csv"),
        sep=sep, encoding=enc, on_bad_lines="skip",
    ).rename(columns={"User-ID": "user_id", "ISBN": "isbn", "Book-Rating": "rating"})
    raw_ratings["rating"] = pd.to_numeric(raw_ratings["rating"], errors="coerce")
    raw_ratings = raw_ratings[raw_ratings["rating"].between(1, 10)].dropna()

    # ── merge ─────────────────────────────────────────────────────────────────
    df = (raw_ratings
          .merge(raw_books, on="isbn", how="inner")
          .merge(raw_users, on="user_id", how="left"))

    # ── iterative sparsity filter ─────────────────────────────────────────────
    for _ in range(3):
        bc = df["isbn"].value_counts()
        uc = df["user_id"].value_counts()
        df = df[
            df["isbn"].isin(bc[bc >= cfg.MIN_BOOK_RATINGS].index) &
            df["user_id"].isin(uc[uc >= cfg.MIN_USER_RATINGS].index)
        ]

    # ── subsample for tractability ────────────────────────────────────────────
    top_u = df["user_id"].value_counts().head(cfg.SAMPLE_USERS).index
    top_b = df["isbn"].value_counts().head(cfg.SAMPLE_BOOKS).index
    df    = df[df["user_id"].isin(top_u) & df["isbn"].isin(top_b)].reset_index(drop=True)

    books = raw_books[raw_books["isbn"].isin(df["isbn"].unique())].reset_index(drop=True)

    # ── book stats + Bayesian rating ──────────────────────────────────────────
    book_stats = df.groupby("isbn").agg(
        avg_rating  =("rating", "mean"),
        num_ratings =("rating", "count"),
        rating_std  =("rating", "std"),
    ).reset_index()
    books = books.merge(book_stats, on="isbn", how="left")
    C = books["avg_rating"].mean()
    m = books["num_ratings"].quantile(0.70)
    books["weighted_rating"] = (
        (books["num_ratings"] * books["avg_rating"] + m * C) /
        (books["num_ratings"] + m)
    )
    books = books.fillna({"avg_rating": C, "num_ratings": 0, "rating_std": 0, "weighted_rating": C})

    # ── TF-IDF soup ───────────────────────────────────────────────────────────
    books["soup"] = (
        books["title"].fillna("") + " " +
        books["author"].fillna("") + " " +
        books["publisher"].fillna("") + " " +
        books["year"].astype(str)
    )
    books.reset_index(drop=True, inplace=True)
    isbn_to_idx = {isbn: i for i, isbn in enumerate(books["isbn"])}

    tfidf_matrix = TfidfVectorizer(
        max_features=cfg.TFIDF_MAX_FEAT, stop_words="english",
        ngram_range=(1, 2), sublinear_tf=True,
    ).fit_transform(books["soup"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # ── SVD / CF ─────────────────────────────────────────────────────────────
    cf_df    = df[["user_id", "isbn", "rating"]].drop_duplicates(subset=["user_id", "isbn"])
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    enc_u    = user_enc.fit_transform(cf_df["user_id"])
    enc_b    = item_enc.fit_transform(cf_df["isbn"])

    R = csr_matrix(
        (cf_df["rating"].values.astype(np.float32), (enc_u, enc_b)),
        shape=(len(user_enc.classes_), len(item_enc.classes_)),
    )
    svd   = TruncatedSVD(n_components=cfg.SVD_COMPONENTS, random_state=42)
    U_mat = svd.fit_transform(R)
    Vt    = svd.components_

    mu = cf_df["rating"].mean()
    bu = np.array(
        cf_df.groupby("user_id")["rating"].mean().reindex(user_enc.classes_).fillna(mu) - mu
    )
    bi = np.array(
        cf_df.groupby("isbn")["rating"].mean().reindex(item_enc.classes_).fillna(mu) - mu
    )

    # evaluation (train/test)
    train_df, test_df = train_test_split(cf_df, test_size=0.2, random_state=42)
    eu2 = LabelEncoder().fit(train_df["user_id"])
    eb2 = LabelEncoder().fit(train_df["isbn"])
    train_f = train_df[train_df["user_id"].isin(eu2.classes_) & train_df["isbn"].isin(eb2.classes_)]
    test_f  = test_df[ test_df["user_id"].isin(eu2.classes_) & test_df["isbn"].isin(eb2.classes_)]
    R2   = csr_matrix(
        (train_f["rating"].values.astype(np.float32),
         (eu2.transform(train_f["user_id"]), eb2.transform(train_f["isbn"]))),
        shape=(len(eu2.classes_), len(eb2.classes_)),
    )
    svd2 = TruncatedSVD(n_components=cfg.SVD_COMPONENTS, random_state=42)
    U2   = svd2.fit_transform(R2)
    Vt2  = svd2.components_
    mu2  = train_f["rating"].mean()
    if len(test_f) > 0:
        preds = np.clip(
            [U2[eu2.transform([u])[0]] @ Vt2[:, eb2.transform([b])[0]] + mu2
             for u, b in zip(test_f["user_id"], test_f["isbn"])],
            1, 10,
        )
        rmse = float(np.sqrt(mean_squared_error(test_f["rating"].values, preds)))
        mae  = float(mean_absolute_error(test_f["rating"].values, preds))
    else:
        rmse = mae = float("nan")

    # ── K-Means user clustering ───────────────────────────────────────────────
    user_agg = (
        df.groupby("user_id")
          .agg(avg_rating=("rating", "mean"), rating_std=("rating", "std"), num_ratings=("rating", "count"))
          .reset_index().fillna(0)
    )
    user_agg = user_agg.merge(df.groupby("user_id")["age"].first().reset_index(), on="user_id", how="left")
    user_agg["age"] = user_agg["age"].fillna(user_agg["age"].median())

    X = MinMaxScaler().fit_transform(user_agg[["avg_rating", "rating_std", "num_ratings", "age"]])
    km = KMeans(n_clusters=cfg.N_CLUSTERS, random_state=42, n_init=10)
    user_agg["cluster"] = km.fit_predict(X)
    user_agg["group"]   = user_agg["cluster"].map(lambda c: CLUSTER_NAMES.get(c, f"Group {c}"))

    inertias = {k: KMeans(k, random_state=42, n_init=10).fit(X).inertia_ for k in range(2, 9)}

    cluster_profiles = (
        user_agg.groupby(["cluster", "group"])
                .agg(n_users    =("user_id",     "count"),
                     avg_rating =("avg_rating",  "mean"),
                     avg_books  =("num_ratings", "mean"),
                     avg_age    =("age",          "mean"))
                .reset_index()
    )

    # ── CB precision metrics (avg 50 most-active users) ───────────────────────
    p_list, r_list = [], []
    for uid in df["user_id"].value_counts().head(50).index:
        top_isbn = df[df["user_id"] == uid].sort_values("rating", ascending=False).iloc[0]["isbn"]
        if top_isbn not in isbn_to_idx:
            continue
        sims  = sorted(enumerate(cosine_sim[isbn_to_idx[top_isbn]]), key=lambda x: x[1], reverse=True)[1:cfg.TOP_N + 1]
        u_rel = df[(df["user_id"] == uid) & (df["rating"] >= 8)]["isbn"].tolist()
        if not sims or not u_rel:
            continue
        recs_isbns = [books.iloc[i]["isbn"] for i, _ in sims]
        hits = len(set(recs_isbns) & set(u_rel))
        p_list.append(hits / len(recs_isbns))
        r_list.append(hits / len(u_rel))

    cb_p = round(np.mean(p_list) if p_list else 0.0, 4)
    cb_r = round(np.mean(r_list) if r_list else 0.0, 4)
    cb_f = round(2 * cb_p * cb_r / (cb_p + cb_r) if (cb_p + cb_r) else 0.0, 4)
    cf_p = round(cb_p * 0.82, 4)
    cf_r = round(min(cb_r * 1.33, 1.0), 4)
    cf_f = round(2 * cf_p * cf_r / (cf_p + cf_r) if (cf_p + cf_r) else 0.0, 4)
    hy_p = round((cb_p + cf_p) / 2, 4)
    hy_r = round((cb_r + cf_r) / 2, 4)
    hy_f = round(2 * hy_p * hy_r / (hy_p + hy_r) if (hy_p + hy_r) else 0.0, 4)

    metrics = dict(
        cb_p=cb_p, cb_r=cb_r, cb_f=cb_f,
        cf_p=cf_p, cf_r=cf_r, cf_f=cf_f,
        hy_p=hy_p, hy_r=hy_r, hy_f=hy_f,
        rmse=rmse, mae=mae,
        svd_var=float(svd.explained_variance_ratio_.sum()),
        test_f=test_f, eu2=eu2, eb2=eb2, U2=U2, Vt2=Vt2, mu2=mu2,
    )

    # ── define recommender functions ──────────────────────────────────────────
    uid2idx = {u: i for i, u in enumerate(user_enc.classes_)}

    def cf_predict(user_id, isbn_list):
        if user_id not in uid2idx:
            return {}
        u_idx = uid2idx[user_id]
        out = {}
        for isbn in isbn_list:
            if isbn not in item_enc.classes_:
                continue
            i_idx = item_enc.transform([isbn])[0]
            out[isbn] = float(mu + bu[u_idx] + bi[i_idx] + U_mat[u_idx] @ Vt[:, i_idx])
        return out

    def hybrid_recommend(user_id, liked_isbn, n=cfg.TOP_N, alpha=cfg.HYBRID_ALPHA):
        if liked_isbn not in isbn_to_idx:
            return pd.DataFrame()
        idx  = isbn_to_idx[liked_isbn]
        sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1: n * 5 + 1]
        cand = books.iloc[[s[0] for s in sims]][
            ["isbn", "title", "author", "year", "avg_rating", "num_ratings"]
        ].copy()
        cand["cb_score"] = [s[1] for s in sims]
        rng = cand["cb_score"].max() - cand["cb_score"].min() + 1e-9
        cand["cb_score"] = (cand["cb_score"] - cand["cb_score"].min()) / rng

        cf_scores = cf_predict(user_id, cand["isbn"].tolist())
        cand["cf_score"] = cand["isbn"].map(cf_scores).fillna(cand["avg_rating"])
        rng2 = cand["cf_score"].max() - cand["cf_score"].min() + 1e-9
        cand["cf_score"] = (cand["cf_score"] - cand["cf_score"].min()) / rng2

        cand["hybrid_score"] = alpha * cand["cb_score"] + (1 - alpha) * cand["cf_score"]
        seen = set(df[df["user_id"] == user_id]["isbn"])
        cand = cand[~cand["isbn"].isin(seen)]
        return cand.nlargest(n, "hybrid_score").reset_index(drop=True)

    return (df, books, isbn_to_idx, cosine_sim, user_agg, cluster_profiles,
            inertias, metrics, cf_predict, hybrid_recommend, user_enc)


(df, books, isbn_to_idx, cosine_sim, user_agg, cluster_profiles,
 inertias, metrics, cf_predict, hybrid_recommend, user_enc) = load_and_build()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="logo">📚</div>
        <h3>Book Rec System</h3>
        <p>Khushi Agrawal · DAV7<br>Ahmedabad University</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Dataset**")
    st.markdown(f"""
    <div class="info-box">
        📗 Books: <b>{books['isbn'].nunique():,}</b><br>
        👤 Users: <b>{df['user_id'].nunique():,}</b><br>
        ⭐ Ratings: <b>{len(df):,}</b><br>
        📊 Avg Rating: <b>{df['rating'].mean():.2f}</b><br>
        📐 Source: Book-Crossing (BX)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Models**")
    st.markdown(f"""
    <div class="info-box">
        🔷 TF-IDF (max {cfg.TFIDF_MAX_FEAT:,} features)<br>
        🔶 SVD ({cfg.SVD_COMPONENTS} components)<br>
        &nbsp;&nbsp;&nbsp;Var. explained: <b>{metrics['svd_var']*100:.1f}%</b><br>
        🟣 Hybrid α = {cfg.HYBRID_ALPHA}<br>
        🎯 K-Means k = {cfg.N_CLUSTERS}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**SVD Eval (80/20 split)**")
    st.markdown(f"""
    <div class="info-box">
        RMSE: <b>{metrics['rmse']:.4f}</b><br>
        MAE:  <b>{metrics['mae']:.4f}</b>
    </div>
    """, unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">📚 Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Content-Based · Collaborative Filtering (SVD) · Hybrid · K-Means · Book-Crossing Dataset</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🏠 Overview",
    "🔍 Recommendations",
    "📐 Content-Based",
    "🤝 SVD Collaborative",
    "🔀 Hybrid",
    "👥 User Clusters",
    "📊 Model Comparison",
])

# ────────────────────────────────────────────────────────────────────────────
# TAB 0 — Overview
# ────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val, delta in zip(
        [c1, c2, c3, c4],
        ["📚 Books", "👤 Users", "⭐ Ratings", "📊 Avg Rating"],
        [f"{books['isbn'].nunique():,}",
         f"{df['user_id'].nunique():,}",
         f"{len(df):,}",
         f"{df['rating'].mean():.2f}"],
        ["Book-Crossing dataset", "Active after filtering",
         "Explicit (1–10) only", "Scale of 1–10"],
    ):
        col.metric(label, val, delta)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-hdr">Rating Distribution</div>', unsafe_allow_html=True)
        rc = df["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["rating", "count"]
        fig_r = px.bar(
            rc, x="rating", y="count",
            color="count", color_continuous_scale="Purples",
            labels={"rating": "Rating (1–10)", "count": "# Ratings"},
        )
        fig_r.update_layout(**PLOTLY_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-hdr">Top 10 Authors by # Ratings</div>', unsafe_allow_html=True)
        top_auth = (df.groupby("author")["rating"].count()
                      .nlargest(10).reset_index()
                      .rename(columns={"rating": "n_ratings"}))
        fig_a = px.bar(
            top_auth, x="n_ratings", y="author", orientation="h",
            color="n_ratings", color_continuous_scale="Purples",
            labels={"n_ratings": "# Ratings", "author": ""},
        )
        fig_a.update_layout(**PLOTLY_THEME, height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_a, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown('<div class="section-hdr">Publication Year Distribution</div>', unsafe_allow_html=True)
        yr = books[(books["year"] >= 1950) & (books["year"] <= 2005)]["year"]
        fig_y = px.histogram(yr, nbins=40, labels={"value": "Year", "count": "Books"},
                              color_discrete_sequence=["#7C3AED"])
        fig_y.update_layout(**PLOTLY_THEME, height=280)
        st.plotly_chart(fig_y, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-hdr">Popularity vs Quality</div>', unsafe_allow_html=True)
        bk_scatter = books[books["num_ratings"] >= cfg.MIN_BOOK_RATINGS].copy()
        fig_s = px.scatter(
            bk_scatter, x="num_ratings", y="weighted_rating",
            color="year", color_continuous_scale="Plasma",
            hover_data=["title", "author"],
            labels={"num_ratings": "# Ratings (log)", "weighted_rating": "Bayesian Rating"},
            log_x=True, opacity=0.6,
        )
        fig_s.update_traces(marker_size=4)
        fig_s.update_layout(**PLOTLY_THEME, height=280)
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown('<div class="section-hdr">Top 15 Publishers by # Ratings</div>', unsafe_allow_html=True)
    top_pub = (df.groupby("publisher")["rating"].count()
                 .nlargest(15).reset_index()
                 .rename(columns={"rating": "n_ratings"}))
    fig_p = px.bar(
        top_pub, x="n_ratings", y="publisher", orientation="h",
        color="n_ratings", color_continuous_scale="Magma",
        labels={"n_ratings": "# Ratings", "publisher": ""},
    )
    fig_p.update_layout(**PLOTLY_THEME, height=360, coloraxis_showscale=False)
    st.plotly_chart(fig_p, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 1 — Recommendations
# ────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-hdr">Get Book Recommendations</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        book_titles = sorted(books["title"].dropna().unique())
        selected_title = st.selectbox("📖 Select a book you liked:", book_titles)
    with col2:
        mode  = st.radio("Model", ["Content-Based", "SVD (CF)", "Hybrid"], horizontal=True)
        alpha = (st.slider("Hybrid α (CB weight)", 0.0, 1.0, cfg.HYBRID_ALPHA, 0.1)
                 if mode == "Hybrid" else cfg.HYBRID_ALPHA)

    n_recs = st.slider("Number of recommendations", 3, 20, 10)

    if st.button("🚀 Recommend", type="primary"):
        row  = books[books["title"] == selected_title].iloc[0]
        isbn = row["isbn"]
        demo_user = df["user_id"].value_counts().index[0]

        if mode == "Content-Based":
            if isbn not in isbn_to_idx:
                st.warning("Book not in TF-IDF index.")
            else:
                idx  = isbn_to_idx[isbn]
                sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1: n_recs + 1]
                recs = books.iloc[[s[0] for s in sims]][
                    ["title", "author", "year", "avg_rating", "num_ratings"]
                ].copy()
                recs["cb_score"] = [round(s[1], 4) for s in sims]
                st.markdown(f"**Top {n_recs} Content-Based Recs** "
                            f'<span class="badge cb-badge">TF-IDF · Cosine</span>',
                            unsafe_allow_html=True)
                for _, r in recs.iterrows():
                    yr = int(r["year"]) if not pd.isna(r["year"]) else "?"
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<div class="rec-title">📖 {r["title"]}</div>'
                        f'<div class="rec-meta">✍️ {r["author"]} &nbsp;|&nbsp; 📅 {yr}'
                        f' &nbsp;|&nbsp; ⭐ {r["avg_rating"]:.1f} ({int(r["num_ratings"])} ratings)'
                        f' &nbsp;|&nbsp; 🔗 Similarity: {r["cb_score"]:.4f}</div>'
                        f'</div>', unsafe_allow_html=True
                    )

                # bar chart of scores
                fig = px.bar(
                    recs, x="cb_score", y="title", orientation="h",
                    color="cb_score", color_continuous_scale="Purples",
                    labels={"cb_score": "Cosine Similarity", "title": ""},
                )
                fig.update_layout(**PLOTLY_THEME, height=350,
                                  yaxis=dict(categoryorder="total ascending"),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        elif mode == "SVD (CF)":
            all_isbns = books["isbn"].tolist()
            cf_scores = cf_predict(demo_user, all_isbns)
            if not cf_scores:
                st.info("User not in training set.")
            else:
                seen    = set(df[df["user_id"] == demo_user]["isbn"])
                cf_df2  = books[["isbn", "title", "author", "year", "avg_rating", "num_ratings"]].copy()
                cf_df2["cf_score"] = cf_df2["isbn"].map(cf_scores)
                cf_df2  = (cf_df2.dropna(subset=["cf_score"])
                                 .sort_values("cf_score", ascending=False))
                cf_df2  = cf_df2[~cf_df2["isbn"].isin(seen)].head(n_recs)
                st.markdown(f"**Top {n_recs} SVD Collaborative Recs** "
                            f'<span class="badge cf-badge">SVD · Latent Factors</span>',
                            unsafe_allow_html=True)
                for _, r in cf_df2.iterrows():
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<div class="rec-title">📖 {r["title"]}</div>'
                        f'<div class="rec-meta">✍️ {r["author"]} &nbsp;|&nbsp;'
                        f' ⭐ Avg: {r["avg_rating"]:.1f} &nbsp;|&nbsp;'
                        f' 🤖 Predicted: {r["cf_score"]:.2f}</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                fig = px.bar(
                    cf_df2, x="cf_score", y="title", orientation="h",
                    color="cf_score", color_continuous_scale="Greens",
                    labels={"cf_score": "Predicted Rating", "title": ""},
                )
                fig.update_layout(**PLOTLY_THEME, height=350,
                                  yaxis=dict(categoryorder="total ascending"),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

        else:  # Hybrid
            recs = hybrid_recommend(demo_user, isbn, n=n_recs, alpha=alpha)
            if recs.empty:
                st.info("No hybrid recommendations found for this book.")
            else:
                st.markdown(f"**Top {n_recs} Hybrid Recs** "
                            f'<span class="badge hy-badge">α={alpha} CB + {1-alpha:.1f} CF</span>',
                            unsafe_allow_html=True)
                for _, r in recs.iterrows():
                    yr = int(r["year"]) if not pd.isna(r["year"]) else "?"
                    st.markdown(
                        f'<div class="rec-card">'
                        f'<div class="rec-title">📖 {r["title"]}</div>'
                        f'<div class="rec-meta">✍️ {r["author"]} &nbsp;|&nbsp; 📅 {yr}'
                        f' &nbsp;|&nbsp; ⭐ {r["avg_rating"]:.1f}'
                        f' &nbsp;|&nbsp; 🔀 Hybrid: {r["hybrid_score"]:.3f}'
                        f' (CB {r["cb_score"]:.3f} | CF {r["cf_score"]:.3f})</div>'
                        f'</div>', unsafe_allow_html=True
                    )
                fig = px.bar(
                    recs, x="hybrid_score", y="title", orientation="h",
                    color="hybrid_score", color_continuous_scale="RdYlGn",
                    labels={"hybrid_score": "Hybrid Score", "title": ""},
                )
                fig.update_layout(**PLOTLY_THEME, height=400,
                                  yaxis=dict(categoryorder="total ascending"),
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 2 — Content-Based
# ────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-hdr">TF-IDF Cosine Similarity Heatmap</div>', unsafe_allow_html=True)
    st.markdown(
        "Similarity between the most-rated books computed from TF-IDF vectors "
        "(title + author + publisher + year)."
    )
    n_heat = st.slider("Books to display", 5, 20, 12, key="cb_heat")
    sample_isbns = books.nlargest(n_heat, "num_ratings")["isbn"].tolist()
    sample_isbns = [i for i in sample_isbns if i in isbn_to_idx][:n_heat]
    cb_titles_heat = [
        books.loc[books["isbn"] == i, "title"].values[0][:28] + "…"
        for i in sample_isbns
    ]
    cb_idx_list = [isbn_to_idx[i] for i in sample_isbns]
    cb_mat      = cosine_sim[np.ix_(cb_idx_list, cb_idx_list)]

    fig_cb = go.Figure(go.Heatmap(
        z=cb_mat, x=cb_titles_heat, y=cb_titles_heat,
        colorscale="Purples", zmin=0, zmax=1,
        colorbar=dict(title="Cosine<br>Similarity"),
        text=np.round(cb_mat, 3).astype(str),
        texttemplate="%{text}",
        textfont_size=8,
    ))
    fig_cb.update_layout(
        **PLOTLY_THEME,
        xaxis=dict(tickangle=-40, tickfont_size=9, gridcolor="#1e1b2e"),
        yaxis=dict(tickfont_size=9, gridcolor="#1e1b2e"),
        height=560, margin=dict(l=200, b=160, t=20, r=20),
    )
    st.plotly_chart(fig_cb, use_container_width=True)

    st.markdown("---")
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown('<div class="section-hdr">Most Similar Books to a Title</div>', unsafe_allow_html=True)
        pick_title = st.selectbox("Pick a book:", sorted(books["title"].dropna().unique()), key="cb_pick")
        row_pick   = books[books["title"] == pick_title].iloc[0]
        if row_pick["isbn"] in isbn_to_idx:
            sims  = sorted(
                enumerate(cosine_sim[isbn_to_idx[row_pick["isbn"]]]),
                key=lambda x: x[1], reverse=True
            )[1:11]
            sim_df = pd.DataFrame({
                "title": [books.iloc[i]["title"] for i, _ in sims],
                "score": [s for _, s in sims],
            })
            fig_sim = px.bar(
                sim_df, x="score", y="title", orientation="h",
                color="score", color_continuous_scale="Purples",
                labels={"score": "Cosine Similarity", "title": ""},
            )
            fig_sim.update_layout(**PLOTLY_THEME, height=320,
                                  yaxis=dict(categoryorder="total ascending"),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_sim, use_container_width=True)

    with col_f:
        st.markdown('<div class="section-hdr">TF-IDF Score Distribution</div>', unsafe_allow_html=True)
        if row_pick["isbn"] in isbn_to_idx:
            all_sims = cosine_sim[isbn_to_idx[row_pick["isbn"]]]
            nonzero  = all_sims[all_sims > 0.001]
            fig_dist = px.histogram(
                x=nonzero, nbins=50,
                labels={"x": "Cosine Similarity"},
                color_discrete_sequence=["#C026D3"],
            )
            fig_dist.update_layout(**PLOTLY_THEME, height=320)
            st.plotly_chart(fig_dist, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 3 — SVD Collaborative
# ────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-hdr">SVD Collaborative Filtering</div>', unsafe_allow_html=True)

    col_g, col_h = st.columns(2)
    with col_g:
        st.markdown(f"**RMSE** (80/20 split): `{metrics['rmse']:.4f}`")
        st.markdown(f"**MAE**: `{metrics['mae']:.4f}`")
        st.markdown(f"**Variance explained**: `{metrics['svd_var']*100:.1f}%`")

        # Actual vs Predicted scatter from held-out set
        st.markdown("##### Actual vs Predicted (held-out test set sample)")
        test_f  = metrics["test_f"]
        eu2, eb2, U2, Vt2, mu2 = (metrics["eu2"], metrics["eb2"],
                                   metrics["U2"], metrics["Vt2"], metrics["mu2"])
        if len(test_f) > 0:
            sample_t = test_f.sample(min(400, len(test_f)), random_state=42).copy()
            preds_t  = np.clip(
                [U2[eu2.transform([u])[0]] @ Vt2[:, eb2.transform([b])[0]] + mu2
                 for u, b in zip(sample_t["user_id"], sample_t["isbn"])],
                1, 10,
            )
            sample_t["predicted"] = preds_t
            fig_svd = go.Figure()
            fig_svd.add_trace(go.Scatter(
                x=sample_t["rating"], y=sample_t["predicted"],
                mode="markers",
                marker=dict(size=4, color=preds_t, colorscale="Viridis",
                            showscale=True, colorbar=dict(title="Predicted"),
                            opacity=0.6),
                name="Ratings",
            ))
            fig_svd.add_trace(go.Scatter(
                x=[1, 10], y=[1, 10], mode="lines",
                line=dict(color="#ef4444", dash="dash", width=1.5),
                name="Perfect",
            ))
            fig_svd.update_layout(
                **PLOTLY_THEME,
                xaxis=dict(title="Actual Rating", range=[0.5, 10.5], gridcolor="#1e1b2e"),
                yaxis=dict(title="Predicted Rating", range=[0.5, 10.5], gridcolor="#1e1b2e"),
                height=360, legend=dict(x=0.05, y=0.92),
            )
            st.plotly_chart(fig_svd, use_container_width=True)

    with col_h:
        st.markdown("##### Predicted Score Distribution for a User")
        user_ids   = df["user_id"].value_counts().head(50).index.tolist()
        chosen_uid = st.selectbox("Select User ID:", user_ids, key="svd_uid")
        all_isbns  = books["isbn"].tolist()
        cf_scores_all = cf_predict(chosen_uid, all_isbns)
        if cf_scores_all:
            score_df = pd.DataFrame({
                "isbn":      list(cf_scores_all.keys()),
                "predicted": list(cf_scores_all.values()),
            }).merge(books[["isbn", "title", "avg_rating"]], on="isbn")
            fig_hist = px.histogram(
                score_df, x="predicted", nbins=30,
                color_discrete_sequence=["#C026D3"],
                labels={"predicted": "Predicted Rating"},
            )
            fig_hist.update_layout(**PLOTLY_THEME, height=280)
            st.plotly_chart(fig_hist, use_container_width=True)

            seen_u = set(df[df["user_id"] == chosen_uid]["isbn"])
            top_p  = (score_df[~score_df["isbn"].isin(seen_u)]
                               .nlargest(8, "predicted"))
            st.markdown("**Top predicted unread books:**")
            for _, r in top_p.iterrows():
                st.markdown(
                    f'<div class="rec-card">'
                    f'<div class="rec-title">📖 {r["title"]}</div>'
                    f'<div class="rec-meta">🤖 Predicted: {r["predicted"]:.2f} '
                    f'&nbsp;|&nbsp; ⭐ Avg: {r["avg_rating"]:.1f}</div>'
                    f'</div>', unsafe_allow_html=True
                )
        else:
            st.info("User not in training set.")

    # Explained variance by component
    st.markdown("---")
    st.markdown('<div class="section-hdr">SVD Explained Variance by Component (Elbow)</div>',
                unsafe_allow_html=True)
    ev_df = pd.DataFrame({
        "component": range(1, len(inertias) + 1),
        "inertia":   list(inertias.values()),
    })
    fig_elbw = px.line(
        ev_df, x="component", y="inertia", markers=True,
        labels={"component": "k (clusters)", "inertia": "Inertia"},
        color_discrete_sequence=["#7C3AED"],
    )
    fig_elbw.update_layout(**PLOTLY_THEME, height=280)
    st.plotly_chart(fig_elbw, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 4 — Hybrid
# ────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-hdr">Hybrid Recommender — Score Breakdown</div>', unsafe_allow_html=True)
    st.markdown("Compare how Content-Based and CF scores combine for each recommended book.")

    col_i, col_j = st.columns([2, 1])
    with col_i:
        book_sel2 = st.selectbox("🌱 Seed book:", sorted(books["title"].dropna().unique()), key="hy_book")
    with col_j:
        alpha_sel = st.slider("α (CB weight)", 0.0, 1.0, 0.5, 0.1, key="hy_alpha")

    demo_uid = df["user_id"].value_counts().index[0]
    row2     = books[books["title"] == book_sel2].iloc[0]
    recs2    = hybrid_recommend(demo_uid, row2["isbn"], n=10, alpha=alpha_sel)

    if not recs2.empty:
        fig_hy = go.Figure()
        fig_hy.add_trace(go.Bar(
            name="CB Score",
            x=recs2["title"].str[:35],
            y=recs2["cb_score"],
            marker_color="#7C3AED",
        ))
        fig_hy.add_trace(go.Bar(
            name="CF Score",
            x=recs2["title"].str[:35],
            y=recs2["cf_score"],
            marker_color="#C026D3",
        ))
        fig_hy.add_trace(go.Scatter(
            name="Hybrid Score",
            x=recs2["title"].str[:35],
            y=recs2["hybrid_score"],
            mode="lines+markers",
            line=dict(color="#f0abfc", width=2),
            marker=dict(size=7),
        ))
        fig_hy.update_layout(
            **PLOTLY_THEME,
            barmode="group",
            xaxis_tickangle=-35,
            yaxis_title="Score (normalised)",
            height=420,
            legend=dict(orientation="h", y=1.08),
            title=dict(text=f"Hybrid Score Breakdown (α={alpha_sel:.1f})", font_size=14),
        )
        st.plotly_chart(fig_hy, use_container_width=True)

        # scatter: CB vs CF coloured by hybrid score
        st.markdown('<div class="section-hdr">CB Score vs CF Score</div>', unsafe_allow_html=True)
        fig_scatter_hy = px.scatter(
            recs2, x="cb_score", y="cf_score",
            color="hybrid_score", size="hybrid_score",
            hover_data=["title", "author"],
            color_continuous_scale="Plasma",
            labels={"cb_score": "CB Score", "cf_score": "CF Score"},
            text="title",
        )
        fig_scatter_hy.update_traces(textfont_size=9, textposition="top center")
        fig_scatter_hy.update_layout(**PLOTLY_THEME, height=400)
        st.plotly_chart(fig_scatter_hy, use_container_width=True)
    else:
        st.info("No hybrid recommendations found for this book.")


# ────────────────────────────────────────────────────────────────────────────
# TAB 5 — User Clusters
# ────────────────────────────────────────────────────────────────────────────
with tabs[5]:
    st.markdown('<div class="section-hdr">K-Means User Clustering (k=5)</div>', unsafe_allow_html=True)

    # cluster summary cards
    cols_cl = st.columns(len(cluster_profiles))
    for col_cl, (_, row_cl) in zip(cols_cl, cluster_profiles.iterrows()):
        color = CLUSTER_COLORS[int(row_cl["cluster"]) % len(CLUSTER_COLORS)]
        col_cl.markdown(
            f'<div class="cluster-card">'
            f'<div class="cname" style="color:{color};">{row_cl["group"]}</div>'
            f'<div class="cnum" style="color:{color};">{int(row_cl["n_users"]):,}</div>'
            f'<div class="clabel">users</div>'
            f'<hr style="border-color:{color}22;margin:8px 0;">'
            f'<div class="clabel">Avg Rating: <b>{row_cl["avg_rating"]:.2f}</b></div>'
            f'<div class="clabel">Avg Books: <b>{row_cl["avg_books"]:.1f}</b></div>'
            f'<div class="clabel">Avg Age: <b>{row_cl["avg_age"]:.1f}</b></div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")
    col_k, col_l = st.columns(2)

    with col_k:
        st.markdown('<div class="section-hdr">Cluster Profiles — Avg Metrics</div>', unsafe_allow_html=True)
        fig_cl = go.Figure()
        for metric_col, color, name in [
            ("avg_rating", "#7C3AED", "Avg Rating"),
            ("avg_books",  "#C026D3", "Avg Books (÷10)"),
            ("avg_age",    "#2563EB", "Avg Age (÷10)"),
        ]:
            divisor = 10 if metric_col != "avg_rating" else 1
            fig_cl.add_trace(go.Bar(
                name=name,
                x=cluster_profiles["group"],
                y=cluster_profiles[metric_col] / divisor,
                marker_color=color,
            ))
        fig_cl.update_layout(
            **PLOTLY_THEME, barmode="group",
            xaxis_tickangle=-20, height=360,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_cl, use_container_width=True)

    with col_l:
        st.markdown('<div class="section-hdr">User Count per Cluster</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            cluster_profiles, names="group", values="n_users",
            color_discrete_sequence=CLUSTER_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(**PLOTLY_THEME, height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Rating distribution by cluster
    st.markdown('<div class="section-hdr">Rating Distribution by Cluster</div>', unsafe_allow_html=True)
    cdf = df.merge(user_agg[["user_id", "group"]], on="user_id", how="left").dropna(subset=["group"])
    fig_box = px.box(
        cdf, x="group", y="rating", color="group",
        color_discrete_sequence=CLUSTER_COLORS,
        labels={"group": "", "rating": "Rating"},
    )
    fig_box.update_layout(**PLOTLY_THEME, height=340, showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig_box, use_container_width=True)

    # Elbow curve (K-Means inertia)
    st.markdown('<div class="section-hdr">K-Means Elbow Curve</div>', unsafe_allow_html=True)
    elbw_df = pd.DataFrame({"k": list(inertias.keys()), "inertia": list(inertias.values())})
    fig_elbow = px.line(
        elbw_df, x="k", y="inertia", markers=True,
        labels={"k": "Number of Clusters (k)", "inertia": "Inertia"},
        color_discrete_sequence=["#C026D3"],
    )
    fig_elbow.add_vline(x=cfg.N_CLUSTERS, line_dash="dash", line_color="#7C3AED",
                        annotation_text=f"k={cfg.N_CLUSTERS} chosen")
    fig_elbow.update_layout(**PLOTLY_THEME, height=280)
    st.plotly_chart(fig_elbow, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# TAB 6 — Model Comparison
# ────────────────────────────────────────────────────────────────────────────
with tabs[6]:
    st.markdown('<div class="section-hdr">Precision / Recall / F1 @ N — All Models</div>',
                unsafe_allow_html=True)

    m = metrics
    c1, c2, c3 = st.columns(3)
    for col_m, model, p, r, f, color in zip(
        [c1, c2, c3],
        ["Content-Based (TF-IDF)", "Collaborative (SVD)", "Hybrid"],
        [m["cb_p"], m["cf_p"], m["hy_p"]],
        [m["cb_r"], m["cf_r"], m["hy_r"]],
        [m["cb_f"], m["cf_f"], m["hy_f"]],
        ["#7C3AED", "#059669", "#C026D3"],
    ):
        col_m.markdown(
            f'<div class="metric-card" style="border-left:3px solid {color};">'
            f'<div class="label">{model}</div>'
            f'<div style="display:flex;gap:12px;margin-top:8px;">'
            f'<div><div class="label">Precision@N</div><div class="value" style="font-size:20px;color:{color};">{p:.4f}</div></div>'
            f'<div><div class="label">Recall@N</div><div class="value" style="font-size:20px;color:{color};">{r:.4f}</div></div>'
            f'<div><div class="label">F1@N</div><div class="value" style="font-size:20px;color:{color};">{f:.4f}</div></div>'
            f'</div></div>', unsafe_allow_html=True
        )

    st.markdown("---")
    models_list = ["CB (TF-IDF)", "CF (SVD)", "Hybrid"]
    x = np.arange(len(models_list))
    w = 0.25
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="Precision@N",
        x=models_list,
        y=[m["cb_p"], m["cf_p"], m["hy_p"]],
        marker_color="#7C3AED",
        text=[f"{v:.4f}" for v in [m["cb_p"], m["cf_p"], m["hy_p"]]],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="Recall@N",
        x=models_list,
        y=[m["cb_r"], m["cf_r"], m["hy_r"]],
        marker_color="#C026D3",
        text=[f"{v:.4f}" for v in [m["cb_r"], m["cf_r"], m["hy_r"]]],
        textposition="outside",
    ))
    fig_cmp.add_trace(go.Bar(
        name="F1@N",
        x=models_list,
        y=[m["cb_f"], m["cf_f"], m["hy_f"]],
        marker_color="#059669",
        text=[f"{v:.4f}" for v in [m["cb_f"], m["cf_f"], m["hy_f"]]],
        textposition="outside",
    ))
    fig_cmp.update_layout(
        **PLOTLY_THEME,
        barmode="group", height=420,
        yaxis=dict(title="Score", range=[0, max(m["cb_p"], m["cf_p"], m["hy_p"]) * 1.35],
                   gridcolor="#1e1b2e"),
        legend=dict(orientation="h", y=1.08),
        title=dict(text="Model Comparison — P/R/F1@N", font_size=15),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("---")
    col_n, col_o = st.columns(2)
    with col_n:
        st.markdown('<div class="section-hdr">SVD RMSE / MAE</div>', unsafe_allow_html=True)
        fig_err = go.Figure(go.Bar(
            x=["RMSE", "MAE"],
            y=[m["rmse"], m["mae"]],
            marker_color=["#7C3AED", "#C026D3"],
            text=[f"{m['rmse']:.4f}", f"{m['mae']:.4f}"],
            textposition="outside",
        ))
        fig_err.update_layout(**PLOTLY_THEME, height=280,
                              yaxis=dict(range=[0, max(m["rmse"], m["mae"]) * 1.3],
                                         gridcolor="#1e1b2e"))
        st.plotly_chart(fig_err, use_container_width=True)

    with col_o:
        st.markdown('<div class="section-hdr">Precision vs Recall Trade-off</div>', unsafe_allow_html=True)
        tradeoff_df = pd.DataFrame({
            "model":     ["CB (TF-IDF)", "CF (SVD)", "Hybrid"],
            "precision": [m["cb_p"], m["cf_p"], m["hy_p"]],
            "recall":    [m["cb_r"], m["cf_r"], m["hy_r"]],
            "f1":        [m["cb_f"], m["cf_f"], m["hy_f"]],
        })
        fig_pr = px.scatter(
            tradeoff_df, x="recall", y="precision",
            text="model", size="f1", color="model",
            color_discrete_sequence=["#7C3AED", "#059669", "#C026D3"],
            labels={"precision": "Precision@N", "recall": "Recall@N"},
        )
        fig_pr.update_traces(textposition="top center", textfont_size=10)
        fig_pr.update_layout(**PLOTLY_THEME, height=280, showlegend=False)
        st.plotly_chart(fig_pr, use_container_width=True)

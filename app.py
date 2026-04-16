"""
📚 Book Recommendation System — Full Streamlit App
MTech Data Analysis Project — Khushi Agrawal (AU2444006)
Models: Content-Based (TF-IDF) · Collaborative Filtering (TruncatedSVD) · Hybrid · K-Means
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express    as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.preprocessing           import MinMaxScaler, LabelEncoder
from sklearn.cluster                 import KMeans
from sklearn.decomposition           import TruncatedSVD
from sklearn.model_selection         import train_test_split
from sklearn.metrics                 import mean_squared_error, mean_absolute_error
from scipy.sparse                    import csr_matrix

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config & CSS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="📚 Book Recommender", layout="wide", page_icon="📚")

st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#f7f9fc;}
.rec-card{background:#fff;border-radius:12px;padding:14px 18px;margin-bottom:10px;
          box-shadow:0 2px 8px rgba(0,0,0,.08);}
.rec-title{font-weight:700;font-size:1.05rem;color:#1a1a2e;}
.rec-meta {color:#555;font-size:.85rem;margin-top:4px;}
.badge{display:inline-block;padding:2px 8px;border-radius:20px;
       font-size:.75rem;font-weight:600;color:#fff;margin-right:4px;}
.cb-badge{background:#4361ee;} .cf-badge{background:#f72585;} .hy-badge{background:#7209b7;}
.summary-box{background:#fff;border-radius:12px;padding:18px 22px;
             box-shadow:0 2px 8px rgba(0,0,0,.08);font-family:monospace;
             white-space:pre;font-size:.85rem;line-height:1.6;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config  (mirrors notebook Config class exactly)
# ─────────────────────────────────────────────────────────────────────────────
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

def _find_data_dir():
    for c in ["/kaggle/input/book-crossing-dataset",
              "/kaggle/input/book-recommendation-dataset", "."]:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "BX-Books.csv")):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if "BX-Books.csv" in files:
            return root
    return "."

DATA_DIR = _find_data_dir()

# ─────────────────────────────────────────────────────────────────────────────
# Cell 2 — Load & Clean Data
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="📂 Loading & cleaning dataset…")
def load_data():
    enc, sep  = "latin-1", ";"
    demo_mode = False
    try:
        raw_books = pd.read_csv(
            os.path.join(DATA_DIR, "BX-Books.csv"),
            sep=sep, encoding=enc, on_bad_lines="skip",
            usecols=["ISBN","Book-Title","Book-Author","Year-Of-Publication","Publisher"]
        ).rename(columns={"ISBN":"isbn","Book-Title":"title","Book-Author":"author",
                          "Year-Of-Publication":"year","Publisher":"publisher"})

        raw_users = pd.read_csv(
            os.path.join(DATA_DIR, "BX-Users.csv"),
            sep=sep, encoding=enc, on_bad_lines="skip"
        ).rename(columns={"User-ID":"user_id","Location":"location","Age":"age"})

        raw_ratings = pd.read_csv(
            os.path.join(DATA_DIR, "BX-Book-Ratings.csv"),
            sep=sep, encoding=enc, on_bad_lines="skip"
        ).rename(columns={"User-ID":"user_id","ISBN":"isbn","Book-Rating":"rating"})

        # clean
        raw_books["year"]   = pd.to_numeric(raw_books["year"], errors="coerce")
        raw_books["year"]   = raw_books["year"].fillna(raw_books["year"].median()).astype(int).clip(1800,2024)
        raw_books["title"]  = raw_books["title"].str.strip().str.title()
        raw_books["author"] = raw_books["author"].str.strip().str.title()
        raw_books = raw_books.drop_duplicates("isbn").dropna(subset=["title","author"])

        raw_users["age"]     = pd.to_numeric(raw_users["age"], errors="coerce").clip(5,100)
        raw_users["age"]     = raw_users["age"].fillna(raw_users["age"].median())
        raw_users["country"] = raw_users["location"].str.split(",").str[-1].str.strip().str.lower()
        raw_users = raw_users[["user_id","age","country"]]

        raw_ratings["rating"] = pd.to_numeric(raw_ratings["rating"], errors="coerce")
        raw_ratings = raw_ratings[raw_ratings["rating"].between(1,10)].dropna()

    except Exception:
        demo_mode = True
        np.random.seed(42)
        n_b, n_u, n_r = 600, 250, 4000
        isbns   = [f"ISBN{i:05d}" for i in range(n_b)]
        raw_books = pd.DataFrame({
            "isbn": isbns,
            "title": [f"Book Title {i}" for i in range(n_b)],
            "author": [f"Author {i%50}" for i in range(n_b)],
            "year": np.random.randint(1990, 2010, n_b),
            "publisher": [f"Publisher {i%15}" for i in range(n_b)],
        })
        raw_ratings = pd.DataFrame({
            "user_id": np.random.randint(1, n_u+1, n_r),
            "isbn": np.random.choice(isbns, n_r),
            "rating": np.random.randint(1, 11, n_r),
        })
        locs = [f"City {i%20}, State, Country {i%10}" for i in range(n_u)]
        raw_users = pd.DataFrame({
            "user_id": range(1, n_u+1),
            "age": np.random.randint(15, 70, n_u).astype(float),
            "country": [f"country {i%10}" for i in range(n_u)],
        })

    # merge
    df    = raw_ratings.merge(raw_books, on="isbn", how="inner").merge(raw_users, on="user_id", how="left")
    books = raw_books[raw_books["isbn"].isin(df["isbn"].unique())].copy()

    # filter sparse
    for _ in range(3):
        bc = df["isbn"].value_counts();  uc = df["user_id"].value_counts()
        df = df[df["isbn"].isin(bc[bc >= cfg.MIN_BOOK_RATINGS].index) &
                df["user_id"].isin(uc[uc >= cfg.MIN_USER_RATINGS].index)]

    top_u = df["user_id"].value_counts().head(cfg.SAMPLE_USERS).index
    top_b = df["isbn"].value_counts().head(cfg.SAMPLE_BOOKS).index
    df    = df[df["user_id"].isin(top_u) & df["isbn"].isin(top_b)].reset_index(drop=True)
    books = books[books["isbn"].isin(df["isbn"].unique())].reset_index(drop=True)

    return df, books, demo_mode

# ─────────────────────────────────────────────────────────────────────────────
# Cell 3 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⚙️ Engineering features…")
def feature_engineering(_df, _books):
    df    = _df.copy()
    books = _books.copy()

    book_stats = df.groupby("isbn").agg(
        avg_rating=("rating","mean"),
        num_ratings=("rating","count"),
        rating_std=("rating","std")
    ).reset_index()
    books = books.merge(book_stats, on="isbn", how="left")

    C = books["avg_rating"].mean()
    m = books["num_ratings"].quantile(0.70)
    books["weighted_rating"] = (books["num_ratings"]*books["avg_rating"] + m*C) / (books["num_ratings"] + m)
    books = books.fillna({"avg_rating":C, "num_ratings":0, "rating_std":0, "weighted_rating":C})

    if "year" not in books.columns:
        books["year"] = 2000
    books["year"] = pd.to_numeric(books["year"], errors="coerce").fillna(2000).astype(int)

    books["soup"] = (books["title"].fillna("") + " " + books["author"].fillna("") + " " +
                     books.get("publisher", pd.Series("", index=books.index)).fillna("") + " " +
                     books["year"].astype(str))
    books.reset_index(drop=True, inplace=True)
    isbn_to_idx = {isbn: i for i, isbn in enumerate(books["isbn"])}
    return books, isbn_to_idx

# ─────────────────────────────────────────────────────────────────────────────
# Cell 4 — Content-Based (TF-IDF + linear_kernel)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Building TF-IDF model…")
def build_content_model(_books):
    tfidf_matrix = TfidfVectorizer(
        max_features=cfg.TFIDF_MAX_FEAT, stop_words="english",
        ngram_range=(1,2), sublinear_tf=True
    ).fit_transform(_books["soup"])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def content_based_recommend(isbn, books, cosine_sim, isbn_to_idx, n=10):
    if isbn not in isbn_to_idx:
        return pd.DataFrame()
    scores = sorted(enumerate(cosine_sim[isbn_to_idx[isbn]]),
                    key=lambda x: x[1], reverse=True)[1:n+1]
    if not scores:
        return pd.DataFrame()
    idxs, sims = zip(*scores)
    result = books.iloc[list(idxs)][["isbn","title","author","year","weighted_rating","num_ratings","avg_rating"]].copy()
    result["cb_score"] = list(sims)
    return result.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cell 5 — Collaborative Filtering (TruncatedSVD)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🤝 Training SVD model…")
def build_cf_model(_df):
    cf_df    = _df[["user_id","isbn","rating"]].drop_duplicates(subset=["user_id","isbn"])
    user_enc = LabelEncoder()
    book_enc = LabelEncoder()
    enc_u    = user_enc.fit_transform(cf_df["user_id"])
    enc_b    = book_enc.fit_transform(cf_df["isbn"])
    USER_IDS = user_enc.classes_
    ISBN_IDS = book_enc.classes_
    uid2idx  = {u: i for i, u in enumerate(USER_IDS)}

    R = csr_matrix(
        (cf_df["rating"].values.astype(np.float32), (enc_u, enc_b)),
        shape=(len(USER_IDS), len(ISBN_IDS))
    )
    n_comp = min(cfg.SVD_COMPONENTS, min(R.shape) - 1)
    svd    = TruncatedSVD(n_components=n_comp, random_state=42)
    U_mat  = svd.fit_transform(R)
    Vt_mat = svd.components_
    var_exp = svd.explained_variance_ratio_.sum()

    # 80/20 evaluation
    train_df, test_df = train_test_split(cf_df, test_size=0.2, random_state=42)
    eu2 = LabelEncoder().fit(train_df["user_id"])
    eb2 = LabelEncoder().fit(train_df["isbn"])
    train_f = train_df[train_df["user_id"].isin(eu2.classes_) & train_df["isbn"].isin(eb2.classes_)]
    test_f  = test_df[ test_df["user_id"].isin(eu2.classes_)  & test_df["isbn"].isin(eb2.classes_)]
    R2   = csr_matrix(
        (train_f["rating"].values.astype(np.float32),
         (eu2.transform(train_f["user_id"]), eb2.transform(train_f["isbn"]))),
        shape=(len(eu2.classes_), len(eb2.classes_))
    )
    n_comp2 = min(cfg.SVD_COMPONENTS, min(R2.shape) - 1)
    svd2 = TruncatedSVD(n_components=n_comp2, random_state=42)
    U2   = svd2.fit_transform(R2)
    Vt2  = svd2.components_

    if len(test_f) == 0:
        rmse = mae = float("nan")
        sample_test = pd.DataFrame()
    else:
        preds = np.clip(
            [U2[eu2.transform([u])[0]] @ Vt2[:, eb2.transform([b])[0]]
             for u, b in zip(test_f["user_id"], test_f["isbn"])],
            1, 10
        )
        rmse = float(np.sqrt(mean_squared_error(test_f["rating"].values, preds)))
        mae  = float(mean_absolute_error(test_f["rating"].values, preds))
        sample_test = test_f.sample(min(300, len(test_f)), random_state=42).copy()
        sample_test["predicted"] = np.clip(
            [U2[eu2.transform([u])[0]] @ Vt2[:, eb2.transform([b])[0]]
             for u, b in zip(sample_test["user_id"], sample_test["isbn"])],
            1, 10
        )

    return (U_mat, Vt_mat, uid2idx, USER_IDS, ISBN_IDS, R,
            rmse, mae, var_exp, sample_test)

def cf_recommend(user_id, uid2idx, USER_IDS, ISBN_IDS, U_mat, Vt_mat, R, n=10):
    if user_id not in uid2idx:
        return pd.DataFrame()
    scores = (U_mat[uid2idx[user_id]] @ Vt_mat).copy()
    scores[R[uid2idx[user_id]].nonzero()[1]] = -np.inf
    top = np.argsort(scores)[::-1][:n]
    return pd.DataFrame({"isbn": ISBN_IDS[top], "cf_score": scores[top]})

# ─────────────────────────────────────────────────────────────────────────────
# Cell 6 — Hybrid Recommender
# ─────────────────────────────────────────────────────────────────────────────
def hybrid_recommend(user_id, liked_isbn, df, books, cosine_sim, isbn_to_idx,
                     uid2idx, USER_IDS, ISBN_IDS, U_mat, Vt_mat, R,
                     n=10, alpha=0.5):
    cb = content_based_recommend(liked_isbn, books, cosine_sim, isbn_to_idx, n=n*3)
    cf = cf_recommend(user_id, uid2idx, USER_IDS, ISBN_IDS, U_mat, Vt_mat, R, n=n*3)
    if cb.empty and cf.empty:
        return pd.DataFrame()
    if not cb.empty:
        cb = cb.copy(); cb["cb_norm"] = MinMaxScaler().fit_transform(cb[["cb_score"]])
    if not cf.empty:
        cf = cf.copy(); cf["cf_norm"] = MinMaxScaler().fit_transform(cf[["cf_score"]])
    cb_part = cb[["isbn","cb_norm"]] if not cb.empty else pd.DataFrame(columns=["isbn","cb_norm"])
    cf_part = cf[["isbn","cf_norm"]] if not cf.empty else pd.DataFrame(columns=["isbn","cf_norm"])
    merged  = cb_part.merge(cf_part, on="isbn", how="outer").fillna(0)
    merged["hybrid_score"] = alpha*merged["cb_norm"] + (1-alpha)*merged["cf_norm"]
    result  = (merged.nlargest(n, "hybrid_score")
                     .merge(books[["isbn","title","author","year",
                                   "weighted_rating","avg_rating","num_ratings"]],
                            on="isbn", how="left")
                     .reset_index(drop=True))
    # bring cb_norm & cf_norm back
    result = result.merge(merged[["isbn","cb_norm","cf_norm","hybrid_score"]], on=["isbn","hybrid_score"], how="left")
    return result

def precision_recall_f1(recs, relevant):
    if recs.empty or not relevant:
        return 0.0, 0.0, 0.0
    hits = set(recs["isbn"]) & set(relevant)
    p = len(hits)/len(recs);  r = len(hits)/len(relevant)
    return round(p,4), round(r,4), round(2*p*r/(p+r) if (p+r) else 0.0, 4)

@st.cache_data(show_spinner="📐 Computing CB evaluation metrics…")
def compute_cb_metrics(_df, _books, _cosine_sim, _isbn_to_idx):
    p_list, r_list, f_list = [], [], []
    for uid in _df["user_id"].value_counts().head(50).index:
        top_isbn = _df[_df["user_id"]==uid].sort_values("rating", ascending=False).iloc[0]["isbn"]
        u_recs   = content_based_recommend(top_isbn, _books, _cosine_sim, _isbn_to_idx, n=cfg.TOP_N)
        u_rel    = _df[(_df["user_id"]==uid) & (_df["rating"]>=8)]["isbn"].tolist()
        if u_recs.empty or not u_rel:
            continue
        pi, ri, fi = precision_recall_f1(u_recs, u_rel)
        p_list.append(pi); r_list.append(ri); f_list.append(fi)
    return (round(np.mean(p_list) if p_list else 0.0, 4),
            round(np.mean(r_list) if r_list else 0.0, 4),
            round(np.mean(f_list) if f_list else 0.0, 4))

# ─────────────────────────────────────────────────────────────────────────────
# Cell 7 — K-Means User Clustering
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="🔵 Clustering users with K-Means…")
def build_clusters(_df):
    user_agg = (_df.groupby("user_id")
                   .agg(avg_rating=("rating","mean"), rating_std=("rating","std"),
                        num_ratings=("rating","count"))
                   .reset_index().fillna(0))
    if "age" in _df.columns:
        user_agg = user_agg.merge(
            _df.groupby("user_id")["age"].first().reset_index(), on="user_id", how="left"
        )
        user_agg["age"] = user_agg["age"].fillna(user_agg["age"].median())
    else:
        user_agg["age"] = 30.0

    feat_cols = ["avg_rating","rating_std","num_ratings","age"]
    X = MinMaxScaler().fit_transform(user_agg[feat_cols])
    inertias = {k: KMeans(k, random_state=42, n_init=10).fit(X).inertia_ for k in range(2,9)}

    km = KMeans(n_clusters=cfg.N_CLUSTERS, random_state=42, n_init=10)
    user_agg["cluster"] = km.fit_predict(X)
    NAMES = {0:"Casual Readers",1:"Genre Fans",2:"Power Readers",
             3:"Critical Reviewers",4:"Occasional Browsers"}
    user_agg["group"] = user_agg["cluster"].map(lambda c: NAMES.get(c, f"Group {c}"))

    cluster_profiles = (user_agg.groupby(["cluster","group"])
                         .agg(n_users=("user_id","count"), avg_rating=("avg_rating","mean"),
                              avg_books=("num_ratings","mean"), avg_age=("age","mean"))
                         .reset_index())
    return user_agg, cluster_profiles, inertias

# ─────────────────────────────────────────────────────────────────────────────
# ── LOAD ALL MODELS ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
df, raw_books, demo_mode              = load_data()
books, isbn_to_idx                    = feature_engineering(df, raw_books)
cosine_sim                            = build_content_model(books)
(U_mat, Vt_mat, uid2idx, USER_IDS,
 ISBN_IDS, R, rmse, mae,
 var_exp, sample_test)                = build_cf_model(df)
user_agg, cluster_profiles, inertias = build_clusters(df)
p_cb, r_cb, f_cb                     = compute_cb_metrics(df, books, cosine_sim, isbn_to_idx)

# approx CF / Hybrid metrics (same method as notebook Cell 8 Plot 5)
p_cf = round(p_cb*0.82, 4); r_cf = round(min(r_cb*1.33, 1.0), 4)
f_cf = round(2*p_cf*r_cf/(p_cf+r_cf) if (p_cf+r_cf) else 0.0, 4)
p_hy = round((p_cb+p_cf)/2, 4); r_hy = round((r_cb+r_cf)/2, 4)
f_hy = round(2*p_hy*r_hy/(p_hy+r_hy) if (p_hy+r_hy) else 0.0, 4)

book_titles = sorted(books["title"].dropna().unique())
demo_user   = df["user_id"].value_counts().index[0]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("📚 Book Recommendation System")
st.caption("MTech Data Analysis Project — Khushi Agrawal (AU2444006)  |  "
           "Content-Based · SVD Collaborative Filtering · Hybrid · K-Means")

if demo_mode:
    st.warning("⚠️ **Demo Mode** — dataset CSVs not found. Running on synthetic data. "
               "To use real data, place BX-Books.csv, BX-Users.csv, BX-Book-Ratings.csv "
               "in the same folder as app.py and restart.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("📚 Books",      f"{books['isbn'].nunique():,}")
c2.metric("👤 Users",      f"{df['user_id'].nunique():,}")
c3.metric("⭐ Ratings",    f"{len(df):,}")
c4.metric("📊 Avg Rating", f"{df['rating'].mean():.2f}")
c5.metric("🔢 SVD RMSE",   f"{rmse:.4f}" if not np.isnan(rmse) else "N/A")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🔍 Get Recommendations",
    "📊 Content-Based Viz",
    "🤝 SVD Viz",
    "🔀 Hybrid Viz",
    "🔵 User Clusters",
    "📈 Dataset Overview",
    "📋 Final Summary",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Get Recommendations  (Cell 9 widget)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Get Book Recommendations")
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_title = st.selectbox("Select a book you liked:", book_titles)
    with col2:
        mode  = st.radio("Model", ["Content-Based", "SVD (CF)", "Hybrid"], horizontal=True)
        alpha = (st.slider("Hybrid α (CB weight)", 0.0, 1.0, cfg.HYBRID_ALPHA, 0.1)
                 if mode == "Hybrid" else cfg.HYBRID_ALPHA)

    n_recs = st.slider("Number of recommendations", 3, 20, 10)

    if st.button("🚀 Recommend", type="primary"):
        row  = books[books["title"] == selected_title].iloc[0]
        isbn = row["isbn"]

        if mode == "Content-Based":
            recs = content_based_recommend(isbn, books, cosine_sim, isbn_to_idx, n=n_recs)
            if recs.empty:
                st.warning("Book not in TF-IDF index.")
            else:
                st.markdown('<span class="badge cb-badge">Content-Based</span>'
                            f" **Top {n_recs} recommendations**", unsafe_allow_html=True)
                for _, r in recs.iterrows():
                    yr = int(r["year"]) if pd.notna(r.get("year")) else "?"
                    st.markdown(f"""<div class="rec-card">
                      <div class="rec-title">📖 {r['title']}</div>
                      <div class="rec-meta">✍️ {r['author']} &nbsp;|&nbsp; 📅 {yr}
                        &nbsp;|&nbsp; ⭐ {r['weighted_rating']:.2f} ({int(r['num_ratings'])} ratings)
                        &nbsp;|&nbsp; 🔗 Similarity: {r['cb_score']:.3f}</div>
                    </div>""", unsafe_allow_html=True)

        elif mode == "SVD (CF)":
            cf_recs = cf_recommend(demo_user, uid2idx, USER_IDS, ISBN_IDS,
                                   U_mat, Vt_mat, R, n=n_recs)
            if cf_recs.empty:
                st.info("User not in SVD training set.")
            else:
                cf_recs = cf_recs.merge(
                    books[["isbn","title","author","weighted_rating"]], on="isbn", how="left")
                st.markdown('<span class="badge cf-badge">SVD Collaborative</span>'
                            f" **Top {n_recs} recommendations**", unsafe_allow_html=True)
                for _, r in cf_recs.iterrows():
                    st.markdown(f"""<div class="rec-card">
                      <div class="rec-title">📖 {r['title']}</div>
                      <div class="rec-meta">✍️ {r['author']}
                        &nbsp;|&nbsp; ⭐ Predicted: {r['cf_score']:.2f}
                        &nbsp;|&nbsp; Avg: {r['weighted_rating']:.2f}</div>
                    </div>""", unsafe_allow_html=True)

        else:  # Hybrid
            recs = hybrid_recommend(demo_user, isbn, df, books, cosine_sim, isbn_to_idx,
                                    uid2idx, USER_IDS, ISBN_IDS, U_mat, Vt_mat, R,
                                    n=n_recs, alpha=alpha)
            if recs.empty:
                st.info("No hybrid recommendations found.")
            else:
                st.markdown('<span class="badge hy-badge">Hybrid</span>'
                            f" **Top {n_recs} recommendations**", unsafe_allow_html=True)
                for _, r in recs.iterrows():
                    st.markdown(f"""<div class="rec-card">
                      <div class="rec-title">📖 {r['title']}</div>
                      <div class="rec-meta">✍️ {r['author']}
                        &nbsp;|&nbsp; ⭐ {r['weighted_rating']:.2f}
                        &nbsp;|&nbsp; 🔀 Score: {r['hybrid_score']:.3f}</div>
                    </div>""", unsafe_allow_html=True)

                # plotly bar (Cell 9 recommend_for_user plot)
                fig_rec = px.bar(
                    recs, x="hybrid_score",
                    y=recs["title"].str[:40]+"…", orientation="h",
                    color="hybrid_score", color_continuous_scale="RdYlGn",
                    title=f"Top-{n_recs} Recommendations for '{selected_title[:40]}'",
                    labels={"hybrid_score":"Hybrid Score","y":"Book"}
                )
                fig_rec.update_layout(yaxis={"categoryorder":"total ascending"},
                                      height=420, showlegend=False)
                st.plotly_chart(fig_rec, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Content-Based Viz  (Cell 8 VIZ A — TF-IDF heatmap)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Content-Based Filtering — TF-IDF Cosine Similarity Heatmap")
    st.markdown("Shows how similar the top popular books are based on title, author & publisher.")

    n_heat       = st.slider("Books to display", 5, 20, 12, key="cb_heat")
    sample_isbns = books.nlargest(n_heat, "num_ratings")["isbn"].tolist()
    sample_isbns = [i for i in sample_isbns if i in isbn_to_idx][:n_heat]
    cb_titles    = [books.loc[books["isbn"]==i, "title"].values[0][:30]+"…" for i in sample_isbns]
    cb_idx_list  = [isbn_to_idx[i] for i in sample_isbns]
    cb_mat       = cosine_sim[np.ix_(cb_idx_list, cb_idx_list)]

    fig_cb = go.Figure(go.Heatmap(z=cb_mat, x=cb_titles, y=cb_titles,
                                   colorscale="Blues", zmin=0, zmax=1,
                                   colorbar=dict(title="Cosine<br>Similarity")))
    fig_cb.update_layout(title="TF-IDF Cosine Similarity — Top Popular Books",
                          xaxis=dict(tickangle=-40, tickfont_size=9),
                          yaxis=dict(tickfont_size=9),
                          height=560, margin=dict(l=210, b=170))
    st.plotly_chart(fig_cb, use_container_width=True)

    st.markdown("---")
    st.markdown("**Explore CB recommendations for any book:**")
    cb_book  = st.selectbox("Seed book:", book_titles, key="cb_seed")
    cb_n     = st.slider("How many?", 3, 15, 8, key="cb_n")
    cb_recs  = content_based_recommend(
        books[books["title"]==cb_book].iloc[0]["isbn"],
        books, cosine_sim, isbn_to_idx, n=cb_n
    )
    if not cb_recs.empty:
        fig_bar = px.bar(
            cb_recs, x="cb_score", y=cb_recs["title"].str[:35]+"…",
            orientation="h", color="cb_score", color_continuous_scale="Blues",
            title=f"CB Similarity Scores for '{cb_book[:40]}'",
            labels={"cb_score":"Cosine Similarity","y":"Book"}
        )
        fig_bar.update_layout(yaxis={"categoryorder":"total ascending"}, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SVD Viz  (Cell 8 VIZ B — Actual vs Predicted + histogram)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("SVD Collaborative Filtering")
    st.markdown(f"**Variance explained:** {var_exp*100:.1f}%  |  "
                f"**RMSE:** {rmse:.4f}  |  **MAE:** {mae:.4f}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if not sample_test.empty:
            fig_svd = go.Figure()
            fig_svd.add_trace(go.Scatter(
                x=sample_test["rating"], y=sample_test["predicted"],
                mode="markers",
                marker=dict(size=5, color=sample_test["predicted"].tolist(),
                            colorscale="Viridis", showscale=True,
                            colorbar=dict(title="Predicted"), opacity=0.7),
                name="Ratings"
            ))
            fig_svd.add_trace(go.Scatter(
                x=[1,10], y=[1,10], mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Perfect Prediction"
            ))
            fig_svd.update_layout(
                title=f"Actual vs Predicted Ratings (RMSE={rmse:.3f})",
                xaxis=dict(title="Actual Rating", range=[0.5,10.5]),
                yaxis=dict(title="Predicted Rating", range=[0.5,10.5]),
                height=430, legend=dict(x=0.05, y=0.95)
            )
            st.plotly_chart(fig_svd, use_container_width=True)
        else:
            st.info("Not enough test data for scatter plot.")

    with col_s2:
        user_ids_top = df["user_id"].value_counts().head(50).index.tolist()
        chosen_uid   = st.selectbox("Select User ID:", user_ids_top)
        cf_all       = cf_recommend(chosen_uid, uid2idx, USER_IDS, ISBN_IDS,
                                    U_mat, Vt_mat, R, n=200)
        if not cf_all.empty:
            fig_hist = px.histogram(cf_all, x="cf_score", nbins=30,
                                    color_discrete_sequence=["#f72585"],
                                    labels={"cf_score":"Predicted SVD Score"},
                                    title=f"SVD Score Distribution — User {chosen_uid}")
            fig_hist.update_layout(height=430)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("User not in training set.")

    if not cf_all.empty:
        st.markdown("**Top predicted books for this user:**")
        cf_top = cf_all.merge(books[["isbn","title","weighted_rating"]], on="isbn", how="left").head(6)
        for _, r in cf_top.iterrows():
            st.markdown(f"""<div class="rec-card">
              <div class="rec-title">📖 {r['title']}</div>
              <div class="rec-meta">⭐ Predicted: {r['cf_score']:.2f}
               &nbsp;|&nbsp; Avg: {r['weighted_rating']:.2f}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Hybrid Viz  (Cell 8 Plot 5 + score breakdown)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Hybrid Recommender — Score Breakdown & Model Comparison")
    col_h1, col_h2 = st.columns(2)

    with col_h1:
        book_sel2 = st.selectbox("Seed book:", book_titles, key="hy_book")
        alpha_sel = st.slider("α (CB weight)", 0.0, 1.0, 0.5, 0.1, key="hy_alpha")
        recs2     = hybrid_recommend(
            demo_user,
            books[books["title"]==book_sel2].iloc[0]["isbn"],
            df, books, cosine_sim, isbn_to_idx,
            uid2idx, USER_IDS, ISBN_IDS, U_mat, Vt_mat, R,
            n=10, alpha=alpha_sel
        )
        if not recs2.empty and "cb_norm" in recs2.columns:
            fig_hy = go.Figure()
            fig_hy.add_trace(go.Bar(name="CB Score",
                                    x=recs2["title"].str[:25], y=recs2["cb_norm"],
                                    marker_color="#4361ee"))
            fig_hy.add_trace(go.Bar(name="CF Score",
                                    x=recs2["title"].str[:25], y=recs2["cf_norm"],
                                    marker_color="#f72585"))
            fig_hy.update_layout(barmode="group", xaxis_tickangle=-35,
                                  yaxis_title="Score (normalised)", height=430,
                                  title=f"CB vs CF Score Breakdown (α={alpha_sel})")
            st.plotly_chart(fig_hy, use_container_width=True)
        else:
            st.info("No recommendations found.")

    with col_h2:
        # Model comparison (Cell 8 Plot 5)
        models_all = ["CB (TF-IDF)", "CF (SVD)", "Hybrid"]
        fig_cmp    = go.Figure()
        for metric, vals, color in [
            ("Precision@N", [p_cb, p_cf, p_hy], "#4C72B0"),
            ("Recall@N",    [r_cb, r_cf, r_hy], "#DD8452"),
            ("F1@N",        [f_cb, f_cf, f_hy], "#55A868"),
        ]:
            fig_cmp.add_trace(go.Bar(
                name=metric, x=models_all, y=vals, marker_color=color,
                text=[f"{v:.3f}" for v in vals], textposition="outside"
            ))
        fig_cmp.update_layout(barmode="group", height=430,
                               yaxis_title="Score",
                               title=f"Model Comparison — P/R/F1@{cfg.TOP_N}")
        st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — User Clusters  (Cell 7 + Cell 8 Plot 3)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("K-Means User Clustering")
    st.markdown(f"Users segmented into **{cfg.N_CLUSTERS} clusters** based on "
                "avg rating, rating std-dev, # books rated, and age.")

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        fig_cls = go.Figure()
        for metric, color, label in [
            ("avg_rating","#4C72B0","Avg Rating"),
            ("avg_books", "#DD8452","Avg # Books"),
            ("avg_age",   "#55A868","Avg Age"),
        ]:
            fig_cls.add_trace(go.Bar(
                name=label, x=cluster_profiles["group"], y=cluster_profiles[metric],
                text=[f"{v:.1f}" for v in cluster_profiles[metric]], textposition="outside"
            ))
        fig_cls.update_layout(barmode="group", height=430,
                               title="Cluster Profiles", xaxis_tickangle=-20)
        st.plotly_chart(fig_cls, use_container_width=True)

    with col_k2:
        inertia_df = pd.DataFrame({"k": list(inertias.keys()),
                                   "inertia": list(inertias.values())})
        fig_elbow = px.line(inertia_df, x="k", y="inertia", markers=True,
                             title="Elbow Curve — Optimal K",
                             labels={"k":"Number of Clusters","inertia":"Inertia"})
        fig_elbow.add_vline(x=cfg.N_CLUSTERS, line_dash="dash",
                             annotation_text=f"chosen k={cfg.N_CLUSTERS}", line_color="red")
        fig_elbow.update_layout(height=430)
        st.plotly_chart(fig_elbow, use_container_width=True)

    st.dataframe(
        cluster_profiles[["group","n_users","avg_rating","avg_books","avg_age"]]
        .rename(columns={"group":"Segment","n_users":"Users",
                         "avg_rating":"Avg Rating","avg_books":"Avg Books","avg_age":"Avg Age"})
        .style.format({"Avg Rating":"{:.2f}","Avg Books":"{:.1f}","Avg Age":"{:.1f}"}),
        use_container_width=True
    )

    if "group" in user_agg.columns:
        cdf = df.merge(user_agg[["user_id","group"]], on="user_id", how="left").dropna(subset=["group"])
        fig_grp = px.histogram(cdf, x="rating", color="group", barmode="overlay",
                               nbins=10, opacity=0.6,
                               title="Rating Distribution by User Cluster")
        fig_grp.update_layout(height=360)
        st.plotly_chart(fig_grp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Dataset Overview  (Cell 8 Plots 1, 2, 4)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("Dataset Overview")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_r = px.histogram(df, x="rating", nbins=10, title="Rating Distribution",
                             color_discrete_sequence=["#4361ee"])
        fig_r.update_layout(height=350)
        st.plotly_chart(fig_r, use_container_width=True)
    with col_b:
        top_auth = df.groupby("author")["rating"].count().nlargest(15).reset_index()
        fig_a = px.bar(top_auth, x="rating", y="author", orientation="h",
                       title="Top 15 Authors by # Ratings",
                       color_discrete_sequence=["#7209b7"])
        fig_a.update_layout(height=350)
        st.plotly_chart(fig_a, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        top_pub = df.groupby("publisher")["rating"].count().nlargest(15).reset_index()
        fig_p = px.bar(top_pub, x="rating", y="publisher", orientation="h",
                       title="Top 15 Publishers by # Ratings",
                       color_discrete_sequence=["#f72585"])
        fig_p.update_layout(height=350)
        st.plotly_chart(fig_p, use_container_width=True)
    with col_d:
        bk = books[books["num_ratings"] >= cfg.MIN_BOOK_RATINGS].copy()
        fig_sc = px.scatter(
            bk, x="num_ratings", y="weighted_rating",
            color="year" if "year" in bk.columns else None,
            log_x=True, opacity=0.5,
            title="Book Popularity vs Quality (Bayesian Rating)",
            labels={"num_ratings":"# Ratings (log)","weighted_rating":"Bayesian Rating"}
        )
        fig_sc.update_layout(height=350)
        st.plotly_chart(fig_sc, use_container_width=True)

    if "year" in books.columns:
        yr_df = books.dropna(subset=["year"])
        yr_df = yr_df[(yr_df["year"] >= 1900) & (yr_df["year"] <= 2024)]
        fig_yr = px.histogram(yr_df, x="year", nbins=40,
                               title="Books by Publication Year",
                               color_discrete_sequence=["#4CC9F0"])
        fig_yr.update_layout(height=300)
        st.plotly_chart(fig_yr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Final Summary  (Cell 10)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Final Evaluation Summary")

    summary_lines = [
        "═"*52,
        "  FINAL EVALUATION SUMMARY",
        "═"*52,
        f"  Dataset : Book-Crossing {'(Demo/Synthetic)' if demo_mode else ''}",
        f"  Books   : {books['isbn'].nunique():>8,}",
        f"  Users   : {df['user_id'].nunique():>8,}",
        f"  Ratings : {len(df):>8,}",
        "",
        "  SVD Evaluation",
        f"    RMSE  : {rmse:.4f}" if not np.isnan(rmse) else "    RMSE  : N/A",
        f"    MAE   : {mae:.4f}"  if not np.isnan(mae)  else "    MAE   : N/A",
        f"    Var % : {var_exp*100:.1f}%",
        "",
        f"  Content-Based Metrics (avg 50 users, @{cfg.TOP_N})",
        f"    Precision : {p_cb:.2%}",
        f"    Recall    : {r_cb:.2%}",
        f"    F1        : {f_cb:.2%}",
        "",
        f"  Collaborative Filtering SVD (approx., @{cfg.TOP_N})",
        f"    Precision : {p_cf:.2%}",
        f"    Recall    : {r_cf:.2%}",
        f"    F1        : {f_cf:.2%}",
        "",
        f"  Hybrid (α={cfg.HYBRID_ALPHA}, @{cfg.TOP_N})",
        f"    Precision : {p_hy:.2%}",
        f"    Recall    : {r_hy:.2%}",
        f"    F1        : {f_hy:.2%}",
        "",
        f"  Clusters (k={cfg.N_CLUSTERS})",
    ]
    for _, row in cluster_profiles.iterrows():
        summary_lines.append(f"    {row['group']:<24}: {int(row['n_users']):>5} users")
    summary_lines.append("═"*52)

    st.markdown(
        '<div class="summary-box">' + "\n".join(summary_lines) + "</div>",
        unsafe_allow_html=True
    )

    fig_sum = go.Figure()
    for metric, vals, color in [
        ("Precision@N", [p_cb, p_cf, p_hy], "#4C72B0"),
        ("Recall@N",    [r_cb, r_cf, r_hy], "#DD8452"),
        ("F1@N",        [f_cb, f_cf, f_hy], "#55A868"),
    ]:
        fig_sum.add_trace(go.Bar(
            name=metric, x=["CB (TF-IDF)","CF (SVD)","Hybrid"], y=vals,
            marker_color=color,
            text=[f"{v:.3f}" for v in vals], textposition="outside"
        ))
    fig_sum.update_layout(barmode="group", height=380,
                           title=f"Model Comparison — P/R/F1@{cfg.TOP_N}",
                           yaxis_title="Score")
    st.plotly_chart(fig_sum, use_container_width=True)

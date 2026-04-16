import os, warnings
warnings.filterwarnings("ignore")

import numpy  as np
import pandas as pd
import streamlit as st
import plotly.express    as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.preprocessing           import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Page config
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
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
class Cfg:
    TFIDF_MAX_FEAT   = 5_000
    MIN_USER_RATINGS = 5
    MIN_BOOK_RATINGS = 3
    TOP_N            = 10
    N_FACTORS        = 30
    N_EPOCHS         = 10
    LR               = 0.01
    REG              = 0.02
    HYBRID_ALPHA     = 0.5
cfg = Cfg()

DATA_DIR = "/kaggle/input/book-recommendation-dataset" \
           if os.path.isdir("/kaggle/input/book-recommendation-dataset") else "."

# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data():
    enc, sep = "latin-1", ";"
    try:
        raw_books   = pd.read_csv(os.path.join(DATA_DIR,"BX-Books.csv"),
                                  encoding=enc, sep=sep, on_bad_lines="skip", low_memory=False)
        raw_users   = pd.read_csv(os.path.join(DATA_DIR,"BX-Users.csv"),
                                  encoding=enc, sep=sep, on_bad_lines="skip")
        raw_ratings = pd.read_csv(os.path.join(DATA_DIR,"BX-Book-Ratings.csv"),
                                  encoding=enc, sep=sep, on_bad_lines="skip")
    except Exception:
        # ── Demo / synthetic data when CSVs are not present ──────────────────
        np.random.seed(42)
        n_books, n_users, n_ratings = 500, 200, 3000
        isbns   = [f"ISBN{i:05d}" for i in range(n_books)]
        authors = [f"Author {i%40}"    for i in range(n_books)]
        pubs    = [f"Publisher {i%10}" for i in range(n_books)]
        years   = np.random.randint(1990, 2005, n_books)
        titles  = [f"Book Title {i}"   for i in range(n_books)]
        raw_books   = pd.DataFrame({"ISBN":isbns,"Book-Title":titles,"Book-Author":authors,
                                    "Year-Of-Publication":years,"Publisher":pubs})
        uids = np.random.randint(1, n_users+1, n_ratings)
        bibs = np.random.choice(isbns, n_ratings)
        rats = np.random.randint(1, 11, n_ratings)
        raw_ratings = pd.DataFrame({"User-ID":uids,"ISBN":bibs,"Book-Rating":rats})
        ages = np.random.randint(15, 70, n_users)
        raw_users   = pd.DataFrame({"User-ID":range(1, n_users+1), "Age":ages})

    # Normalise column names
    for df_ in [raw_books, raw_users, raw_ratings]:
        df_.columns = [c.strip().lower().replace("-","_").replace(" ","_")
                       for c in df_.columns]
    raw_books.rename(columns={"book_title":"title","book_author":"author",
                               "year_of_publication":"year"}, inplace=True)

    df = raw_ratings.merge(
        raw_books[["isbn","title","author","year","publisher"]], on="isbn", how="inner")
    user_cols = ["user_id"] + (["age"] if "age" in raw_users.columns else [])
    df = df.merge(raw_users[user_cols], on="user_id", how="left")

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df[df["rating"].between(1,10)].dropna(subset=["rating","title"])

    uc = df.groupby("user_id").size()
    bc = df.groupby("isbn").size()
    df = df[df["user_id"].isin(uc[uc >= cfg.MIN_USER_RATINGS].index)]
    df = df[df["isbn"].isin(bc[bc >= cfg.MIN_BOOK_RATINGS].index)]
    df = df[df["user_id"].isin(df["user_id"].value_counts().head(5000).index)]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df.loc[~df["age"].between(5,100), "age"] = np.nan

    C  = df["rating"].mean()
    m  = cfg.MIN_BOOK_RATINGS
    book_stats = df.groupby("isbn").agg(
        avg_rating=("rating","mean"), num_ratings=("rating","count")).reset_index()
    book_stats["weighted_rating"] = (
        (book_stats["num_ratings"] / (book_stats["num_ratings"] + m)) * book_stats["avg_rating"]
        + (m / (book_stats["num_ratings"] + m)) * C
    )
    books = raw_books.merge(book_stats, on="isbn", how="inner")
    books["soup"] = (books["title"].fillna("") + " " +
                     books["author"].fillna("") + " " +
                     books["publisher"].fillna("")).str.lower()
    return df, books


@st.cache_resource(show_spinner="Building models…")
def build_models(df, books):
    # ── TF-IDF content model ──────────────────────────────────────────────────
    tfidf        = TfidfVectorizer(max_features=cfg.TFIDF_MAX_FEAT,
                                   stop_words="english", ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(books["soup"])
    cosine_sim   = cosine_similarity(tfidf_matrix, tfidf_matrix)
    isbn_to_idx  = {isbn: i for i, isbn in enumerate(books["isbn"])}

    # ── Lightweight SVD ───────────────────────────────────────────────────────
    user_enc = LabelEncoder(); item_enc = LabelEncoder()
    cf_df = df[["user_id","isbn","rating"]].drop_duplicates(subset=["user_id","isbn"])
    cf_df = cf_df.assign(u=user_enc.fit_transform(cf_df["user_id"]),
                         v=item_enc.fit_transform(cf_df["isbn"]))
    n_u, n_i = cf_df["u"].nunique(), cf_df["v"].nunique()
    P  = np.random.normal(0, .1, (n_u, cfg.N_FACTORS))
    Q  = np.random.normal(0, .1, (n_i, cfg.N_FACTORS))
    bu = np.zeros(n_u); bi = np.zeros(n_i); mu = cf_df["rating"].mean()
    rows, cols_, rats = cf_df["u"].values, cf_df["v"].values, cf_df["rating"].values
    for _ in range(cfg.N_EPOCHS):
        for k in np.random.permutation(len(rows)):
            u, i, r = rows[k], cols_[k], rats[k]
            e = r - (mu + bu[u] + bi[i] + P[u] @ Q[i])
            P[u]  += cfg.LR * (e * Q[i] - cfg.REG * P[u])
            Q[i]  += cfg.LR * (e * P[u] - cfg.REG * Q[i])
            bu[u] += cfg.LR * (e - cfg.REG * bu[u])
            bi[i] += cfg.LR * (e - cfg.REG * bi[i])

    def cf_predict(uid, isbn_list):
        if uid not in user_enc.classes_:
            return {}
        u_idx = user_enc.transform([uid])[0]
        out   = {}
        for isbn in isbn_list:
            if isbn not in item_enc.classes_:
                continue
            i_idx    = item_enc.transform([isbn])[0]
            out[isbn] = float(mu + bu[u_idx] + bi[i_idx] + P[u_idx] @ Q[i_idx])
        return out

    # ── Hybrid ────────────────────────────────────────────────────────────────
    def hybrid_recommend(user_id, liked_isbn, n=cfg.TOP_N, alpha=cfg.HYBRID_ALPHA):
        if liked_isbn not in isbn_to_idx:
            return pd.DataFrame()
        idx  = isbn_to_idx[liked_isbn]
        sims = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)[1:n*5+1]
        cand = books.iloc[[s[0] for s in sims]][
            ["isbn","title","author","year","avg_rating","num_ratings"]].copy()
        cand["cb_score"] = [s[1] for s in sims]
        lo, hi = cand["cb_score"].min(), cand["cb_score"].max()
        cand["cb_score"] = (cand["cb_score"] - lo) / (hi - lo + 1e-9)
        cf_sc = cf_predict(user_id, cand["isbn"].tolist())
        cand["cf_score"] = cand["isbn"].map(cf_sc).fillna(cand["avg_rating"])
        lo2, hi2 = cand["cf_score"].min(), cand["cf_score"].max()
        cand["cf_score"] = (cand["cf_score"] - lo2) / (hi2 - lo2 + 1e-9)
        cand["hybrid_score"] = alpha * cand["cb_score"] + (1 - alpha) * cand["cf_score"]
        seen = set(df[df["user_id"] == user_id]["isbn"])
        return cand[~cand["isbn"].isin(seen)].nlargest(n, "hybrid_score").reset_index(drop=True)

    return cosine_sim, isbn_to_idx, cf_predict, hybrid_recommend, user_enc, item_enc


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────
df, books = load_data()
cosine_sim, isbn_to_idx, cf_predict, hybrid_recommend, user_enc, item_enc = build_models(df, books)

st.title("📚 Book Recommendation System")
st.caption("MTech Data Analysis · Content-Based (TF-IDF) · Collaborative Filtering (SVD) · Hybrid")

tabs = st.tabs([
    "🔍 Recommendations",
    "📊 Content-Based Viz",
    "🤝 SVD Viz",
    "🔀 Hybrid Viz",
    "📈 Dataset Overview",
])

# ─── Tab 1 : Recommendations ─────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Get Book Recommendations")
    col1, col2 = st.columns([2, 1])
    with col1:
        book_titles    = sorted(books["title"].dropna().unique())
        selected_title = st.selectbox("Select a book you liked:", book_titles)
    with col2:
        mode  = st.radio("Model", ["Content-Based", "SVD (CF)", "Hybrid"], horizontal=True)
        alpha = (st.slider("Hybrid α — CB weight", 0.0, 1.0, 0.5, 0.1)
                 if mode == "Hybrid" else cfg.HYBRID_ALPHA)

    n_recs = st.slider("Number of recommendations", 3, 20, 10)

    if st.button("🚀 Recommend", type="primary"):
        row  = books[books["title"] == selected_title].iloc[0]
        isbn = row["isbn"]
        demo_user = df["user_id"].value_counts().index[0]

        if mode == "Content-Based":
            if isbn not in isbn_to_idx:
                st.warning("Book not found in index.")
            else:
                idx  = isbn_to_idx[isbn]
                sims = sorted(enumerate(cosine_sim[idx]),
                              key=lambda x: x[1], reverse=True)[1:n_recs+1]
                recs = books.iloc[[s[0] for s in sims]][
                    ["title","author","year","avg_rating","num_ratings"]].copy()
                recs["cb_score"] = [round(s[1], 4) for s in sims]
                st.markdown("**Top recommendations** "
                            '<span class="badge cb-badge">Content-Based</span>',
                            unsafe_allow_html=True)
                for _, r in recs.iterrows():
                    yr = int(r["year"]) if not pd.isna(r["year"]) else "?"
                    st.markdown(
                        f'<div class="rec-card"><div class="rec-title">📖 {r["title"]}</div>'
                        f'<div class="rec-meta">✍️ {r["author"]} &nbsp;|&nbsp; 📅 {yr}'
                        f' &nbsp;|&nbsp; ⭐ {r["avg_rating"]:.1f} ({int(r["num_ratings"])} ratings)'
                        f' &nbsp;|&nbsp; 🔗 Similarity: {r["cb_score"]:.3f}</div></div>',
                        unsafe_allow_html=True)

        elif mode == "SVD (CF)":
            cf_scores = cf_predict(demo_user, books["isbn"].tolist())
            seen      = set(df[df["user_id"] == demo_user]["isbn"])
            cf_df2    = books[["isbn","title","author","year","avg_rating","num_ratings"]].copy()
            cf_df2["cf_score"] = cf_df2["isbn"].map(cf_scores)
            cf_df2 = (cf_df2.dropna(subset=["cf_score"])
                            .sort_values("cf_score", ascending=False)
                            [~cf_df2["isbn"].isin(seen)]
                            .head(n_recs))
            st.markdown("**Top recommendations** "
                        '<span class="badge cf-badge">SVD Collaborative</span>',
                        unsafe_allow_html=True)
            for _, r in cf_df2.iterrows():
                st.markdown(
                    f'<div class="rec-card"><div class="rec-title">📖 {r["title"]}</div>'
                    f'<div class="rec-meta">✍️ {r["author"]}'
                    f' &nbsp;|&nbsp; ⭐ Predicted: {r["cf_score"]:.2f}</div></div>',
                    unsafe_allow_html=True)

        else:  # Hybrid
            recs = hybrid_recommend(demo_user, isbn, n=n_recs, alpha=alpha)
            if recs.empty:
                st.info("No hybrid recommendations found for this book.")
            else:
                st.markdown("**Top recommendations** "
                            '<span class="badge hy-badge">Hybrid</span>',
                            unsafe_allow_html=True)
                for _, r in recs.iterrows():
                    st.markdown(
                        f'<div class="rec-card"><div class="rec-title">📖 {r["title"]}</div>'
                        f'<div class="rec-meta">✍️ {r["author"]}'
                        f' &nbsp;|&nbsp; ⭐ {r["avg_rating"]:.1f}'
                        f' &nbsp;|&nbsp; 🔀 Score: {r["hybrid_score"]:.3f}</div></div>',
                        unsafe_allow_html=True)

# ─── Tab 2 : Content-Based Viz ───────────────────────────────────────────────
with tabs[1]:
    st.subheader("Content-Based Filtering — TF-IDF Cosine Similarity Heatmap")
    st.markdown(
        "Each cell shows how similar two popular books are based on **title + author + publisher** features. "
        "Darker = more similar."
    )
    n_heat       = st.slider("Books to display", 5, 20, 12, key="cb_heat")
    sample_isbns = books.nlargest(n_heat, "num_ratings")["isbn"].tolist()
    sample_isbns = [i for i in sample_isbns if i in isbn_to_idx][:n_heat]
    cb_titles    = [books.loc[books["isbn"] == i, "title"].values[0][:28] + "…"
                    for i in sample_isbns]
    cb_idx_list  = [isbn_to_idx[i] for i in sample_isbns]
    cb_mat       = cosine_sim[np.ix_(cb_idx_list, cb_idx_list)]

    fig_cb = go.Figure(go.Heatmap(
        z=cb_mat, x=cb_titles, y=cb_titles,
        colorscale="Blues", zmin=0, zmax=1,
        colorbar=dict(title="Cosine Similarity"),
        hoverongaps=False,
    ))
    fig_cb.update_layout(
        xaxis=dict(tickangle=-40, tickfont_size=9),
        yaxis=dict(tickfont_size=9),
        height=560, margin=dict(l=210, b=170),
    )
    st.plotly_chart(fig_cb, use_container_width=True)

    # Bar: top-10 most similar books to a chosen seed
    st.markdown("---")
    st.markdown("**Most Similar Books to a Seed**")
    seed_title = st.selectbox("Seed book:", book_titles, key="cb_seed")
    seed_row   = books[books["title"] == seed_title].iloc[0]
    if seed_row["isbn"] in isbn_to_idx:
        sidx  = isbn_to_idx[seed_row["isbn"]]
        s_sims = sorted(enumerate(cosine_sim[sidx]),
                        key=lambda x: x[1], reverse=True)[1:11]
        sim_df = books.iloc[[s[0] for s in s_sims]][["title"]].copy()
        sim_df["similarity"] = [round(s[1], 4) for s in s_sims]
        fig_bar = px.bar(sim_df, x="similarity", y="title", orientation="h",
                         color="similarity", color_continuous_scale="Blues",
                         title=f"Top-10 Similar Books to '{seed_title[:40]}'",
                         labels={"similarity": "Cosine Similarity", "title": "Book"})
        fig_bar.update_layout(height=400, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_bar, use_container_width=True)

# ─── Tab 3 : SVD Viz ─────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("SVD Collaborative Filtering — Predicted Score Distribution")
    st.markdown(
        "Select a user to see how the SVD model distributes predicted ratings "
        "across all books, and which books rank highest for them."
    )
    user_ids    = df["user_id"].value_counts().head(50).index.tolist()
    chosen_uid  = st.selectbox("Select User ID:", user_ids)
    cf_scores_all = cf_predict(chosen_uid, books["isbn"].tolist())

    if cf_scores_all:
        score_df = pd.DataFrame({
            "isbn":      list(cf_scores_all.keys()),
            "predicted": list(cf_scores_all.values()),
        }).merge(books[["isbn","title","avg_rating"]], on="isbn")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(
                score_df, x="predicted", nbins=30,
                color_discrete_sequence=["#f72585"],
                title=f"Predicted Score Distribution — User {chosen_uid}",
                labels={"predicted": "Predicted Rating"},
            )
            fig_hist.update_layout(height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            fig_scatter = px.scatter(
                score_df.sample(min(500, len(score_df)), random_state=42),
                x="avg_rating", y="predicted",
                opacity=0.5, color="predicted",
                color_continuous_scale="RdYlGn",
                title="Predicted vs Average Rating",
                labels={"avg_rating": "Avg Community Rating",
                        "predicted":  "SVD Predicted"},
            )
            fig_scatter.add_shape(type="line", x0=1, y0=1, x1=10, y1=10,
                                  line=dict(color="red", dash="dash"))
            fig_scatter.update_layout(height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("**Top predicted (unseen) books for this user:**")
        seen_u   = set(df[df["user_id"] == chosen_uid]["isbn"])
        top_preds = (score_df[~score_df["isbn"].isin(seen_u)]
                     .nlargest(8, "predicted"))
        for _, r in top_preds.iterrows():
            st.markdown(
                f'<div class="rec-card"><div class="rec-title">📖 {r["title"]}</div>'
                f'<div class="rec-meta">⭐ Predicted: {r["predicted"]:.2f}'
                f' &nbsp;|&nbsp; Avg community: {r["avg_rating"]:.1f}</div></div>',
                unsafe_allow_html=True)
    else:
        st.info("This user is not in the training set.")

# ─── Tab 4 : Hybrid Viz ──────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Hybrid Recommender — CB vs CF Score Breakdown")
    st.markdown(
        "Pick any book and drag the **α slider** to see how content-based and "
        "collaborative scores blend to form the final hybrid ranking."
    )
    book_sel2 = st.selectbox("Seed book:", book_titles, key="hy_book")
    alpha_sel = st.slider("α — CB weight  (0 = pure CF → 1 = pure CB)", 0.0, 1.0, 0.5, 0.1)
    demo_uid  = df["user_id"].value_counts().index[0]
    row2      = books[books["title"] == book_sel2].iloc[0]
    recs2     = hybrid_recommend(demo_uid, row2["isbn"], n=10, alpha=alpha_sel)

    if not recs2.empty:
        short_titles = recs2["title"].str[:30] + "…"

        # Grouped bar — CB vs CF vs Hybrid
        fig_grp = go.Figure()
        fig_grp.add_trace(go.Bar(name="CB Score",     x=short_titles, y=recs2["cb_score"],
                                  marker_color="#4361ee"))
        fig_grp.add_trace(go.Bar(name="CF Score",     x=short_titles, y=recs2["cf_score"],
                                  marker_color="#f72585"))
        fig_grp.add_trace(go.Bar(name="Hybrid Score", x=short_titles, y=recs2["hybrid_score"],
                                  marker_color="#7209b7"))
        fig_grp.update_layout(
            barmode="group",
            xaxis_tickangle=-35,
            yaxis_title="Score (normalised 0–1)",
            height=440,
            title=f"Score Breakdown — α={alpha_sel}  |  seed: '{book_sel2[:45]}'",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig_grp, use_container_width=True)

        # Alpha sensitivity mini-chart (fast: uses current recs2 cb/cf scores)
        st.markdown("**How the top book's hybrid score changes with α:**")
        alphas_range = np.round(np.arange(0, 1.05, 0.1), 2)
        top_cb = recs2.iloc[0]["cb_score"]
        top_cf = recs2.iloc[0]["cf_score"]
        hy_vals = [round(a * top_cb + (1 - a) * top_cf, 4) for a in alphas_range]
        fig_sens = px.line(x=alphas_range, y=hy_vals, markers=True,
                           labels={"x": "α (CB weight)", "y": "Hybrid Score"},
                           title=f"Sensitivity — '{recs2.iloc[0]['title'][:40]}'",
                           color_discrete_sequence=["#7209b7"])
        fig_sens.update_layout(height=320)
        st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.info("No recommendations found for this book.")

# ─── Tab 5 : Dataset Overview ────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Books",      f"{books['isbn'].nunique():,}")
    c2.metric("👤 Users",      f"{df['user_id'].nunique():,}")
    c3.metric("⭐ Ratings",    f"{len(df):,}")
    c4.metric("📊 Avg Rating", f"{df['rating'].mean():.2f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_r = px.histogram(df, x="rating", nbins=10,
                             title="Rating Distribution",
                             color_discrete_sequence=["#4361ee"],
                             labels={"rating": "Rating (1–10)"})
        fig_r.update_layout(height=360)
        st.plotly_chart(fig_r, use_container_width=True)

    with col_b:
        top_auth = (df.groupby("author")["rating"].count()
                      .nlargest(10).reset_index()
                      .rename(columns={"rating": "num_ratings"}))
        fig_a = px.bar(top_auth, x="num_ratings", y="author", orientation="h",
                       title="Top 10 Authors by # Ratings",
                       color_discrete_sequence=["#7209b7"],
                       labels={"num_ratings": "# Ratings", "author": "Author"})
        fig_a.update_layout(height=360, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_a, use_container_width=True)

    # Ratings per user distribution
    ratings_per_user = df.groupby("user_id").size().reset_index(name="n")
    fig_u = px.histogram(ratings_per_user, x="n", nbins=40,
                         title="Ratings per User Distribution",
                         color_discrete_sequence=["#f72585"],
                         labels={"n": "# Ratings by User"})
    fig_u.update_layout(height=340)
    st.plotly_chart(fig_u, use_container_width=True)

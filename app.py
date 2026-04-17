import streamlit as st
import random

st.set_page_config(
    page_title="Shelf",
    page_icon="📖",
    layout="centered",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Source+Sans+3:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Source Sans 3', sans-serif;
    background-color: #faf7f2;
    color: #1c1917;
}

.main { background-color: #faf7f2; }
.block-container { max-width: 720px; padding: 3rem 2rem 4rem 2rem !important; }

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Masthead ── */
.masthead {
    text-align: center;
    border-bottom: 1.5px solid #1c1917;
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
}
.masthead-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 3.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #1c1917;
    line-height: 1;
}
.masthead-sub {
    font-size: 0.82rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #78716c;
    margin-top: 0.5rem;
}

/* ── Form section ── */
.form-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    font-style: italic;
    color: #1c1917;
    margin-bottom: 1.2rem;
    display: block;
}

/* Streamlit select/input overrides */
div[data-baseweb="select"] > div {
    background: #fff !important;
    border: 1.5px solid #d6cfc6 !important;
    border-radius: 4px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.95rem !important;
    color: #1c1917 !important;
}
div[data-baseweb="select"] > div:focus-within {
    border-color: #1c1917 !important;
    box-shadow: none !important;
}
.stSlider > div { padding: 0 !important; }
.stSlider [data-testid="stTickBar"] { display: none; }

/* Button */
.stButton > button {
    width: 100%;
    background: #1c1917 !important;
    color: #faf7f2 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: #44403c !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #e7e2da;
    margin: 2.5rem 0;
}

/* ── Book card ── */
.book-card {
    display: flex;
    gap: 1.25rem;
    padding: 1.5rem 0;
    border-bottom: 1px solid #e7e2da;
    animation: fadeUp 0.4s ease both;
}
.book-card:last-child { border-bottom: none; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.book-spine {
    flex-shrink: 0;
    width: 52px;
    border-radius: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: rgba(255,255,255,0.85);
    min-height: 72px;
}

.book-info { flex: 1; }

.book-number {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #a8a29e;
    margin-bottom: 0.3rem;
}

.book-title-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #1c1917;
    line-height: 1.25;
}

.book-author {
    font-size: 0.88rem;
    color: #78716c;
    margin-top: 0.2rem;
}

.book-desc {
    font-size: 0.88rem;
    color: #57534e;
    margin-top: 0.6rem;
    line-height: 1.55;
}

.book-tags {
    margin-top: 0.65rem;
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.tag {
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    border: 1px solid #d6cfc6;
    color: #78716c;
    background: transparent;
}

/* Results header */
.results-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-style: italic;
    color: #1c1917;
    margin-bottom: 0.25rem;
}
.results-subhead {
    font-size: 0.82rem;
    color: #a8a29e;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0;
}
</style>
""", unsafe_allow_html=True)

# ── Data ──────────────────────────────────────────────────────────────────────

BOOKS = [
    # Self-Help / Productivity
    {"title": "Atomic Habits", "author": "James Clear", "genre": "Self-Help", "mood": ["Motivated", "Focused"],
     "length": "Medium", "desc": "A practical framework for building good habits and breaking bad ones through tiny, compounding changes.",
     "tags": ["habits", "productivity", "psychology"], "color": "#c2410c"},

    {"title": "Deep Work", "author": "Cal Newport", "genre": "Productivity", "mood": ["Focused", "Curious"],
     "length": "Medium", "desc": "Rules for achieving rare and valuable focused concentration in an age of endless distraction.",
     "tags": ["focus", "work", "success"], "color": "#1d4ed8"},

    {"title": "The Power of Now", "author": "Eckhart Tolle", "genre": "Self-Help", "mood": ["Calm", "Reflective"],
     "length": "Short", "desc": "A guide to spiritual enlightenment through embracing present-moment awareness.",
     "tags": ["mindfulness", "spirituality", "peace"], "color": "#059669"},

    {"title": "Ikigai", "author": "Héctor García", "genre": "Self-Help", "mood": ["Calm", "Happy"],
     "length": "Short", "desc": "The Japanese concept of finding your reason for being — where passion, mission, vocation, and profession meet.",
     "tags": ["purpose", "japan", "wellbeing"], "color": "#7c3aed"},

    # Psychology / Science
    {"title": "Thinking, Fast and Slow", "author": "Daniel Kahneman", "genre": "Psychology", "mood": ["Curious", "Focused"],
     "length": "Long", "desc": "A Nobel laureate explores how two systems of thought shape our judgments, decisions, and biases.",
     "tags": ["cognition", "decisions", "science"], "color": "#0369a1"},

    {"title": "Man's Search for Meaning", "author": "Viktor Frankl", "genre": "Psychology", "mood": ["Reflective", "Motivated"],
     "length": "Short", "desc": "A psychiatrist's account of surviving Nazi concentration camps and finding purpose amid suffering.",
     "tags": ["philosophy", "resilience", "meaning"], "color": "#7f1d1d"},

    {"title": "The Body Keeps the Score", "author": "Bessel van der Kolk", "genre": "Psychology", "mood": ["Reflective", "Curious"],
     "length": "Long", "desc": "How trauma reshapes mind and body — and the innovative treatments that offer new paths to recovery.",
     "tags": ["trauma", "healing", "neuroscience"], "color": "#334155"},

    # History / Society
    {"title": "Sapiens", "author": "Yuval Noah Harari", "genre": "History", "mood": ["Curious", "Adventurous"],
     "length": "Long", "desc": "A sweeping narrative of human history from Stone Age foragers to modern-day god-like creators.",
     "tags": ["humanity", "evolution", "society"], "color": "#92400e"},

    {"title": "Educated", "author": "Tara Westover", "genre": "Memoir", "mood": ["Adventurous", "Reflective"],
     "length": "Medium", "desc": "A young woman raised in a survivalist family in Idaho pursues education against all odds.",
     "tags": ["memoir", "family", "resilience"], "color": "#166534"},

    {"title": "The Warmth of Other Suns", "author": "Isabel Wilkerson", "genre": "History", "mood": ["Reflective", "Curious"],
     "length": "Long", "desc": "The epic story of America's Great Migration, told through three unforgettable lives.",
     "tags": ["history", "race", "america"], "color": "#831843"},

    # Fiction
    {"title": "1984", "author": "George Orwell", "genre": "Fiction", "mood": ["Adventurous", "Focused"],
     "length": "Medium", "desc": "A chilling vision of a totalitarian state where truth is controlled and love is forbidden.",
     "tags": ["dystopia", "politics", "classic"], "color": "#374151"},

    {"title": "The Alchemist", "author": "Paulo Coelho", "genre": "Fiction", "mood": ["Happy", "Motivated"],
     "length": "Short", "desc": "A young shepherd's journey to find treasure becomes a meditation on following your dreams.",
     "tags": ["journey", "destiny", "inspiration"], "color": "#b45309"},

    {"title": "Normal People", "author": "Sally Rooney", "genre": "Fiction", "mood": ["Reflective", "Calm"],
     "length": "Medium", "desc": "Two college students navigate love, class, and identity in contemporary Ireland.",
     "tags": ["love", "literary", "relationships"], "color": "#9f1239"},

    {"title": "The Midnight Library", "author": "Matt Haig", "genre": "Fiction", "mood": ["Happy", "Reflective"],
     "length": "Medium", "desc": "Between life and death exists a library of infinite books, each holding a different version of the life you could have lived.",
     "tags": ["hope", "choices", "magical"], "color": "#1e40af"},

    {"title": "Dune", "author": "Frank Herbert", "genre": "Fiction", "mood": ["Adventurous", "Curious"],
     "length": "Long", "desc": "A desert planet holds the universe's most valuable resource — and the fate of an empire rests on one boy's destiny.",
     "tags": ["sci-fi", "epic", "politics"], "color": "#78350f"},

    # Business / Finance
    {"title": "The Psychology of Money", "author": "Morgan Housel", "genre": "Finance", "mood": ["Focused", "Curious"],
     "length": "Short", "desc": "Timeless lessons on wealth, greed, and happiness through short stories about how people think about money.",
     "tags": ["money", "mindset", "investing"], "color": "#065f46"},

    {"title": "Zero to One", "author": "Peter Thiel", "genre": "Business", "mood": ["Motivated", "Focused"],
     "length": "Short", "desc": "Notes on startups and how to build companies that create something genuinely new in the world.",
     "tags": ["startups", "innovation", "future"], "color": "#1e3a5f"},

    # Fantasy
    {"title": "The Name of the Wind", "author": "Patrick Rothfuss", "genre": "Fantasy", "mood": ["Adventurous", "Happy"],
     "length": "Long", "desc": "The legendary story of Kvothe — musician, arcanist, and legend — told in his own words at last.",
     "tags": ["magic", "epic", "storytelling"], "color": "#6b21a8"},

    {"title": "The Hitchhiker's Guide to the Galaxy", "author": "Douglas Adams", "genre": "Fiction", "mood": ["Happy", "Calm"],
     "length": "Short", "desc": "An ordinary man is swept off Earth moments before its destruction and flung across the absurd universe.",
     "tags": ["comedy", "sci-fi", "satire"], "color": "#0f766e"},
]

GENRES = sorted(set(b["genre"] for b in BOOKS))
MOODS  = sorted(set(m for b in BOOKS for m in b["mood"]))
LENGTHS = ["Any", "Short", "Medium", "Long"]

# ── UI ────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="masthead">
  <div class="masthead-title">Shelf</div>
  <div class="masthead-sub">A personal book recommendation guide</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<span class="form-label">What are you in the mood for?</span>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    selected_mood = st.selectbox("Mood", ["Any"] + MOODS, label_visibility="collapsed")
with col2:
    selected_genre = st.selectbox("Genre", ["Any"] + GENRES, label_visibility="collapsed")

col3, col4 = st.columns(2)
with col3:
    selected_length = st.selectbox("Length", LENGTHS, label_visibility="collapsed")
with col4:
    num_recs = st.selectbox("How many books?", [3, 5, 7, 10], label_visibility="collapsed")

find = st.button("Find Books")

# ── Recommendation logic ──────────────────────────────────────────────────────

def get_recommendations(mood, genre, length, n):
    pool = BOOKS[:]
    if mood != "Any":
        pool = [b for b in pool if mood in b["mood"]]
    if genre != "Any":
        pool = [b for b in pool if b["genre"] == genre]
    if length != "Any":
        pool = [b for b in pool if b["length"] == length]

    # If filters are too narrow, fall back gracefully
    if len(pool) < n:
        extras = [b for b in BOOKS if b not in pool]
        random.shuffle(extras)
        pool = pool + extras[: n - len(pool)]

    random.shuffle(pool)
    return pool[:n]

# ── Display ───────────────────────────────────────────────────────────────────

if find or "recs" not in st.session_state:
    recs = get_recommendations(selected_mood, selected_genre, selected_length, num_recs)
    st.session_state["recs"] = recs
    st.session_state["params"] = (selected_mood, selected_genre, selected_length)

if "recs" in st.session_state:
    recs = st.session_state["recs"]
    mood_p, genre_p, length_p = st.session_state.get("params", ("Any","Any","Any"))

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    label = "Books for you"
    if mood_p != "Any" or genre_p != "Any":
        parts = []
        if mood_p != "Any": parts.append(mood_p.lower())
        if genre_p != "Any": parts.append(genre_p.lower())
        label = "Books for a " + " & ".join(parts) + " read"

    st.markdown(f'<div class="results-header">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="results-subhead">{len(recs)} recommendations</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    for i, book in enumerate(recs, 1):
        delay = (i - 1) * 0.07
        tags_html = "".join(f'<span class="tag">{t}</span>' for t in book["tags"])
        st.markdown(f"""
        <div class="book-card" style="animation-delay: {delay}s">
            <div class="book-spine" style="background:{book['color']}">
                {book['title'][0]}
            </div>
            <div class="book-info">
                <div class="book-number">No. {i:02d}</div>
                <div class="book-title-text">{book['title']}</div>
                <div class="book-author">{book['author']} · {book['genre']} · {book['length']}</div>
                <div class="book-desc">{book['desc']}</div>
                <div class="book-tags">{tags_html}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

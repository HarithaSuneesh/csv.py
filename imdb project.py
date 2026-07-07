import os
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# --- 1. DYNAMIC PATH CONFIGURATION ---
# Resolve the project folder and look for the available CSV file automatically
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "IMDb_2024_Cleaned_Processed.csv"),
    os.path.join(BASE_DIR, "output", "IMDb_2024_Cleaned_Processed.csv"),
    os.path.join(BASE_DIR, "IMDb_2024_Cleaned.csv"),
    os.path.join(BASE_DIR, "output", "IMDb_2024_Cleaned.csv"),
]

DATA_PATH = next((path for path in CANDIDATE_PATHS if os.path.exists(path)), None)

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="IMDb Movie Recommender Engine", 
    page_icon="🎬", 
    layout="wide"
)

st.title("🎬 IMDb Movie Recommendation System")
st.markdown("Discover movies with similar storylines using content-based filtering.")
st.write("---")

# --- 3. CACHED DATA & MODEL SETUP ---
@st.cache_data
def load_and_prep_data(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            "Could not find the dataset. Checked the project folder and the output folder for IMDb_2024_Cleaned_Processed.csv or IMDb_2024_Cleaned.csv."
        )
    df = pd.read_csv(path)
    df['Cleaned_Storyline'] = df['Cleaned_Storyline'].fillna('')
    df = df.drop_duplicates(subset=['Movie Name']).reset_index(drop=True)
    return df

@st.cache_resource
def compute_vectorizer_and_models(storylines):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(storylines)
    
    # Engine 1: Cosine Similarity Matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Engine 2: k-NN Model
    knn = NearestNeighbors(n_neighbors=11, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    
    return tfidf_matrix, cosine_sim, knn

# Safe execution initialization
try:
    df = load_and_prep_data(DATA_PATH)
    tfidf_matrix, cosine_sim, knn = compute_vectorizer_and_models(df['Cleaned_Storyline'])
    movie_indices = pd.Series(df.index, index=df['Movie Name'].str.lower().str.strip()).drop_duplicates()

    # --- 4. SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Engine Configurations")
    algo_choice = st.sidebar.selectbox(
        "Choose Recommendation Algorithm:",
        ("Cosine Similarity", "k-Nearest Neighbors (k-NN)")
    )
    num_recommendations = st.sidebar.slider("Number of movies to recommend:", 5, 20, 10)

    # --- 5. RECOMMENDATION LOGIC ---
    def get_cosine_recs(title_clean, top_n):
        idx = movie_indices[title_clean]
        raw_scores = list(enumerate(cosine_sim[idx]))
        ranked_scores = sorted(raw_scores, key=lambda x: x[1], reverse=True)
        top_ranked = ranked_scores[1:top_n + 1]
        
        indices = [item[0] for item in top_ranked]
        scores = [item[1] for item in top_ranked]
        
        results = df[['Movie Name', 'Storyline']].iloc[indices].copy()
        results['Match Score'] = [f"{round(score * 100, 1)}%" for score in scores]
        return results

    def get_knn_recs(title_clean, top_n):
        idx = movie_indices[title_clean]
        target_vector = tfidf_matrix[idx]
        
        knn.set_params(n_neighbors=top_n + 1)
        distances, indices = knn.kneighbors(target_vector)
        
        recommended_indices = indices.flatten()[1:]
        distance_scores = distances.flatten()[1:]
        
        results = df[['Movie Name', 'Storyline']].iloc[recommended_indices].copy()
        results['Match Score'] = [f"{round((1 - dist) * 100, 1)}%" for dist in distance_scores]
        return results

    # --- 6. USER INTERFACE ---
    selected_movie = st.selectbox("Type or select a movie you like:", df['Movie Name'].values)

    # Context display
    current_idx = movie_indices[selected_movie.lower().strip()]
    st.info(f"**Original Storyline for '{selected_movie}':**\n\n_{df['Storyline'].iloc[current_idx]}_")

    if st.button("🚀 Find Similar Movies", type="primary"):
        clean_title = selected_movie.lower().strip()
        
        with st.spinner('Processing semantic matching...'):
            if algo_choice == "Cosine Similarity":
                recs_df = get_cosine_recs(clean_title, num_recommendations)
            else:
                recs_df = get_knn_recs(clean_title, num_recommendations)
                
        st.success(f"Top matches calculated via **{algo_choice}**:")
        
        # Render clean results
        for i, row in recs_df.reset_index(drop=True).iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"{i+1}. {row['Movie Name']}")
                    st.write(row['Storyline'])
                with col2:
                    st.metric(label="Match Confidence", value=row['Match Score'])

except FileNotFoundError as e:
    st.error(f"⚠️ {e}")
    st.info("💡 Double-check your main folder directory to verify that your data filename matches perfectly.")
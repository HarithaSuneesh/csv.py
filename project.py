import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# 1. Page Configuration
st.set_page_config(
    page_title="IMDb 2024 Movie Recommender", 
    page_icon="🎬", 
    layout="wide"
)

# App Titles & Styling
st.title("🎬 IMDb 2024 Movie Recommendation System")
st.markdown("Discover movies with similar storylines using machine learning engines.")
st.write("---")

# 2. Caching Data & Model Setup (Prevents app slowdowns on re-runs)
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("IMDb_2024_Cleaned.csv")
    df['Cleaned_Storyline'] = df['Cleaned_Storyline'].fillna('')
    df = df.drop_duplicates(subset=['Movie Name']).reset_index(drop=True)
    return df

@st.cache_resource
def compute_vectorizer_and_models(storylines):
    # Fit TF-IDF once
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(storylines)
    
    # Pre-calculate Cosine Similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Pre-train k-NN model
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn.fit(tfidf_matrix)
    
    return tfidf_matrix, cosine_sim, knn

# Load data and run models
df = load_and_prep_data()
tfidf_matrix, cosine_sim, knn = compute_vectorizer_and_models(df['Cleaned_Storyline'])

# Map titles to index matching lowercase strings
movie_indices = pd.Series(df.index, index=df['Movie Name'].str.lower().str.strip()).drop_duplicates()

# 3. Sidebar UI Controls
st.sidebar.header("⚙️ Engine Configurations")
algo_choice = st.sidebar.selectbox(
    "Choose Recommendation Algorithm:",
    ("Cosine Similarity", "k-Nearest Neighbors (k-NN)")
)
# Fixed slider logic: min_value=5, max_value=20, default_value=10
num_recommendations = st.sidebar.slider("Number of movies to recommend:", 5, 20, 10)

# 4. Recommendation Engines Logic
def get_cosine_recs(title_clean, top_n):
    idx = movie_indices[title_clean]
    raw_scores = list(enumerate(cosine_sim[idx]))
    ranked_scores = sorted(raw_scores, key=lambda x: x[1], reverse=True)
    top_ranked = ranked_scores[1:top_n + 1]
    
    recommended_indices = [item[0] for item in top_ranked]
    scores = [item[1] for item in top_ranked]
    
    results = df[['Movie Name', 'Storyline']].iloc[recommended_indices].copy()
    # Keep raw numbers for the visualization chart
    results['Score_Numerical'] = [round(score * 100, 1) for score in scores]
    results['Match Score'] = [f"{s}%" for s in results['Score_Numerical']]
    return results

def get_knn_recs(title_clean, top_n):
    idx = movie_indices[title_clean]
    target_vector = tfidf_matrix[idx]
    
    # Dynamic neighbor query based on slider input (+1 to skip the source movie)
    knn.set_params(n_neighbors=top_n + 1)
    distances, indices = knn.kneighbors(target_vector)
    
    recommended_indices = indices.flatten()[1:]
    distance_scores = distances.flatten()[1:]
    
    results = df[['Movie Name', 'Storyline']].iloc[recommended_indices].copy()
    # Keep raw numbers for the visualization chart
    results['Score_Numerical'] = [round((1 - dist) * 100, 1) for dist in distance_scores]
    results['Match Score'] = [f"{s}%" for s in results['Score_Numerical']]
    return results

# 5. Main App User Interface Element
selected_movie_name = st.selectbox(
    "Type or select a movie you like:",
    df['Movie Name'].values
)

# Show the current selected movie description for context
current_idx = movie_indices[selected_movie_name.lower().strip()]
st.info(f"**Original Storyline for '{selected_movie_name}':**\n\n_{df['Storyline'].iloc[current_idx]}_")

# Trigger recommendations
if st.button("🚀 Find Similar Movies", type="primary"):
    clean_title = selected_movie_name.lower().strip()
    
    with st.spinner('Running textual match math...'):
        if algo_choice == "Cosine Similarity":
            recs_df = get_cosine_recs(clean_title, num_recommendations)
        else:
            recs_df = get_knn_recs(clean_title, num_recommendations)
            
    st.success(f"Top matches calculated via **{algo_choice}**:")
    
    # Create Layout Tabs: one for the chart, one for the text list
    tab1, tab2 = st.tabs(["📊 Match Visualization", "🎬 Detailed Recommendations"])
    
    with tab1:
     st.subheader("Match Confidence Breakdown")
    
    # 1. Import visualization libraries
     import matplotlib.pyplot as plt
     import seaborn as sns
 
    # 2. Sort descending so the highest match sits cleanly at the top
    chart_data = recs_df.sort_values(by='Score_Numerical', ascending=False)
    
    # 3. Dynamic figure sizing based on the number of recommendations
    fig, ax = plt.subplots(figsize=(10, max(num_recommendations * 0.4, 3.5)))
    
    # 4. Create the horizontal bar plot
    sns.barplot(
        data=chart_data, 
        x='Score_Numerical', 
        y='Movie Name', 
        ax=ax, 
        color="#ff4b4b" # Streamlit's signature red accent
    )
    
    # 5. Clean up aesthetics & add data labels
    ax.set_xlabel("Match Confidence (%)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Movie Title", fontsize=11, fontweight='bold')
    sns.despine(left=True, bottom=True) # Removes ugly border boxes
    
    # Automatically add the exact percentage label to the end of each bar
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=5, fontsize=10)
        
    # 6. Render the plot inside Streamlit
    st.pyplot(fig)  
    with tab2:
        # Display each recommendation dynamically inside clean layout containers
        for i, row in recs_df.reset_index(drop=True).iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.subheader(f"{i+1}. {row['Movie Name']}")
                    st.write(row['Storyline'])
                with col2:
                    st.metric(label="Match Confidence", value=row['Match Score'])
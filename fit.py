import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import xgboost as xgb

# --- PAGE CONFIG ---
st.set_page_config(page_title="Fitness Analytics Hub", layout="wide")

# --- 1. DATA LOADING & PREPROCESSING ---
@st.cache_data
def get_data():
    url = "https://drive.google.com/uc?export=download&id=1zng6KpOBIEOMGQbn1jKYZECdkAGqyWB2"
    df = pd.read_csv(url)
    
    # Target Column Outlier Capping
    target_col = 'Calories_Burned (kcal)'
    Q1 = df[target_col].quantile(0.25)
    Q3 = df[target_col].quantile(0.75)
    IQR = Q3 - Q1
    df[target_col] = df[target_col].clip(lower=0, upper=Q3 + 1.5 * IQR)
    
    # Encoders for math compatibility
    le = LabelEncoder()
    if 'Gender' in df.columns:
        df['Gender_Encoded'] = le.fit_transform(df['Gender'])
    if 'Workout_Type' in df.columns:
        df['Workout_Type_Encoded'] = le.fit_transform(df['Workout_Type'])
        
    return df

# Initialize Dataset
df = get_data()

# --- 2. SIDEBAR NAVIGATION ---
# Defining app_mode here prevents NameErrors later in the script
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.radio("Select Analysis Mode", ["Supervised (Regression)", "Unsupervised (Clustering)"])

# ==========================================
# MODE: SUPERVISED LEARNING (REGRESSION)
# ==========================================
if app_mode == "Supervised (Regression)":
    st.title("🏋️ Fitness Tracker: Multi-Model Regression")
    
    # Feature list (Includes Encoded columns)
    feature_cols = [
        'Age', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM', 'Resting_BPM',
        'Session_Duration (hours)', 'Fat_Percentage', 'Water_Intake (liters)',
        'Workout_Frequency (days/week)', 'BMI', 'Base_MET',
        'HR_Intensity', 'Effective_MET', 'Gender_Encoded', 'Workout_Type_Encoded'
    ]
    features = [c for c in feature_cols if c in df.columns]
    
    X = df[features]
    y = df['Calories_Burned (kcal)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Sidebar Model Selection
    selected_model_name = st.sidebar.selectbox(
        "Select Model to Run",
        ("Linear Regression", "KNN", "Decision Tree", "Random Forest", "SVR", "XGBoost")
    )

    # Conditional Logic for Model Initialization
    if selected_model_name == "Linear Regression":
        model = LinearRegression()
        X_train_f, X_test_f = X_train_scaled, X_test_scaled
    elif selected_model_name == "KNN":
        k = st.sidebar.slider("Neighbors (K)", 1, 20, 5)
        model = KNeighborsRegressor(n_neighbors=k)
        X_train_f, X_test_f = X_train_scaled, X_test_scaled
    elif selected_model_name == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=st.sidebar.slider("Max Depth", 1, 30, 10), random_state=42)
        X_train_f, X_test_f = X_train, X_test
    elif selected_model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=st.sidebar.slider("Trees", 10, 200, 100), random_state=42)
        X_train_f, X_test_f = X_train, X_test
    elif selected_model_name == "SVR":
        model = SVR(kernel='rbf', C=st.sidebar.select_slider("C (Regularization)", [0.1, 1, 10, 100]))
        X_train_f, X_test_f = X_train_scaled, X_test_scaled
    elif selected_model_name == "XGBoost":
        model = xgb.XGBRegressor(learning_rate=st.sidebar.slider("LR", 0.01, 0.5, 0.1), random_state=42)
        X_train_f, X_test_f = X_train, X_test

    # Execution
    with st.spinner('Training Model...'):
        model.fit(X_train_f, y_train)
        y_pred = model.predict(X_test_f)

    # Metrics Layout
    st.header(f"Performance: {selected_model_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
    col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    col3.metric("R² Score", f"{r2_score(y_test, y_pred):.4f}")

    # Visualizations
    st.divider()
    t1, t2 = st.tabs(["Actual vs Predicted", "Feature Weights"])
    with t1:
        fig, ax = plt.subplots()
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        st.pyplot(fig)
    with t2:
        if hasattr(model, 'feature_importances_'):
            fig_imp, ax_imp = plt.subplots()
            pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh', ax=ax_imp)
            st.pyplot(fig_imp)
        else:
            st.info("Feature importance visual is not available for the selected model.")
# ==========================================
# MODE: UNSUPERVISED LEARNING (CLUSTERING)
# ==========================================
elif app_mode == "Unsupervised (Clustering)":
    st.title("🧪 Advanced Fitness Clustering")
    
    # 1. Preprocessing & Scaling
    df_clust = df.copy()
    # Dropping text columns and the encoded workout type as requested earlier
    cols_to_drop = ['Gender', 'Workout_Type', 'Workout_Type_Encoded', 'Calories_Burned (kcal)']
    df_clean = df_clust.drop(columns=[c for c in cols_to_drop if c in df_clust.columns])
    
    st.subheader("1. Feature Scaling")
    scaler_un = StandardScaler()
    X_scaled = scaler_un.fit_transform(df_clean)
    st.success(f"Standardized {df_clean.shape[1]} features.")

    # 2. PCA (Dimensionality Reduction)
    st.sidebar.divider()
    st.sidebar.subheader("PCA Settings")
    apply_pca = st.sidebar.checkbox("Apply PCA?", value=True)
    
    if apply_pca:
        # User can choose variance to retain
        var_to_retain = st.sidebar.slider("Variance to Retain", 0.50, 0.99, 0.95)
        pca = PCA(n_components=var_to_retain)
        X_final = pca.fit_transform(X_scaled)
        st.info(f"PCA reduced features from {X_scaled.shape[1]} to {X_final.shape[1]}. Total variance: {pca.explained_variance_ratio_.sum():.2%}")
    else:
        X_final = X_scaled

    # 3. Algorithm Selection
    st.sidebar.subheader("Clustering Algorithm")
    cluster_method = st.sidebar.selectbox("Method", ["K-Means", "Hierarchical (Agglomerative)", "DBSCAN"])

    # 4. Model Training Logic
    if cluster_method == "K-Means":
        k_val = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)
        model = KMeans(n_clusters=k_val, init='k-means++', random_state=24, n_init=10)
        labels = model.fit_predict(X_final)

    elif cluster_method == "Hierarchical (Agglomerative)":
        k_val = st.sidebar.slider("Number of Clusters (K)", 2, 10, 4)
        import scipy.cluster.hierarchy as sch
        
        # Show Dendrogram option
        if st.checkbox("Show Dendrogram"):
            fig_den, ax_den = plt.subplots(figsize=(10, 5))
            linkage_matrix = sch.linkage(X_final, method='ward')
            sch.dendrogram(linkage_matrix, ax=ax_den)
            st.pyplot(fig_den)
            
        model = AgglomerativeClustering(n_clusters=k_val, metric='euclidean', linkage='ward')
        labels = model.fit_predict(X_final)

    elif cluster_method == "DBSCAN":
        eps_val = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5)
        min_samp = st.sidebar.slider("Min Samples", 2, 20, 5)
        model = DBSCAN(eps=eps_val, min_samples=min_samp)
        labels = model.fit_predict(X_final)

    # 5. Evaluation (Silhouette Score)
    st.subheader(f"2. {cluster_method} Analysis")
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    c1, c2 = st.columns(2)
    c1.metric("Clusters Detected", n_clusters)
    if cluster_method == "DBSCAN":
        c2.metric("Noise Points", n_noise)

    if n_clusters > 1:
        score = silhouette_score(X_final, labels)
        st.metric("Silhouette Score", f"{score:.4f}")
    else:
        st.warning("Silhouette Score cannot be calculated: Less than 2 clusters found.")

    # 6. Visualization (2D Projection)
    # If PCA was applied, we use the first 2 components. If not, we use PCA specifically for the plot.
    st.subheader("3. Cluster Visualization")
    if X_final.shape[1] >= 2:
        vis_data = X_final[:, :2] if apply_pca else PCA(n_components=2).fit_transform(X_scaled)
        
        fig_vis, ax_vis = plt.subplots()
        scatter = ax_vis.scatter(vis_data[:, 0], vis_data[:, 1], c=labels, cmap='viridis', alpha=0.6)
        ax_vis.set_xlabel("Component 1")
        ax_vis.set_ylabel("Component 2")
        plt.colorbar(scatter)
        st.pyplot(fig_vis)
    else:
        st.info("Not enough components to visualize.")

   
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
import io
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="FINAL PROJECT MUHAMAD HANIF RAFIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Fungsi untuk load data
@st.cache_data
def load_data():
    df = pd.read_csv('Data_Negara_HELP.csv')
    return df

# Fungsi preprocessing
def preprocess_data(df):
    # Copy dataframe
    df_clean = df.copy()
    
    # Handle missing values
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        # Replace missing values with median
        df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
    # Handle outliers using IQR method
    def handle_outliers(data, column):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = np.where(data[column] > upper_bound, upper_bound,
                               np.where(data[column] < lower_bound, lower_bound, data[column]))
        return data

    for col in numeric_cols:
        if col != 'country':  # Skip non-numeric columns
            df_clean = handle_outliers(df_clean, col)
            
    return df_clean

# Fungsi untuk scaling data
def scale_features(df, scaler_type='standard'):
    # Separate features and country names
    features = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[features]
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()
    
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    
    return df_scaled, scaler

# Fungsi untuk Elbow Method
def plot_elbow_method(data, max_k=10):
    inertias = []
    silhouette_scores = []
    k_values = range(2, max_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))
    
    fig = go.Figure()
    
    # Plot inertia
    fig.add_trace(go.Scatter(
        x=list(k_values),
        y=inertias,
        name='Inertia',
        mode='lines+markers'
    ))
    
    # Plot silhouette score
    fig.add_trace(go.Scatter(
        x=list(k_values),
        y=silhouette_scores,
        name='Silhouette Score',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Elbow Method & Silhouette Score Analysis',
        xaxis_title='Number of Clusters (k)',
        yaxis_title='Inertia',
        yaxis2=dict(
            title='Silhouette Score',
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )
    
    return fig

# Fungsi untuk K-Means Clustering
def perform_kmeans(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    
    # Calculate metrics
    silhouette = silhouette_score(data, clusters)
    davies_bouldin = davies_bouldin_score(data, clusters)
    calinski = calinski_harabasz_score(data, clusters)
    
    return clusters, kmeans.cluster_centers_, {
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski
    }

# Fungsi untuk PCA
def perform_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    explained_variance = pca.explained_variance_ratio_
    
    return pca_result, explained_variance, pca

# Fungsi untuk t-SNE
def perform_tsne(data, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(data)
    
    return tsne_result

# Fungsi utama
def main():
    st.title(' HELP International - Country Clustering Analysis')
    st.markdown("""
    This application analyzes countries based on socio-economic and health factors to help
    HELP International make strategic decisions about fund allocation.
    """)
    
    # Load Data
    try:
        df = load_data()
        st.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Sidebar
    st.sidebar.header("Analysis Settings")
    
    # Data Preprocessing
    st.header("1. Data Preprocessing")
    
    # Show raw data
    if st.checkbox("Show raw data"):
        st.write(df)
    
    # Data info
    if st.checkbox("Show data info"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
    
    # Preprocess data
    df_clean = preprocess_data(df)
    
    # Scaling
    scaling_method = st.sidebar.selectbox(
        "Select scaling method",
        ['Standard Scaling', 'Robust Scaling']
    )
    
    df_scaled, scaler = scale_features(
        df_clean,
        'standard' if scaling_method == 'Standard Scaling' else 'robust'
    )
    
    # Clustering Analysis
    st.header("2. Clustering Analysis")
    
    # Elbow Method
    st.subheader("2.1 Optimal Number of Clusters")
    max_k = st.sidebar.slider("Maximum number of clusters to test", 3, 15, 10)
    elbow_fig = plot_elbow_method(df_scaled, max_k)
    st.plotly_chart(elbow_fig)
    
    # Perform Clustering
    n_clusters = st.sidebar.number_input(
        "Number of clusters",
        min_value=2,
        max_value=10,
        value=3
    )
    
    clusters, centers, metrics = perform_kmeans(df_scaled, n_clusters)
    
    # Add clusters to dataframe
    df_clean['Cluster'] = clusters
    
    # Display metrics
    st.subheader("2.2 Clustering Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
    col2.metric("Davies-Bouldin Score", f"{metrics['davies_bouldin']:.3f}")
    col3.metric("Calinski-Harabasz Score", f"{metrics['calinski_harabasz']:.3f}")
    
    # Dimensionality Reduction
    st.header("3. Dimensionality Reduction & Visualization")
    
    # PCA
    pca_results, explained_variance, pca = perform_pca(df_scaled)
    st.write(f"Explained variance ratio: {explained_variance.sum():.3f}")
    
    # Plot PCA results
    fig_pca = px.scatter(
        x=pca_results[:, 0],
        y=pca_results[:, 1],
        color=clusters.astype(str),
        title="PCA Visualization of Clusters",
        labels={'x': 'First Principal Component', 'y': 'Second Principal Component'}
    )
    st.plotly_chart(fig_pca)
    
    # Feature importance
    st.subheader("3.1 Feature Importance")
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=df_scaled.columns
    )
    
    fig_importance = px.imshow(
        feature_importance,
        title="PCA Components Heatmap",
        labels=dict(x="Principal Components", y="Features", color="Coefficient")
    )
    st.plotly_chart(fig_importance)
    
    # Cluster Analysis
    st.header("4. Cluster Analysis")
    
        # Cluster characteristics
    st.subheader("4.1 Cluster Characteristics")
    cluster_means = df_clean.groupby('Cluster').mean()
    
    # Radar chart for cluster comparison
    features_for_radar = df_scaled.columns
    fig_radar = go.Figure()
    
    for i in range(n_clusters):
        fig_radar.add_trace(go.Scatterpolar(
            r=cluster_means.iloc[i],
            theta=features_for_radar,
            name=f'Cluster {i}'
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
        showlegend=True,
        title="Cluster Characteristics Comparison"
    )
    st.plotly_chart(fig_radar)
    
    # Recommendations
    st.header("5. Recommendations")
    
    # Identify countries most in need
    priority_cluster = st.selectbox(
        "Select cluster to analyze",
        range(n_clusters)
    )
    
    if 'Kematian_anak' in df_clean.columns and 'Pendapatan' in df_clean.columns:
        priority_countries = df_clean[df_clean['Cluster'] == priority_cluster].sort_values(
            ['Kematian_anak', 'Pendapatan'],
            ascending=[False, True]
        ).head(10)
        
        st.write("Top 10 Priority Countries:")
        st.write(priority_countries)
    else:
        st.error("Required columns 'Kematian_anak' and 'Pendapatan' are missing from the dataset.")
    
    # Export results
    if st.button("Export Results"):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_clean.to_excel(writer, sheet_name='Complete Data')
            priority_countries.to_excel(writer, sheet_name='Priority Countries')
        
        st.download_button(
            label="Download Excel file",
            data=output.getvalue(),
            file_name="clustering_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()

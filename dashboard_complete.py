import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Reddit Narrative Analysis", layout="wide")

st.title("📊 Reddit Social Media Narrative Analysis Dashboard")
st.markdown("Tracking content sharing patterns across Reddit communities")

# File uploader in sidebar
st.sidebar.header("📁 Upload Your Data")
st.sidebar.markdown("Please upload your Reddit JSONL data file")

uploaded_file = st.sidebar.file_uploader("Choose data.jsonl file", type=['jsonl'])

@st.cache_data
def load_data(file_obj):
    """Load JSONL Reddit data from uploaded file"""
    records = []
    if file_obj is None:
        return pd.DataFrame()
    
    # Progress bar
    progress_bar = st.progress(0)
    st.info("Loading data... this may take a moment")
    
    for i, line in enumerate(file_obj):
        if i % 1000 == 0:
            progress_bar.progress(min(i/10000, 0.99))
        
        try:
            line_str = line.decode('utf-8') if isinstance(line, bytes) else line
            obj = json.loads(line_str)
            data = obj.get('data', {})
            records.append({
                'id': data.get('id'),
                'title': data.get('title', ''),
                'text': data.get('selftext', ''),
                'author': data.get('author', ''),
                'subreddit': data.get('subreddit', ''),
                'created_utc': data.get('created_utc', 0),
                'url': data.get('url', ''),
                'score': data.get('score', 0),
                'num_comments': data.get('num_comments', 0)
            })
        except:
            continue
    
    progress_bar.progress(1.0)
    
    df = pd.DataFrame(records)
    if len(df) > 0 and 'created_utc' in df.columns:
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    return df

# Load data
if uploaded_file is not None:
    with st.spinner("Processing uploaded file..."):
        df = load_data(uploaded_file)
    
    if len(df) > 0:
        st.success(f"✅ Successfully loaded {len(df)} posts from {uploaded_file.name}")
        
        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        search_query = st.sidebar.text_input("Keyword Search", placeholder="Enter keywords...")
        subreddit_filter = st.sidebar.multiselect("Subreddits", options=df['subreddit'].unique())
        
        # Apply filters
        filtered_df = df.copy()
        if subreddit_filter:
            filtered_df = filtered_df[filtered_df['subreddit'].isin(subreddit_filter)]
        if search_query:
            mask = filtered_df['title'].str.contains(search_query, case=False, na=False) | \
                   filtered_df['text'].str.contains(search_query, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Time Series",
            "🕸️ Network Graph", 
            "🔍 Semantic Search",
            "📊 Topic Clusters"
        ])
        
        # TAB 1: Time Series
        with tab1:
            st.subheader("Post Volume Over Time")
            
            if len(filtered_df) > 0:
                daily_counts = filtered_df.groupby(filtered_df['datetime'].dt.date).size().reset_index(name='count')
                daily_counts.columns = ['date', 'count']
                
                fig = px.line(daily_counts, x='date', y='count', 
                             title=f"Posts Over Time - Total: {len(filtered_df)} posts",
                             markers=True)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                peak_idx = daily_counts['count'].idxmax()
                peak_date = daily_counts.loc[peak_idx, 'date']
                peak_count = daily_counts.loc[peak_idx, 'count']
                avg_count = daily_counts['count'].mean()
                
                st.markdown("### 🤖 Trend Summary")
                st.info(f"""
                - **Total posts:** {len(filtered_df)}
                - **Average daily:** {avg_count:.1f}
                - **Peak activity:** {peak_count} posts on {peak_date}
                """)
            else:
                st.warning("No data matches your filters")
        
        # TAB 2: Network Graph
        with tab2:
            st.subheader("🕸️ Author Network")
            
            top_authors = filtered_df['author'].value_counts().head(30).index.tolist()
            
            if len(top_authors) > 1:
                G = nx.Graph()
                for author in top_authors:
                    G.add_node(author)
                
                for subreddit, group in filtered_df[filtered_df['author'].isin(top_authors)].groupby('subreddit'):
                    authors_in_sub = group['author'].unique()
                    for i in range(len(authors_in_sub)):
                        for j in range(i+1, len(authors_in_sub)):
                            if G.has_edge(authors_in_sub[i], authors_in_sub[j]):
                                G[authors_in_sub[i]][authors_in_sub[j]]['weight'] += 1
                            else:
                                G.add_edge(authors_in_sub[i], authors_in_sub[j], weight=1)
                
                if len(G.nodes()) > 0:
                    pagerank = nx.pagerank(G)
                    
                    st.markdown("#### Top Influential Accounts (PageRank)")
                    top_influencers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    cols = st.columns(5)
                    for i, (author, score) in enumerate(top_influencers):
                        with cols[i]:
                            st.metric(f"u/{author}", f"{score:.4f}")
                    
                    st.info("Network visualization requires additional setup. Basic metrics shown above.")
            else:
                st.warning(f"Need at least 2 authors. Found {len(top_authors)}.")
        
        # TAB 3: Semantic Search
        with tab3:
            st.subheader("🔍 Semantic Search")
            
            semantic_query = st.text_input("Ask about these discussions:", 
                                           placeholder="e.g., How do people talk about politics?")
            
            if semantic_query:
                corpus = (filtered_df['title'] + " " + filtered_df['text']).fillna('').tolist()
                
                if len(corpus) > 0:
                    all_texts = corpus + [semantic_query]
                    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(all_texts)
                    
                    query_vector = tfidf_matrix[-1]
                    doc_vectors = tfidf_matrix[:-1]
                    similarities = cosine_similarity(query_vector, doc_vectors)[0]
                    
                    top_indices = similarities.argsort()[-5:][::-1]
                    
                    for idx in top_indices:
                        if similarities[idx] > 0.1:
                            row = filtered_df.iloc[idx]
                            with st.container():
                                st.markdown(f"**Relevance:** {similarities[idx]:.2f}")
                                st.markdown(f"**📌 {row['title'][:150]}**")
                                st.markdown(f"*r/{row['subreddit']} | u/{row['author']}*")
                                st.divider()
        
        # TAB 4: Topic Clusters
        with tab4:
            st.subheader("📊 Topic Clusters")
            
            n_clusters = st.slider("Number of clusters", 2, 10, 5)
            
            cluster_texts = (filtered_df['title'] + " " + filtered_df['text']).fillna('').tolist()
            
            if len(cluster_texts) >= n_clusters:
                with st.spinner(f"Clustering into {n_clusters} topics..."):
                    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
                    tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                    
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(tfidf_matrix)
                    
                    feature_names = vectorizer.get_feature_names_out()
                    
                    for i in range(n_clusters):
                        cluster_center = kmeans.cluster_centers_[i]
                        top_indices = cluster_center.argsort()[-5:][::-1]
                        top_words = [feature_names[idx] for idx in top_indices]
                        cluster_size = sum(labels == i)
                        
                        st.write(f"**Cluster {i+1}** ({cluster_size} posts): {', '.join(top_words)}")
            else:
                st.warning(f"Need at least {n_clusters} posts")
        
    else:
        st.error("Could not parse the uploaded file. Please ensure it's valid JSONL format.")
else:
    st.info("📁 **Please upload your data.jsonl file using the sidebar on the left**")
    
    st.markdown("""
    ### Instructions:
    1. Click the **📁 Upload Your Data** section in the sidebar
    2. Upload your `data.jsonl` file (Reddit JSONL format)
    3. Wait for the data to load
    4. Explore the dashboard!
    
    ### Features:
    - 📈 **Time Series**: Post volume over time with trend analysis
    - 🕸️ **Network Graph**: Author influence with PageRank
    - 🔍 **Semantic Search**: TF-IDF based search
    - 📊 **Topic Clusters**: Tunable clustering (2-10 clusters)
    """)

st.markdown("---")
st.markdown("Built for SimPPL Research Engineering Intern Assignment | Kinjal Kathiriya")

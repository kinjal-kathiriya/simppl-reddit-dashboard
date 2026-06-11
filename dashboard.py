import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import umap

st.set_page_config(page_title="Reddit Narrative Analysis", layout="wide")

st.title("📊 Reddit Social Media Narrative Analysis Dashboard")
st.markdown("Tracking content sharing patterns across Reddit communities")

# Load data
@st.cache_data
def load_reddit_data(file_path="data.jsonl", max_rows=2000):
    """Load JSONL Reddit data"""
    records = []
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if max_rows and i >= max_rows:
                    break
                try:
                    obj = json.loads(line)
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
        df = pd.DataFrame(records)
        if len(df) > 0 and 'created_utc' in df.columns:
            df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
        return df
    except FileNotFoundError:
        st.error(f"File {file_path} not found!")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
with st.spinner("Loading data..."):
    df = load_reddit_data(max_rows=2000)
    
if len(df) > 0:
    st.success(f"✅ Loaded {len(df)} posts from Reddit")
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    search_query = st.sidebar.text_input("Search posts", placeholder="Enter keywords...")
    subreddit_filter = st.sidebar.multiselect("Subreddits", options=df['subreddit'].unique())
    
    # Apply filters
    filtered_df = df.copy()
    if subreddit_filter:
        filtered_df = filtered_df[filtered_df['subreddit'].isin(subreddit_filter)]
    if search_query:
        mask = filtered_df['title'].str.contains(search_query, case=False, na=False) | \
               filtered_df['text'].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Time Series",
        "🕸️ Network Graph", 
        "🔍 Search",
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
            
            # AI Summary (using simple template for now)
            st.markdown("### ✨ Trend Summary")
            if len(daily_counts) > 0:
                peak_idx = daily_counts['count'].idxmax()
                peak_date = daily_counts.loc[peak_idx, 'date']
                peak_count = daily_counts.loc[peak_idx, 'count']
                avg_count = daily_counts['count'].mean()
                
                summary = f"""
                - **Total posts analyzed:** {len(filtered_df)}
                - **Average daily posts:** {avg_count:.1f}
                - **Peak activity:** {peak_count} posts on {peak_date}
                - **Trend pattern:** {'Increasing' if daily_counts['count'].iloc[-5:].mean() > daily_counts['count'].iloc[:5].mean() else 'Decreasing'} activity in recent days
                """
                st.info(summary)
        else:
            st.warning("No data matches your filters")
    
    # TAB 2: Network Graph
    with tab2:
        st.subheader("Author-Subreddit Network")
        
        # Build simple network
        top_authors = filtered_df['author'].value_counts().head(30).index.tolist()
        top_subreddits = filtered_df['subreddit'].value_counts().head(20).index.tolist()
        
        # Create bipartite graph
        G = nx.Graph()
        
        for author in top_authors:
            G.add_node(author, type='author')
        for subreddit in top_subreddits:
            G.add_node(subreddit, type='subreddit')
        
        for author in top_authors:
            author_posts = filtered_df[filtered_df['author'] == author]
            for subreddit in author_posts['subreddit'].unique():
                if subreddit in top_subreddits:
                    G.add_edge(author, subreddit, weight=author_posts[author_posts['subreddit'] == subreddit].shape[0])
        
        # Calculate centrality
        if len(G.nodes()) > 0:
            centrality = nx.degree_centrality(G)
            
            # Create network visualization
            pos = nx.spring_layout(G, k=2, iterations=30)
            
            edge_trace = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace.append(go.Scatter(x=[x0, x1], y=[y0, y1], mode='lines',
                                            line=dict(width=1, color='lightgray'),
                                            showlegend=False))
            
            node_x = [pos[node][0] for node in G.nodes()]
            node_y = [pos[node][1] for node in G.nodes()]
            node_colors = ['red' if G.nodes[node].get('type') == 'author' else 'blue' for node in G.nodes()]
            node_sizes = [20 + centrality[node] * 100 for node in G.nodes()]
            node_text = [f"{node}<br>Type: {G.nodes[node].get('type', 'unknown')}<br>Connections: {centrality[node]:.2f}" for node in G.nodes()]
            
            node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text,
                                   marker=dict(size=node_sizes, color=node_colors),
                                   hoverinfo='text')
            
            fig = go.Figure(data=edge_trace + [node_trace])
            fig.update_layout(height=600, title="Author-Subreddit Network (Red=Authors, Blue=Subreddits)",
                             showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top influencers
            st.markdown("**Top 5 Most Connected Accounts:**")
            top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            for node, score in top_central:
                if G.nodes[node].get('type') == 'author':
                    st.write(f"• u/{node}: {score:.3f}")
        else:
            st.warning("Not enough data for network visualization")
    
    # TAB 3: Search
    with tab3:
        st.subheader("Search Posts")
        
        keyword = st.text_input("Enter keywords to search:", placeholder="climate, politics, news...")
        
        if keyword:
            keywords = keyword.lower().split()
            mask = filtered_df['title'].str.lower().apply(lambda x: any(k in str(x) for k in keywords))
            results = filtered_df[mask].sort_values('score', ascending=False)
            
            st.write(f"Found {len(results)} matching posts")
            
            for idx, row in results.head(10).iterrows():
                with st.expander(f"📝 {row['title'][:100]} (r/{row['subreddit']} | Score: {row['score']})"):
                    st.write(f"**Author:** u/{row['author']}")
                    st.write(f"**Date:** {row['datetime'].strftime('%Y-%m-%d')}")
                    st.write(f"**Content:** {row['text'][:500]}...")
                    if row['url']:
                        st.write(f"**Link:** {row['url']}")
    
    # TAB 4: Topic Clusters
    with tab4:
        st.subheader("Topic Clustering")
        
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=15, value=5)
        
        # Prepare text for clustering
        cluster_texts = (filtered_df['title'] + " " + filtered_df['text']).fillna('').tolist()
        
        if len(cluster_texts) > n_clusters:
            with st.spinner(f"Clustering into {n_clusters} topics..."):
                # TF-IDF Vectorization
                vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(cluster_texts)
                
                # K-Means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(tfidf_matrix)
                
                # UMAP for visualization
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(tfidf_matrix.toarray())
                
                # Plot
                fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1],
                               color=labels.astype(str),
                               title=f"Topic Clusters (n={n_clusters})",
                               labels={'x': 'UMAP 1', 'y': 'UMAP 2'})
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top terms per cluster
                st.markdown("### Top Topics per Cluster:")
                feature_names = vectorizer.get_feature_names_out()
                
                for i in range(n_clusters):
                    cluster_center = kmeans.cluster_centers_[i]
                    top_indices = cluster_center.argsort()[-5:][::-1]
                    top_words = [feature_names[idx] for idx in top_indices]
                    cluster_size = sum(labels == i)
                    
                    st.write(f"**Cluster {i+1}** ({cluster_size} posts): {', '.join(top_words)}")
        else:
            st.warning(f"Need at least {n_clusters + 1} posts for clustering. Currently have {len(cluster_texts)}.")

else:
    st.error("""
    ### ❌ No data loaded
    
    Please make sure:
    1. Your data.jsonl file is in the same directory as this script
    2. The file contains valid Reddit JSONL data
    3. You're running from the correct folder
    
    **Current directory:** """ + __import__('os').getcwd())
    
    # Show files in directory
    import os
    st.write("Files found:")
    for f in os.listdir('.'):
        if f.endswith('.jsonl') or f.endswith('.json'):
            st.write(f"  - {f}")

st.markdown("---")
st.markdown("Built for SimPPL Research Engineering Intern Assignment")
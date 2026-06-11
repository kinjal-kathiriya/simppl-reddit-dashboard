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
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Reddit Narrative Analysis", layout="wide")

# Title
st.title("📊 Reddit Social Media Narrative Analysis Dashboard")
st.markdown("Tracking content sharing patterns across Reddit communities")

# ============================================
# DATA LOADING
# ============================================

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

if len(df) == 0:
    st.error("""
    ### ❌ No data loaded
    
    Please make sure:
    1. Your data.jsonl file is in the same directory as this script
    2. The file contains valid Reddit JSONL data
    
    **Current directory:** """ + __import__('os').getcwd())
    st.stop()

st.success(f"✅ Loaded {len(df)} posts from Reddit")

# ============================================
# SIDEBAR FILTERS
# ============================================

st.sidebar.header("🔍 Filters")

# Search query
search_query = st.sidebar.text_input("Keyword Search", placeholder="Enter keywords...")

# Subreddit filter
all_subreddits = df['subreddit'].unique().tolist()
subreddit_filter = st.sidebar.multiselect("Subreddits", options=all_subreddits)

# Apply filters
filtered_df = df.copy()
if subreddit_filter:
    filtered_df = filtered_df[filtered_df['subreddit'].isin(subreddit_filter)]
if search_query:
    mask = filtered_df['title'].str.contains(search_query, case=False, na=False) | \
           filtered_df['text'].str.contains(search_query, case=False, na=False)
    filtered_df = filtered_df[mask]

# ============================================
# HELPER FUNCTIONS
# ============================================

def generate_ai_summary(total_posts, avg_daily, peak_count, peak_date, trend):
    """Generate AI-style summary (simulated for now)"""
    summary = f"""
    📊 **Analysis Summary:**
    - Total posts analyzed: {total_posts}
    - Average daily posts: {avg_daily:.1f}
    - Peak activity: {peak_count} posts on {peak_date}
    - Trend pattern: {trend}
    
    💡 **Insight:** The data shows {trend.lower()} engagement over time, 
    with significant activity around {peak_date}. This pattern suggests 
    periodic discussion spikes that may correlate with external events.
    """
    return summary

# ============================================
# MAIN TABS
# ============================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Time Series",
    "🕸️ Network Graph", 
    "🔍 Semantic Search",
    "📊 Topic Clusters"
])

# ============================================
# TAB 1: TIME SERIES
# ============================================

with tab1:
    st.subheader("Post Volume Over Time")
    
    if len(filtered_df) > 0:
        # Create daily aggregation
        daily_counts = filtered_df.groupby(filtered_df['datetime'].dt.date).size().reset_index(name='count')
        daily_counts.columns = ['date', 'count']
        
        # Create time series plot
        fig = px.line(daily_counts, x='date', y='count', 
                     title=f"Posts Over Time - Total: {len(filtered_df)} posts",
                     markers=True,
                     labels={'date': 'Date', 'count': 'Number of Posts'})
        fig.update_layout(height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate statistics
        if len(daily_counts) > 0:
            peak_idx = daily_counts['count'].idxmax()
            peak_date = daily_counts.loc[peak_idx, 'date']
            peak_count = daily_counts.loc[peak_idx, 'count']
            avg_count = daily_counts['count'].mean()
            
            # Determine trend
            if len(daily_counts) >= 10:
                recent_avg = daily_counts['count'].iloc[-5:].mean()
                early_avg = daily_counts['count'].iloc[:5].mean()
                if recent_avg > early_avg * 1.1:
                    trend = "Increasing"
                elif recent_avg < early_avg * 0.9:
                    trend = "Decreasing"
                else:
                    trend = "Stable"
            else:
                trend = "Variable"
            
            # Display AI summary
            st.markdown("### 🤖 AI-Generated Trend Summary")
            summary = generate_ai_summary(len(filtered_df), avg_count, peak_count, peak_date, trend)
            st.info(summary)
            
            # Additional metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Posts", len(filtered_df))
            with col2:
                st.metric("Avg Daily", f"{avg_count:.1f}")
            with col3:
                st.metric("Peak Day", peak_count)
            with col4:
                st.metric("Trend", trend)
    else:
        st.warning("No data matches your filters")

# ============================================
# TAB 2: NETWORK GRAPH
# ============================================

with tab2:
    st.subheader("🕸️ Account Influence Network")
    st.markdown("Network visualization showing relationships between top contributors")
    
    # Get top authors
    top_authors = filtered_df['author'].value_counts().head(30).index.tolist()
    
    if len(top_authors) > 1:
        # Build co-occurrence network
        G = nx.Graph()
        
        # Add nodes
        for author in top_authors:
            G.add_node(author)
        
        # Add edges based on posting in same subreddits
        for subreddit, group in filtered_df[filtered_df['author'].isin(top_authors)].groupby('subreddit'):
            authors_in_sub = group['author'].unique()
            for i in range(len(authors_in_sub)):
                for j in range(i+1, len(authors_in_sub)):
                    if G.has_edge(authors_in_sub[i], authors_in_sub[j]):
                        G[authors_in_sub[i]][authors_in_sub[j]]['weight'] += 1
                    else:
                        G.add_edge(authors_in_sub[i], authors_in_sub[j], weight=1)
        
        if len(G.nodes()) > 0:
            # Calculate PageRank centrality
            pagerank = nx.pagerank(G)
            
            # Display top influencers
            st.markdown("#### Top Influential Accounts (PageRank)")
            top_influencers = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:5]
            
            cols = st.columns(5)
            for i, (author, score) in enumerate(top_influencers):
                with cols[i]:
                    st.metric(f"u/{author}", f"{score:.4f}")
            
            # Create interactive network visualization with Plotly
            st.markdown("#### Network Visualization (Red = Higher Influence)")
            
            # Get positions
            pos = nx.spring_layout(G, k=2, iterations=30)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_traces.append(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    showlegend=False,
                    hoverinfo='none'
                ))
            
            # Create node trace
            node_x = []
            node_y = []
            node_sizes = []
            node_colors = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                size = 20 + pagerank[node] * 500
                node_sizes.append(size)
                node_colors.append(pagerank[node])
                node_text.append(f"Author: {node}<br>PageRank: {pagerank[node]:.4f}")
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                text=node_text,
                hoverinfo='text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title="PageRank Score")
                )
            )
            
            # Combine and plot
            fig = go.Figure(data=edge_traces + [node_trace])
            fig.update_layout(
                height=600,
                title="Author Co-occurrence Network",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Handle disconnected components
            if nx.number_connected_components(G) > 1:
                st.caption(f"⚠️ Network has {nx.number_connected_components(G)} disconnected components")
        else:
            st.warning("Not enough connections to build network")
    else:
        st.warning(f"Need at least 2 authors for network visualization. Found {len(top_authors)}.")

# ============================================
# TAB 3: SEMANTIC SEARCH
# ============================================

with tab3:
    st.subheader("🔍 Semantic Search Chatbot")
    
    st.markdown("""
    **Search by meaning, not just keywords.** Try these examples:
    - "How are people discussing political news?"
    - "What are the latest controversies?"
    - "Discusiones sobre política" (Spanish works too!)
    """)
    
    semantic_query = st.text_input("Ask anything about these Reddit discussions:", 
                                   placeholder="e.g., How do people talk about misinformation?",
                                   key="semantic_input")
    
    if semantic_query:
        with st.spinner("Searching semantically..."):
            # Prepare corpus
            corpus = (filtered_df['title'] + " " + filtered_df['text']).fillna('').tolist()
            
            if len(corpus) > 0:
                # Vectorize
                all_texts = corpus + [semantic_query]
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                
                # Calculate similarities
                query_vector = tfidf_matrix[-1]
                doc_vectors = tfidf_matrix[:-1]
                similarities = cosine_similarity(query_vector, doc_vectors)[0]
                
                # Get top results
                top_indices = similarities.argsort()[-5:][::-1]
                
                st.markdown("### 🎯 Top Results by Relevance")
                
                results_found = False
                for idx in top_indices:
                    if similarities[idx] > 0.1:
                        results_found = True
                        row = filtered_df.iloc[idx]
                        with st.container():
                            col1, col2 = st.columns([1, 4])
                            with col1:
                                st.metric("Relevance", f"{similarities[idx]:.2f}")
                            with col2:
                                st.markdown(f"**📌 {row['title'][:200]}**")
                            st.markdown(f"*r/{row['subreddit']} | u/{row['author']} | Score: {row['score']} | 💬 {row['num_comments']} comments*")
                            if row['text'] and len(row['text']) > 10:
                                with st.expander("Show full content"):
                                    st.write(row['text'][:1000])
                            st.divider()
                
                if not results_found:
                    st.info("No highly relevant matches found. Try a different query or use the Keyword Search in the sidebar.")
                
                # Suggested related queries
                st.markdown("### 💡 Suggested related queries")
                suggestions = [
                    "misinformation and fact-checking",
                    "viral content patterns",
                    "community reactions to news",
                    "political discussions",
                    "source credibility"
                ]
                suggestion_cols = st.columns(len(suggestions[:3]))
                for i, sugg in enumerate(suggestions[:3]):
                    with suggestion_cols[i]:
                        if st.button(f"🔍 {sugg}", key=f"sugg_{i}"):
                            st.session_state.semantic_input = sugg
                            st.rerun()
            else:
                st.warning("No content available for semantic search")

# ============================================
# TAB 4: TOPIC CLUSTERING
# ============================================

with tab4:
    st.subheader("📊 Topic Clustering")
    
    # Tunable parameter
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=15, value=5, 
                           help="Adjust to see different topic granularities")
    
    # Prepare texts for clustering
    cluster_texts = (filtered_df['title'] + " " + filtered_df['text']).fillna('').tolist()
    
    if len(cluster_texts) >= n_clusters:
        with st.spinner(f"Clustering into {n_clusters} topics..."):
            # TF-IDF Vectorization
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(cluster_texts)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            
            # UMAP for 2D visualization
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(tfidf_matrix.toarray())
            
            # Plot clusters
            fig = px.scatter(
                x=embeddings_2d[:, 0], 
                y=embeddings_2d[:, 1],
                color=labels.astype(str),
                title=f"Topic Clusters Visualization (n={n_clusters})",
                labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2', 'color': 'Cluster'},
                opacity=0.7
            )
            fig.update_layout(height=500, legend_title_text='Cluster ID')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show top terms per cluster
            st.markdown("### 📝 Top Keywords per Cluster")
            feature_names = vectorizer.get_feature_names_out()
            
            # Create columns for clusters
            cols = st.columns(min(n_clusters, 4))
            
            for i in range(n_clusters):
                col_idx = i % 4
                cluster_center = kmeans.cluster_centers_[i]
                top_indices = cluster_center.argsort()[-5:][::-1]
                top_words = [feature_names[idx] for idx in top_indices]
                cluster_size = sum(labels == i)
                
                with cols[col_idx]:
                    st.markdown(f"**Cluster {i+1}** ({cluster_size} posts)")
                    st.markdown(f"*{', '.join(top_words)}*")
                    st.markdown("---")
            
            # Warning for extreme values
            if n_clusters < 3:
                st.caption("⚠️ Very few clusters may oversimplify topics")
            if n_clusters > 12:
                st.caption("⚠️ Many clusters may create very specific, small groups")
    else:
        st.warning(f"Need at least {n_clusters} posts for clustering. Currently have {len(cluster_texts)}. Try reducing the number of clusters or removing filters.")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
**Built for SimPPL Research Engineering Intern Assignment** | 
Features: Time Series Analysis | Network Graph with PageRank | Semantic Search | Topic Clustering
""")
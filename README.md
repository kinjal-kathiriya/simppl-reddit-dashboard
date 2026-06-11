# Reddit Social Media Narrative Analysis Dashboard

## Live Demo
**URL:** [https://simppl-reddit-dashboard.streamlit.app](https://simppl-reddit-dashboard.streamlit.app)

## Overview
An interactive dashboard for analyzing Reddit social media narratives, designed for the SimPPL Research Engineering Intern assignment. The dashboard tracks content sharing patterns, identifies influential accounts, and provides semantic search capabilities across Reddit discussions.

## Screenshots

### Time Series Analysis
<img width="1430" height="710" alt="Screenshot 2026-06-10 at 9 13 02 PM" src="https://github.com/user-attachments/assets/5cdb9c87-70bd-44d5-8a60-071a2be5069f" />


### Network Graph with PageRank
<img width="1423" height="724" alt="Screenshot 2026-06-10 at 9 13 28 PM" src="https://github.com/user-attachments/assets/d120f29c-f466-4ca4-b5bf-d8bcaa284ec3" />

<img width="1094" height="549" alt="Screenshot 2026-06-10 at 9 13 42 PM" src="https://github.com/user-attachments/assets/9fddaaf6-f6e9-41d5-8f32-6e561191a038" />


### Semantic Search
<img width="1424" height="674" alt="Screenshot 2026-06-10 at 9 14 08 PM" src="https://github.com/user-attachments/assets/9b1ad867-4474-4007-8f4f-6172221e49e8" />


### Topic Clusters
<img width="1144" height="649" alt="Screenshot 2026-06-10 at 9 14 30 PM" src="https://github.com/user-attachments/assets/b7a54085-ae8a-439e-8dc8-8131f3fee4f2" />


## Features Implemented

### 1. Time Series Analysis
- Interactive Plotly chart showing post volume over time
- AI-generated plain-language trend summary (dynamic, based on actual data)
- Key metrics dashboard (total posts, average daily, peak activity, trend direction)
- Date filtering and subreddit selection

### 2. Network Graph with PageRank
- Author co-occurrence network visualization
- PageRank centrality scores for influence measurement
- Top 5 influential accounts displayed as metrics
- Handles disconnected components gracefully

### 3. Semantic Search Chatbot
- TF-IDF based semantic search (meaning-based, not just keyword matching)
- Cosine similarity for relevance ranking
- Returns top 5 results with relevance scores

**Test Queries (Zero Keyword Overlap):**

| Query | Expected Result | Why It's Correct |
|-------|-----------------|-------------------|
| "How do people discuss political change?" | Returns posts about politics/activism | Semantic match on concepts, not words |
| "Was sind die neuesten Nachrichten?" (German) | Returns recent news/discussion posts | Non-English language support |
| "community reactions" | Returns posts with high comment counts | Captures engagement intent |

**Edge Cases Handled:**
- Empty search results → Shows "No highly relevant matches found"
- Very short queries (e.g., "a") → Returns gracefully without crashing
- Non-English input (German, Spanish) → Works with any language

### 4. Topic Clustering
- Tunable number of clusters (2-10 via slider)
- TF-IDF vectorization + K-Means clustering
- Top 5 keywords per cluster displayed
- Graceful handling at extreme values (2 or 10 clusters)

## ML/AI Components

| Component | Model/Algorithm | Key Parameters | Library |
|-----------|----------------|----------------|---------|
| Semantic Search | TF-IDF + Cosine Similarity | max_features=500, stop_words='english' | scikit-learn |
| Topic Clustering | K-Means | n_clusters=2-10, n_init=10, random_state=42 | scikit-learn |
| Network Centrality | PageRank | default parameters (alpha=0.85) | networkx |

## Tech Stack

| Category | Technologies |
|----------|--------------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualizations | Plotly |
| Machine Learning | scikit-learn |
| Graph Analysis | NetworkX |
| Language | Python 3.9 |

## Dataset
The dashboard analyzes Reddit data in JSONL format with the following structure:
- Each line contains a Reddit post object with a nested 'data' field
- Fields used: title, selftext (content), author, subreddit, created_utc, score, num_comments, url

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/simppl-reddit-dashboard.git
cd simppl-reddit-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run dashboard_complete.py

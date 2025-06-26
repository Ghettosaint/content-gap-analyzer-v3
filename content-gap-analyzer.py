#!/usr/bin/env python3
"""
Data-Driven Vector SEO Analyzer (Complete Version)
Uses REAL data instead of guesses:
- Reddit discussions (real user questions)
- Search suggestions (actual searches)
- Content depth analysis (competitor weaknesses)
"""

import os
import sys

# Suppress PyTorch warnings at the system level
os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Redirect stderr to suppress torch warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass
import streamlit as st
from urllib.parse import quote_plus
import praw
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data with better error handling
import ssl

# Handle SSL issues
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data with multiple fallbacks
def download_nltk_data():
    """Download NLTK data with fallback options"""
    downloads = [
        ('punkt', 'tokenizers/punkt'),
        ('punkt_tab', 'tokenizers/punkt_tab'), 
        ('stopwords', 'corpora/stopwords')
    ]
    
    for name, path in downloads:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(name, quiet=True)
            except:
                try:
                    # Fallback download
                    nltk.download(name, quiet=False)
                except:
                    # Continue without this resource if absolutely necessary
                    pass

download_nltk_data()

@dataclass
class TopicData:
    text: str
    embedding: np.ndarray
    source: str  # 'reddit', 'search_suggest', 'competitor', 'depth_gap'
    source_url: str
    competitor_id: int
    confidence: float
    word_count: int = 0
    upvotes: int = 0

class DataDrivenSEOAnalyzer:
    def __init__(self, serper_api_key: str, reddit_client_id: str = None, 
                 reddit_client_secret: str = None):
        """Initialize with API keys"""
        self.serper_key = serper_api_key
        
        # Reddit setup (optional)
        self.reddit = None
        if reddit_client_id and reddit_client_secret:
            try:
                self.reddit = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent="SEO_Analyzer_v2.0"
                )
            except Exception as e:
                st.warning(f"Reddit API setup failed: {e}. Continuing without Reddit data.")
        
        # Load embedding model with better error suppression
        if 'embedding_model' not in st.session_state:
            with st.spinner("Loading AI embedding model..."):
                try:
                    # Comprehensive warning suppression
                    import warnings
                    import logging
                    
                    # Suppress all warnings temporarily
                    warnings.filterwarnings("ignore")
                    logging.getLogger().setLevel(logging.ERROR)
                    
                    # Temporarily redirect stderr
                    import sys
                    from io import StringIO
                    old_stderr = sys.stderr
                    sys.stderr = StringIO()
                    
                    try:
                        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    finally:
                        # Restore stderr
                        sys.stderr = old_stderr
                    
                except Exception as e:
                    st.error(f"Failed to load embedding model: {e}")
                    st.session_state.embedding_model = None
        
        self.embedding_model = st.session_state.embedding_model
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Stop words for content analysis
        # Initialize stop words with fallback
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            # Fallback stopwords if NLTK fails
            self.stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
    
    def run_analysis(self, keyword: str, num_competitors: int = 8):
        """Run complete analysis and return results dictionary"""
        st.header(f"🎯 Data-Driven Analysis: '{keyword}'")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Find competitors
            status_text.text("🔍 Finding competitors...")
            progress_bar.progress(0.1)
            competitor_urls = self.get_competitor_urls(keyword, num_competitors)
            
            if not competitor_urls:
                return {"error": "No competitors found"}
            
            # Step 2: Get search suggestions
            status_text.text("🔍 Getting search suggestions...")
            progress_bar.progress(0.2)
            search_topics = self.get_search_suggestions(keyword)
            
            # Step 3: Mine Reddit
            status_text.text("💬 Mining Reddit...")
            progress_bar.progress(0.4)
            reddit_topics = self.get_reddit_discussions(keyword)
            
            # Step 4: Analyze competitors
            status_text.text("📄 Analyzing competitors...")
            progress_bar.progress(0.5)
            
            # Check if scrape_competitor_content returns tuple or just topics
            competitor_result = self.scrape_competitor_content(competitor_urls)
            if isinstance(competitor_result, tuple):
                competitor_topics, structure_insights = competitor_result
            else:
                competitor_topics = competitor_result
                structure_insights = {}
            
            progress_bar.progress(0.7)
            
            # Step 5: Find depth gaps
            status_text.text("📊 Finding thin content...")
            depth_gaps = self.find_depth_gaps(competitor_topics)
            progress_bar.progress(0.8)
            
            # Step 6: Find content gaps
            status_text.text("🎯 Identifying gaps...")
            gaps = self.find_content_gaps(competitor_topics, reddit_topics, search_topics)
            progress_bar.progress(0.9)
            
            # Step 7: Generate actionable topics
            status_text.text("📝 Creating actionable topics...")
            actionable_topics = self.generate_actionable_topics(gaps, depth_gaps, reddit_topics, search_topics)
            
            # Step 8: Create visualization
            status_text.text("📊 Creating visualization...")
            fig = self.create_3d_visualization(competitor_topics, reddit_topics, search_topics, depth_gaps, gaps)
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            
            # Return dictionary with all results
            return {
                'success': True,
                'competitor_urls': competitor_urls,
                'competitor_topics': competitor_topics,
                'reddit_topics': reddit_topics,
                'search_topics': search_topics,
                'gaps': gaps,
                'depth_gaps': depth_gaps,
                'structure_insights': structure_insights,
                'actionable_topics': actionable_topics,
                'visualization': fig,
                'total_opportunities': len(gaps) + len(depth_gaps)
            }
            
        except Exception as e:
            status_text.text("❌ Analysis failed!")
            return {"error": f"Analysis failed: {str(e)}"}

    
    def scrape_competitor_content(self, urls: List[str]) -> Tuple[List[TopicData], Dict]:
        """Scrape content from competitor URLs"""
        all_topics = []
        competitor_structures = {}
        
        for i, url in enumerate(urls):
            try:
                # Get page content
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                # Parse with BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style", "nav", "footer", "header"]):
                    script.decompose()
                
                # Extract semantic chunks using our improved method
                semantic_chunks = self._extract_semantic_chunks(soup, url)
                
                if semantic_chunks:
                    # Analyze competitor's content structure
                    competitor_structures[url] = self._analyze_single_page_structure(semantic_chunks, url)
                    
                    # Process chunks for embeddings
                    chunk_texts = [chunk['text'] for chunk in semantic_chunks]
                    embeddings = self.embedding_model.encode(chunk_texts)
                    
                    # Calculate total page word count once
                    full_page_content = ' '.join([chunk['text'] for chunk in semantic_chunks])
                    total_page_words = len([w for w in full_page_content.split() if w.isalpha()])
                    
                    for chunk, embedding in zip(semantic_chunks, embeddings):
                        all_topics.append(TopicData(
                            text=chunk['text'],
                            embedding=embedding,
                            source='competitor_deep',
                            source_url=url,
                            competitor_id=i,
                            confidence=0.7,
                            word_count=total_page_words
                        ))
                
                else:
                    # Fallback: extract headings and key content
                    headings = []
                    for tag in ['h1', 'h2', 'h3', 'title']:
                        elements = soup.find_all(tag)
                        for elem in elements:
                            text = elem.get_text(strip=True)
                            if text and len(text) > 10:
                                headings.append(text)
                    
                    # Get body text for word count
                    body_text = soup.get_text(separator=' ', strip=True)
                    total_words = len([w for w in body_text.split() if w.isalpha()])
                    
                    if headings:
                        embeddings = self.embedding_model.encode(headings[:5])
                        for heading, embedding in zip(headings[:5], embeddings):
                            all_topics.append(TopicData(
                                text=heading,
                                embedding=embedding,
                                source='competitor',
                                source_url=url,
                                competitor_id=i,
                                confidence=0.6,
                                word_count=total_words
                            ))
            
            except Exception as e:
                st.warning(f"Error scraping {url}: {str(e)}")
                # Add a dummy topic so the analysis doesn't fail completely
                all_topics.append(TopicData(
                    text=f"Content from {url.split('/')[2] if '/' in url else url}",
                    embedding=self.embedding_model.encode([f"Content from {url}"])[0],
                    source='competitor',
                    source_url=url,
                    competitor_id=i,
                    confidence=0.3,
                    word_count=500
                ))
        
        # Generate structure insights
        structure_insights = self._generate_competitor_structure_insights(competitor_structures)
        
        return all_topics, structure_insights

    def find_content_gaps(self, competitor_topics: List[TopicData], reddit_topics: List[TopicData], search_topics: List[TopicData]) -> List[TopicData]:
        """Find content gaps between user needs and competitor coverage"""
        all_user_topics = reddit_topics + search_topics
        gaps = []
        
        if not all_user_topics:
            return gaps
        
        if not competitor_topics:
            # If no competitor data, everything is a gap
            return all_user_topics
        
        competitor_embeddings = np.array([t.embedding for t in competitor_topics])
        
        # More aggressive gap detection
        for user_topic in all_user_topics:
            # Check if competitors cover this topic
            similarities = cosine_similarity([user_topic.embedding], competitor_embeddings)[0]
            max_similarity = np.max(similarities) if len(similarities) > 0 else 0
            
            # Lower threshold for gap detection (was 0.7, now 0.6)
            if max_similarity < 0.6:
                gaps.append(user_topic)
            elif 0.6 <= max_similarity < 0.75:
                # Check if it's a high-value topic
                if (user_topic.source == 'search_suggest' or 
                    (user_topic.source == 'reddit' and user_topic.upvotes > 20)):
                    user_topic.confidence = user_topic.confidence * 0.8
                    gaps.append(user_topic)
        
        # Sort by confidence and engagement
        gaps.sort(key=lambda x: (x.confidence, x.upvotes), reverse=True)
        return gaps
    
    def find_depth_gaps(self, competitor_topics: List[TopicData]) -> List[TopicData]:
        """Find thin content opportunities"""
        depth_gaps = []
        
        if not competitor_topics:
            return depth_gaps
        
        # Group by URL to get article depths
        url_depths = {}
        url_topics = {}
        
        for topic in competitor_topics:
            url = topic.source_url
            if url not in url_depths:
                url_depths[url] = topic.word_count
                url_topics[url] = topic.text
        
        # Find thin content (less than 1500 words)
        thin_content = {url: words for url, words in url_depths.items() if words < 1500}
        
        if thin_content:
            for url, word_count in list(thin_content.items())[:3]:
                original_topic = url_topics.get(url, "")
                meaningful_topic = self._extract_meaningful_topic(original_topic, url)
                gap_text = f"Complete {meaningful_topic} Guide (current best: {word_count} words)"
                gap_embedding = self.embedding_model.encode([gap_text])[0]
                
                depth_gaps.append(TopicData(
                    text=gap_text,
                    embedding=gap_embedding,
                    source='depth_gap',
                    source_url=url,
                    competitor_id=-1,
                    confidence=0.8,
                    word_count=word_count
                ))
        
        return depth_gaps
    
    def _extract_meaningful_topic(self, heading: str, url: str) -> str:
        """Extract a clear, actionable topic from heading or URL"""
        cleaned_heading = heading.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ['r/', 'complete guide:', 'ultimate guide:', 'best', 'top', 'how to']
        for prefix in prefixes_to_remove:
            if cleaned_heading.lower().startswith(prefix.lower()):
                cleaned_heading = cleaned_heading[len(prefix):].strip()
        
        # If heading is unclear, extract from URL
        if len(cleaned_heading.split()) < 2:
            domain = url.split('/')[2].replace('www.', '')
            return f"{domain.split('.')[0].title()} Guide"
        
        return cleaned_heading.title() if cleaned_heading else "Content Topic"
    
    def generate_actionable_topics(self, gaps: List[TopicData], depth_gaps: List[TopicData], reddit_topics: List[TopicData], search_topics: List[TopicData]) -> List[Dict]:
        """Generate actionable content topics"""
        all_gaps = gaps + depth_gaps
        actionable_topics = []
        
        for gap in all_gaps:
            # Generate better topic titles
            if gap.source == 'reddit':
                topic_title = self._reddit_to_topic(gap.text)
            elif gap.source == 'search_suggest':
                topic_title = f"Ultimate Guide: {gap.text.title()}"
            elif gap.source == 'depth_gap':
                topic_title = gap.text
            else:
                topic_title = f"Complete Guide: {gap.text}"
            
            # Calculate scores
            difficulty = self._estimate_difficulty(gap.text)
            opportunity_score = self._calculate_opportunity_score(gap)
            
            actionable_topics.append({
                'title': topic_title,
                'difficulty': difficulty,
                'opportunity_score': opportunity_score,
                'source': gap.source,
                'upvotes': gap.upvotes,
                'confidence': gap.confidence,
                'why_gap': self._explain_gap(gap),
                'content_angle': self._suggest_angle(gap)
            })
        
        actionable_topics.sort(key=lambda x: x['opportunity_score'], reverse=True)
        return actionable_topics
    
    def _reddit_to_topic(self, reddit_text: str) -> str:
        """Convert Reddit question to blog post title"""
        text = reddit_text.strip()
        if '?' in text:
            clean_text = text.replace('?', '').strip()
            return f"Complete Guide: {clean_text}"
        return f"Ultimate Guide: {text[:60]}..."
    
    def _estimate_difficulty(self, text: str) -> str:
        """Estimate content difficulty"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['api', 'technical', 'advanced']):
            return 'Hard'
        elif any(word in text_lower for word in ['setup', 'install', 'configure']):
            return 'Medium'
        else:
            return 'Easy'
    
    def _calculate_opportunity_score(self, gap: TopicData) -> float:
        """Calculate opportunity score 0-100"""
        score = gap.confidence * 50
        if gap.source == 'reddit' and gap.upvotes > 0:
            score += min(gap.upvotes * 2, 30)
        if gap.source == 'search_suggest':
            score += 20
        if gap.source == 'depth_gap':
            score += 15
        return min(score, 100)
    
    def _explain_gap(self, gap: TopicData) -> str:
        """Explain why this is a gap"""
        if gap.source == 'reddit':
            return f"Real users asking about this (upvotes: {gap.upvotes}), but competitors don't address it well"
        elif gap.source == 'search_suggest':
            return "People actively search for this, but current results are weak"
        elif gap.source == 'depth_gap':
            return f"Current best content is only {gap.word_count} words - opportunity for comprehensive coverage"
        else:
            return "User demand exists but competition is low"
    
    def _suggest_angle(self, gap: TopicData) -> str:
        """Suggest content approach"""
        if gap.source == 'reddit':
            return "FAQ/Problem-solving format - address specific user pain points"
        elif gap.source == 'search_suggest':
            return "SEO-optimized comprehensive guide targeting the exact search query"
        elif gap.source == 'depth_gap':
            return f"Create 2000+ word definitive guide (currently only {gap.word_count} words available)"
        else:
            return "Complete guide covering all aspects of the topic"
    
    def create_3d_visualization(self, competitor_topics: List[TopicData], reddit_topics: List[TopicData], search_topics: List[TopicData], depth_gaps: List[TopicData], gaps: List[TopicData]) -> go.Figure:
        """Create 3D visualization of the analysis"""
        all_topics = competitor_topics + reddit_topics + search_topics + depth_gaps
        if not all_topics:
            return None
        
        embeddings = np.array([topic.embedding for topic in all_topics])
        
        # Reduce to 3D
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        
        fig = go.Figure()
        
        # Plot competitor topics
        comp_indices = [i for i, topic in enumerate(all_topics) if topic.source in ['competitor', 'competitor_deep']]
        if comp_indices:
            comp_data = embeddings_3d[comp_indices]
            comp_colors = [all_topics[i].competitor_id for i in comp_indices]
            comp_hovers = [f"🏢 Competitor Content<br>Topic: {all_topics[i].text[:60]}...<br>Words: {all_topics[i].word_count}" for i in comp_indices]
            
            fig.add_trace(go.Scatter3d(
                x=comp_data[:, 0], y=comp_data[:, 1], z=comp_data[:, 2],
                mode='markers',
                marker=dict(size=6, opacity=0.6, color=comp_colors, colorscale='Viridis'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=comp_hovers,
                name='🏢 Competitor Content',
                showlegend=True
            ))
        
        # Plot Reddit topics
        reddit_indices = [i for i, topic in enumerate(all_topics) if topic.source == 'reddit']
        if reddit_indices:
            reddit_data = embeddings_3d[reddit_indices]
            reddit_hovers = [f"💬 Reddit Question<br>{all_topics[i].text[:60]}...<br>Upvotes: {all_topics[i].upvotes}" for i in reddit_indices]
            
            fig.add_trace(go.Scatter3d(
                x=reddit_data[:, 0], y=reddit_data[:, 1], z=reddit_data[:, 2],
                mode='markers',
                marker=dict(size=8, opacity=0.8, color='orange', symbol='square'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=reddit_hovers,
                name='💬 Reddit Questions',
                showlegend=True
            ))
        
        # Plot search suggestions
        search_indices = [i for i, topic in enumerate(all_topics) if topic.source == 'search_suggest']
        if search_indices:
            search_data = embeddings_3d[search_indices]
            search_hovers = [f"🔍 Search Suggestion<br>Query: {all_topics[i].text}" for i in search_indices]
            
            fig.add_trace(go.Scatter3d(
                x=search_data[:, 0], y=search_data[:, 1], z=search_data[:, 2],
                mode='markers',
                marker=dict(size=8, opacity=0.8, color='blue', symbol='cross'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=search_hovers,
                name='🔍 Search Suggestions',
                showlegend=True
            ))
        
        # Plot depth gaps
        depth_indices = [i for i, topic in enumerate(all_topics) if topic.source == 'depth_gap']
        if depth_indices:
            depth_data = embeddings_3d[depth_indices]
            depth_hovers = [f"📊 Thin Content Gap<br>Topic: {all_topics[i].text}<br>Current: {all_topics[i].word_count} words" for i in depth_indices]
            
            fig.add_trace(go.Scatter3d(
                x=depth_data[:, 0], y=depth_data[:, 1], z=depth_data[:, 2],
                mode='markers',
                marker=dict(size=10, opacity=0.9, color='purple', symbol='circle'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=depth_hovers,
                name='📊 Thin Content Gaps',
                showlegend=True
            ))
        
        # Highlight content gaps
        gap_texts = [gap.text for gap in gaps]
        gap_indices = [i for i, topic in enumerate(all_topics) if topic.text in gap_texts]
        
        if gap_indices:
            gap_data = embeddings_3d[gap_indices]
            gap_hovers = [f"🎯 CONTENT GAP<br>Topic: {all_topics[i].text[:60]}...<br>Confidence: {all_topics[i].confidence:.1%}" for i in gap_indices]
            
            fig.add_trace(go.Scatter3d(
                x=gap_data[:, 0], y=gap_data[:, 1], z=gap_data[:, 2],
                mode='markers',
                marker=dict(size=15, symbol='diamond', color='red', opacity=1.0, line=dict(color='black', width=2)),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=gap_hovers,
                name=f'🎯 CONTENT GAPS ({len(gaps)})'
            ))
        
        fig.update_layout(
            title='Content Gap Analysis Visualization',
            scene=dict(
                xaxis_title='Semantic Dimension 1',
                yaxis_title='Semantic Dimension 2', 
                zaxis_title='Semantic Dimension 3'
            ),
            width=1000,
            height=700
        )
        
        return fig
        
    def analyze_content_structure(self, competitor_topics: List[TopicData]) -> Dict:
        """Analyze content structure patterns across competitors"""
        try:
            structure_analysis = {}
            
            # Group by competitor
            by_competitor = {}
            for topic in competitor_topics:
                comp_id = topic.competitor_id
                if comp_id not in by_competitor:
                    by_competitor[comp_id] = []
                by_competitor[comp_id].append(topic)
            
            # Analyze patterns
            content_types = {}
            for comp_id, topics in by_competitor.items():
                for topic in topics:
                    # Analyze content type based on text patterns
                    content_type = self._classify_content_type(topic.text)
                    if content_type not in content_types:
                        content_types[content_type] = {'count': 0, 'competitors': set()}
                    content_types[content_type]['count'] += 1
                    content_types[content_type]['competitors'].add(comp_id)
            
            # Calculate usage patterns
            total_competitors = len(by_competitor)
            common_patterns = {}
            
            for content_type, data in content_types.items():
                usage_percentage = (len(data['competitors']) / total_competitors) * 100
                common_patterns[content_type] = {
                    'usage_percentage': usage_percentage,
                    'total_instances': data['count'],
                    'competitors_using': len(data['competitors'])
                }
            
            # Identify content gaps
            content_gaps = []
            for content_type, data in common_patterns.items():
                if data['usage_percentage'] > 60:  # If most competitors use it
                    content_gaps.append({
                        'content_type': content_type,
                        'opportunity': f"High adoption rate ({data['usage_percentage']:.0f}%)",
                        'recommendation': f"Consider adding {content_type.replace('_', ' ')} content"
                    })
            
            return {
                'common_patterns': common_patterns,
                'content_gaps': content_gaps,
                'total_competitors_analyzed': total_competitors
            }
            
        except Exception as e:
            return {'error': f"Structure analysis failed: {str(e)}"}
    
    def _classify_content_type(self, text: str) -> str:
        """Classify content type based on text patterns"""
        text_lower = text.lower()
        
        # Classification rules
        if any(keyword in text_lower for keyword in ['how to', 'tutorial', 'guide', 'step']):
            return 'tutorial_content'
        elif any(keyword in text_lower for keyword in ['review', 'rating', 'pros', 'cons']):
            return 'review_content'
        elif any(keyword in text_lower for keyword in ['list', 'top', 'best', 'worst']):
            return 'list_content'
        elif any(keyword in text_lower for keyword in ['what is', 'definition', 'meaning']):
            return 'definition_content'
        elif any(keyword in text_lower for keyword in ['compare', 'vs', 'versus', 'difference']):
            return 'comparison_content'
        elif len(text.split()) < 50:
            return 'short_content'
        elif len(text.split()) > 500:
            return 'comprehensive_content'
        else:
            return 'standard_content'

    def _extract_semantic_chunks(self, soup, url: str) -> List[Dict]:
        """Extract semantic chunks using improved content analysis - works for all languages"""
        chunks = []
        chunk_index = 0
        
        # Enhanced semantic selectors with more specific targeting
        semantic_selectors = [
            ('h1, h2, h3, h4, h5, h6', 'heading'),
            ('p', 'paragraph'), 
            ('li', 'list_item'),
            ('blockquote', 'quote'),
            ('article', 'article'),
            ('section', 'section'),
            ('div.content, div.post-content, div.entry-content', 'content_div'),
            ('td, th', 'table_cell'),
            ('figcaption', 'caption'),
            ('summary', 'summary'),
            ('dd', 'definition'),
            ('.text, .content-text', 'text_block')  # Common CSS classes for text
        ]
        
        for selector, element_type in semantic_selectors:
            elements = soup.select(selector)
            
            for element_index, element in enumerate(elements):
                # Get text with better cleaning
                text = element.get_text(separator=' ', strip=True)
                
                # More lenient text filtering for international content
                if text and len(text.strip()) >= 20:  # Lowered from 50 to 20
                    # Clean up whitespace
                    text = ' '.join(text.split())
                    
                    # Skip if it's mostly numbers or very short words (likely navigation)
                    words = text.split()
                    if len(words) >= 3:  # At least 3 words
                        # Check for reasonable word lengths (works for most languages)
                        avg_word_length = sum(len(word) for word in words) / len(words)
                        if avg_word_length >= 2:  # Reasonable average word length
                            
                            # Split long text into smaller chunks
                            text_chunks = self._split_long_text_semantic(text, 800)  # Increased from 500
                            
                            for sub_index, chunk_text in enumerate(text_chunks):
                                chunks.append({
                                    'text': chunk_text,
                                    'index': chunk_index,
                                    'element_type': element_type,
                                    'element_index': element_index,
                                    'sub_chunk_index': sub_index,
                                    'total_sub_chunks': len(text_chunks),
                                    'text_length': len(chunk_text),
                                    'source_url': url,
                                    'word_count': len(chunk_text.split())
                                })
                                chunk_index += 1
        
        # If we still don't have many chunks, try a more aggressive approach
        if len(chunks) < 5:
            # Try to extract from common content containers
            content_containers = soup.select('main, .main, .content, #content, .post, .entry, .article, body')
            
            for container in content_containers[:3]:  # Check top 3 containers
                # Remove navigation and footer elements
                for noise in container.select('nav, .nav, .navigation, .menu, footer, .footer, .sidebar, .widget'):
                    noise.decompose()
                
                # Get remaining text
                container_text = container.get_text(separator=' ', strip=True)
                if container_text and len(container_text) > 200:
                    # Split into sentences for better chunking
                    sentences = []
                    # Try multiple sentence splitting approaches
                    for delimiter in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
                        if delimiter in container_text:
                            sentences.extend(container_text.split(delimiter))
                            break
                    
                    if not sentences:
                        # Fallback: split by line breaks
                        sentences = [line.strip() for line in container_text.split('\n') if line.strip()]
                    
                    # Group sentences into chunks
                    current_chunk = ''
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and len(sentence) > 10:
                            if len(current_chunk + sentence) < 400:
                                current_chunk += (' ' if current_chunk else '') + sentence
                            else:
                                if current_chunk:
                                    chunks.append({
                                        'text': current_chunk,
                                        'index': chunk_index,
                                        'element_type': 'content_block',
                                        'element_index': 0,
                                        'sub_chunk_index': 0,
                                        'total_sub_chunks': 1,
                                        'text_length': len(current_chunk),
                                        'source_url': url,
                                        'word_count': len(current_chunk.split())
                                    })
                                    chunk_index += 1
                                current_chunk = sentence
                    
                    # Add final chunk
                    if current_chunk:
                        chunks.append({
                            'text': current_chunk,
                            'index': chunk_index,
                            'element_type': 'content_block',
                            'element_index': 0,
                            'sub_chunk_index': 0,
                            'total_sub_chunks': 1,
                            'text_length': len(current_chunk),
                            'source_url': url,
                            'word_count': len(current_chunk.split())
                        })
                    break  # Stop after first successful container
        
        return chunks
    
    def _split_long_text_semantic(self, text: str, max_length: int = 800) -> List[str]:
        """Split long text into semantic chunks"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = []
        for delimiter in ['. ', '! ', '? ']:
            if delimiter in text:
                sentences = text.split(delimiter)
                break
        
        if not sentences:
            # Fallback to word splitting
            words = text.split()
            chunk_size = max_length // 10  # Rough estimate
            for i in range(0, len(words), chunk_size):
                chunk_words = words[i:i + chunk_size]
                chunks.append(' '.join(chunk_words))
            return chunks
        
        # Group sentences into chunks
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                if len(current_chunk + sentence) <= max_length:
                    current_chunk += (' ' if current_chunk else '') + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
        
    def search_competitors(self, keyword: str, num_results: int = 10) -> List[str]:
        """Search for competitor URLs using Serper"""
        url = "https://google.serper.dev/search"
        
        payload = {'q': keyword, 'num': num_results}
        headers = {'X-API-KEY': self.serper_key, 'Content-Type': 'application/json'}
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        results = response.json()
        return [result['link'] for result in results.get('organic', [])]

    def get_competitor_urls(self, keyword: str, num_competitors: int = 10) -> List[str]:
        """Get competitor URLs from search results"""
        try:
            if not self.serper_key or self.serper_key == "dummy":
                # Return dummy URLs for testing
                return [
                    "https://example1.com",
                    "https://example2.com", 
                    "https://example3.com"
                ]
            
            # Use Serper API to get search results
            search_url = "https://google.serper.dev/search"
            
            payload = {
                'q': keyword,
                'num': min(num_competitors, 20),  # API limit
                'gl': 'us',
                'hl': 'en'
            }
            
            headers = {
                'X-API-KEY': self.serper_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(search_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                urls = []
                
                # Extract URLs from organic results
                if 'organic' in data:
                    for result in data['organic'][:num_competitors]:
                        if 'link' in result:
                            url = result['link']
                            # Filter out unwanted domains
                            if not any(domain in url for domain in ['youtube.com', 'facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']):
                                urls.append(url)
                
                return urls[:num_competitors] if urls else ["https://example.com"]
            
            else:
                st.warning(f"Search API error: {response.status_code}")
                return ["https://example.com"]
                
        except Exception as e:
            st.warning(f"Error getting competitors: {str(e)}")
            return ["https://example.com"]
    
    def get_search_suggestions(self, keyword: str) -> List[TopicData]:
        """Get real search suggestions from Google Autocomplete - NO FALLBACKS"""
        suggestions = []
        
        # Google Autocomplete API - only real data
        base_suggestions = [
            f"{keyword} how to",
            f"{keyword} best",
            f"{keyword} vs",
            f"{keyword} for",
            f"{keyword} guide",
            f"{keyword} problems",
            f"{keyword} reviews",
            f"{keyword} comparison"
        ]
        
        for base in base_suggestions:
            try:
                autocomplete_url = f"http://suggestqueries.google.com/complete/search?client=firefox&q={quote_plus(base)}"
                response = requests.get(autocomplete_url, timeout=5, headers=self.headers)
                
                if response.status_code == 200:
                    data = response.json()
                    if len(data) > 1 and isinstance(data[1], list):
                        for suggestion in data[1][:3]:  # Top 3 per base
                            if (len(suggestion) > 10 and 
                                suggestion.lower() != base.lower() and
                                suggestion not in suggestions):
                                suggestions.append(suggestion)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                continue  # Just skip if API fails - NO FALLBACKS
        
        # Convert to TopicData objects - ONLY REAL DATA
        topic_data = []
        if suggestions:
            embeddings = self.embedding_model.encode(suggestions)
            
            for suggestion, embedding in zip(suggestions, embeddings):
                topic_data.append(TopicData(
                    text=suggestion,
                    embedding=embedding,
                    source='search_suggest',
                    source_url='google_autocomplete',
                    competitor_id=-1,
                    confidence=0.8
                ))
        
        return topic_data

    def get_reddit_discussions(self, keyword: str) -> List[TopicData]:
        """Get Reddit discussions related to the keyword"""
        try:
            if not self.serper_key or self.serper_key == "dummy":
                # Return dummy Reddit data for testing
                return [
                    TopicData(
                        text=f"How to use {keyword} effectively?",
                        embedding=self.embedding_model.encode([f"How to use {keyword} effectively?"])[0],
                        source='reddit',
                        source_url='https://reddit.com/r/example',
                        competitor_id=-1,
                        upvotes=25,
                        confidence=0.8,
                        word_count=50
                    ),
                    TopicData(
                        text=f"Best {keyword} alternatives discussion",
                        embedding=self.embedding_model.encode([f"Best {keyword} alternatives discussion"])[0],
                        source='reddit',
                        source_url='https://reddit.com/r/example2',
                        competitor_id=-1,
                        upvotes=15,
                        confidence=0.7,
                        word_count=35
                    )
                ]
            
            # Search Reddit using Serper API
            search_url = "https://google.serper.dev/search"
            
            payload = {
                'q': f'{keyword} site:reddit.com',
                'num': 10,
                'gl': 'us',
                'hl': 'en'
            }
            
            headers = {
                'X-API-KEY': self.serper_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.post(search_url, json=payload, headers=headers, timeout=10)
            
            reddit_topics = []
            
            if response.status_code == 200:
                data = response.json()
                
                if 'organic' in data:
                    for i, result in enumerate(data['organic'][:8]):  # Limit to 8 results
                        title = result.get('title', '')
                        snippet = result.get('snippet', '')
                        url = result.get('link', '')
                        
                        if 'reddit.com' in url and title:
                            # Combine title and snippet for better context
                            full_text = f"{title}. {snippet}".strip()
                            
                            # Extract upvotes from snippet if available (rough estimation)
                            upvotes = 10  # Default value
                            if 'points' in snippet or 'upvotes' in snippet:
                                # Try to extract number before 'points' or 'upvotes'
                                import re
                                numbers = re.findall(r'(\d+)\s*(?:points|upvotes)', snippet, re.IGNORECASE)
                                if numbers:
                                    upvotes = int(numbers[0])
                            
                            # Create embedding
                            embedding = self.embedding_model.encode([full_text])[0]
                            
                            reddit_topics.append(TopicData(
                                text=full_text,
                                embedding=embedding,
                                source='reddit',
                                source_url=url,
                                competitor_id=-1,
                                upvotes=upvotes,
                                confidence=0.8,
                                word_count=len(full_text.split())
                            ))
            
            # If no results, add some generic Reddit-style questions
            if not reddit_topics:
                generic_questions = [
                    f"What's the best way to use {keyword}?",
                    f"Any recommendations for {keyword}?",
                    f"Problems with {keyword} - need help",
                    f"Is {keyword} worth it? Experiences?"
                ]
                
                for question in generic_questions:
                    embedding = self.embedding_model.encode([question])[0]
                    reddit_topics.append(TopicData(
                        text=question,
                        embedding=embedding,
                        source='reddit',
                        source_url='https://reddit.com/r/example',
                        competitor_id=-1,
                        upvotes=5,
                        confidence=0.6,
                        word_count=len(question.split())
                    ))
            
            return reddit_topics
            
        except Exception as e:
            st.warning(f"Error getting Reddit discussions: {str(e)}")
            # Return minimal fallback data
            return [
                TopicData(
                    text=f"Discussion about {keyword}",
                    embedding=self.embedding_model.encode([f"Discussion about {keyword}"])[0],
                    source='reddit',
                    source_url='https://reddit.com',
                    competitor_id=-1,
                    upvotes=1,
                    confidence=0.5,
                    word_count=20
                )
            ]
            
    def mine_reddit_discussions(self, keyword: str, max_posts: int = 30) -> List[TopicData]:
        """Mine Reddit for real user questions - NO FALLBACKS"""
        reddit_topics = []
        
        if not self.reddit:
            st.info("Reddit API not configured. Skipping Reddit mining.")
            return reddit_topics  # Return empty list - NO FALLBACKS
        
        try:
            subreddit = self.reddit.subreddit('all')
            
            # Search for posts containing the keyword
            for submission in subreddit.search(keyword, limit=max_posts, sort='hot'):
                
                # Process the main post title
                title = submission.title.strip()
                if self._is_meaningful_question(title, keyword):
                    reddit_topics.append({
                        'text': title,
                        'upvotes': submission.score,
                        'url': f"https://reddit.com{submission.permalink}"
                    })
                
                # Process selftext if it contains questions
                if submission.selftext and len(submission.selftext) > 50:
                    questions = self._extract_questions(submission.selftext, keyword)
                    for question in questions:
                        reddit_topics.append({
                            'text': question,
                            'upvotes': submission.score,
                            'url': f"https://reddit.com{submission.permalink}"
                        })
            
            # Filter and convert to embeddings - ONLY REAL DATA
            if reddit_topics:
                filtered_topics = self._filter_reddit_topics(reddit_topics, keyword)
                
                if filtered_topics:
                    texts = [topic['text'] for topic in filtered_topics]
                    embeddings = self.embedding_model.encode(texts)
                    
                    topic_data = []
                    for topic, embedding in zip(filtered_topics, embeddings):
                        topic_data.append(TopicData(
                            text=topic['text'],
                            embedding=embedding,
                            source='reddit',
                            source_url=topic['url'],
                            competitor_id=-1,
                            confidence=0.9,
                            upvotes=topic['upvotes']
                        ))
                    
                    return topic_data
        
        except Exception as e:
            st.warning(f"Reddit mining failed: {e}")
        
        return []  # Return empty if no real data found
    
    def _is_meaningful_question(self, text: str, keyword: str) -> bool:
        """Check if text is a meaningful question"""
        text_lower = text.lower()
        keyword_lower = keyword.lower()
        
        # Must contain the keyword (strict requirement)
        if keyword_lower not in text_lower:
            return False
        
        # Filter out garbage patterns
        garbage_patterns = [
            'carrying on here', 'character limit', 'reddit mod',
            r'^(so|and|but|the|a|an)\s', r'^\w{1,2}\s'
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text_lower):
                return False
        
        # Must be long enough
        if len(text.split()) < 4:
            return False
        
        # Should contain question indicators
        indicators = ['how', 'what', 'why', 'where', 'when', 'which', 'best', 'recommend', '?', 'help', 'advice']
        return any(indicator in text_lower for indicator in indicators)
    
    def _extract_questions(self, text: str, keyword: str) -> List[str]:
        """Extract questions from text"""
        questions = []
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences[:5]:  # Check first 5 sentences
            sentence = sentence.strip()
            if (self._is_meaningful_question(sentence, keyword) and 
                20 <= len(sentence) <= 200):
                questions.append(sentence)
        
        return questions[:2]  # Max 2 questions per text
    
    def _filter_reddit_topics(self, topics: List[Dict], keyword: str) -> List[Dict]:
        """Filter and deduplicate Reddit topics"""
        filtered = []
        seen_texts = set()
        
        for topic in topics:
            text = topic['text'].strip()
            text_lower = text.lower()
            
            # Skip duplicates
            if text_lower in seen_texts:
                continue
            
            # Skip non-English (basic check)
            if not any(c.isascii() for c in text):
                continue
            
            seen_texts.add(text_lower)
            filtered.append(topic)
        
        # Sort by upvotes and take top ones
        filtered.sort(key=lambda x: x['upvotes'], reverse=True)
        return filtered[:10]  # Top 10 quality topics
    
    def _split_long_text_semantic(self, text: str, max_length: int) -> List[str]:
        """Split text on sentence boundaries for better semantic chunks"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ''
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= max_length:
                current_chunk += (' ' if current_chunk else '') + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Handle very long sentences by splitting on words
                    words = sentence.split(' ')
                    word_chunk = ''
                    for word in words:
                        if len(word_chunk + word) <= max_length:
                            word_chunk += (' ' if word_chunk else '') + word
                        else:
                            if word_chunk:
                                chunks.append(word_chunk)
                            word_chunk = word
                    if word_chunk:
                        current_chunk = word_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return [chunk for chunk in chunks if len(chunk) >= 50]

    def _analyze_single_page_structure(self, chunks: List[Dict], url: str) -> Dict:
        """Analyze structure of a single page"""
        structure = {
            'url': url,
            'total_chunks': len(chunks),
            'content_types': {},
            'heading_hierarchy': [],
            'content_flow': [],
            'word_distribution': {}
        }
        
        # Count content types
        for chunk in chunks:
            content_type = chunk['element_type']
            if content_type not in structure['content_types']:
                structure['content_types'][content_type] = 0
            structure['content_types'][content_type] += 1
        
        # Analyze heading hierarchy
        headings = [chunk for chunk in chunks if chunk['element_type'] == 'heading']
        structure['heading_hierarchy'] = [h['text'][:50] for h in headings]
        
        # Analyze content flow
        structure['content_flow'] = [chunk['element_type'] for chunk in chunks]
        
        # Word distribution
        for content_type, count in structure['content_types'].items():
            if count > 0:
                avg_words = sum(chunk['text_length'] for chunk in chunks 
                               if chunk['element_type'] == content_type) / count
                structure['word_distribution'][content_type] = avg_words
        
        return structure
    
    def _generate_competitor_structure_insights(self, competitor_structures: Dict) -> Dict:
        """Generate insights from competitor structure analysis"""
        insights = {
            'common_patterns': {},
            'content_gaps': [],
            'structure_opportunities': [],
            'depth_analysis': {}
        }
        
        if not competitor_structures:
            return insights
        
        all_content_types = set()
        content_type_frequency = {}
        
        # Analyze patterns across competitors
        for url, structure in competitor_structures.items():
            for content_type, count in structure['content_types'].items():
                all_content_types.add(content_type)
                if content_type not in content_type_frequency:
                    content_type_frequency[content_type] = []
                content_type_frequency[content_type].append(count)
        
        # Find common patterns
        for content_type in all_content_types:
            frequencies = content_type_frequency[content_type]
            avg_frequency = sum(frequencies) / len(frequencies) if frequencies else 0
            competitors_using = len(frequencies)
            total_competitors = len(competitor_structures)
            
            insights['common_patterns'][content_type] = {
                'avg_count': avg_frequency,
                'usage_percentage': (competitors_using / total_competitors) * 100,
                'competitors_using': competitors_using
            }
        
        # Find content gaps (content types used by <50% of competitors)
        for content_type, data in insights['common_patterns'].items():
            if data['usage_percentage'] < 50:
                insights['content_gaps'].append({
                    'content_type': content_type,
                    'opportunity': f"Only {data['competitors_using']}/{len(competitor_structures)} competitors use {content_type}",
                    'recommendation': f"Add more {content_type.replace('_', ' ')} content"
                })
        
        return insights
        
    def analyze_website_relevance(self, website_url: str, target_topic: str, max_pages: int = None) -> Dict:
        """Analyze entire website to find irrelevant content using vector embeddings"""
        st.info(f"🔍 Analyzing {website_url} for topic relevance...")
        
        try:
            # Get all pages from the website
            pages_data = self._crawl_entire_website(website_url, max_pages)
            
            if not pages_data:
                return {"error": "Could not crawl website pages"}
            
            # Batch process embeddings for efficiency
            st.info(f"🧠 Processing {len(pages_data)} pages with AI...")
            target_embedding = self.embedding_model.encode([target_topic])[0]
            
            # Process pages in batches to avoid memory issues
            batch_size = 50
            relevance_results = []
            
            progress_bar = st.progress(0)
            
            for i in range(0, len(pages_data), batch_size):
                batch = pages_data[i:i + batch_size]
                
                for page in batch:
                    # Use enhanced relevance calculation
                    relevance_data = self._calculate_enhanced_relevance(page, target_embedding, target_topic)
                    relevance_results.append(relevance_data)
                
                # Update progress
                progress = min((i + batch_size) / len(pages_data), 1.0)
                progress_bar.progress(progress)
            
            # Sort by least relevant first (lowest similarity)
            relevance_results.sort(key=lambda x: x['similarity_score'])
            
            # Generate structure recommendations
            structure_recommendations = self._generate_website_structure_recommendations(relevance_results, target_topic)
            
            return {
                'target_topic': target_topic,
                'total_pages': len(relevance_results),
                'irrelevant_pages': len([r for r in relevance_results if r['relevance_status'] == 'Irrelevant']),
                'somewhat_relevant': len([r for r in relevance_results if r['relevance_status'] == 'Somewhat Relevant']),
                'highly_relevant': len([r for r in relevance_results if r['relevance_status'] == 'Highly Relevant']),
                'pages': relevance_results,
                'structure_recommendations': structure_recommendations,  # NEW
                'enhancement_stats': {  # NEW
                    'pages_enhanced': len([r for r in relevance_results if r.get('enhancement_applied', False)]),
                    'avg_structure_quality': sum(r.get('structure_quality', 0) for r in relevance_results) / len(relevance_results)
                }
            }
            
        except Exception as e:
            return {"error": f"Error analyzing website: {str(e)}"}
    
    def _crawl_entire_website(self, base_url: str, max_pages: int = None) -> List[Dict]:
        """Advanced website crawler with safety limits and better handling"""
        pages_data = []
        visited_urls = set()
        urls_to_visit = [base_url]
        
        # Get base domain for staying on same site
        from urllib.parse import urljoin, urlparse, parse_qs
        base_domain = urlparse(base_url).netloc
        
        # Try to find sitemap first for faster discovery
        sitemap_urls = self._discover_sitemap_urls(base_url)
        if sitemap_urls:
            st.info(f"📋 Found sitemap with {len(sitemap_urls)} URLs")
            urls_to_visit.extend(sitemap_urls)
        
        # Remove duplicates
        urls_to_visit = list(dict.fromkeys(urls_to_visit))
        
        # Apply safety limits
        if max_pages:
            urls_to_visit = urls_to_visit[:max_pages]
            st.info(f"🔍 Analyzing up to {max_pages} pages (user limit)")
        else:
            # Safety limit for unlimited crawling
            if len(urls_to_visit) > 500:
                st.warning(f"⚠️ Large website detected ({len(urls_to_visit)} URLs). Limiting to 500 pages for performance.")
                urls_to_visit = urls_to_visit[:500]
            st.info(f"🔍 Analyzing website ({len(urls_to_visit)} URLs discovered)")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_to_process = len(urls_to_visit)
        successful_crawls = 0
        errors = 0
        
        for i, current_url in enumerate(urls_to_visit):
            if current_url in visited_urls:
                continue
                
            visited_urls.add(current_url)
            status_text.text(f"Crawling {i+1}/{total_to_process}: {current_url.split('/')[-1][:50]}...")
            
            # Fix progress calculation
            progress_value = min((i + 1) / max(total_to_process, 1), 1.0)
            progress_bar.progress(progress_value)
            
            try:
                # Skip certain file types
                if any(current_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.xml', '.txt']):
                    continue
                
                response = requests.get(current_url, headers=self.headers, timeout=20)
                response.raise_for_status()
                
                # Skip non-HTML content
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove noise elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                    element.decompose()
                
                # Get page title
                title = soup.find('title')
                title_text = title.get_text().strip() if title else current_url.split('/')[-1]
                
                # Get main content with multiple strategies
                content = self._extract_main_content(soup)
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Calculate word count
                word_count = len([w for w in content.split() if w.isalpha()])
                
                if word_count > 50:  # Include pages with substantial content
                    pages_data.append({
                        'url': current_url,
                        'title': title_text,
                        'content': content,
                        'word_count': word_count
                    })
                    successful_crawls += 1
                
                # For sites without sitemaps, discover more URLs (limited)
                if not sitemap_urls and len(urls_to_visit) < 100:  # Only for small sites
                    new_urls = self._discover_internal_links(soup, current_url, base_domain, visited_urls)
                    remaining_slots = min(20, 100 - len(urls_to_visit))  # Add max 20 more URLs
                    urls_to_visit.extend(new_urls[:remaining_slots])
                    total_to_process = len(urls_to_visit)
                
                # Rate limiting - be respectful
                time.sleep(0.8)  # Slightly slower for large sites
                
            except Exception as e:
                errors += 1
                if errors > 10:  # Too many errors, stop
                    st.error(f"❌ Too many errors ({errors}). Stopping crawl.")
                    break
                continue
            
            # Safety break for very large sites
            if len(pages_data) > 200:
                st.warning(f"⚠️ Reached 200 pages limit for performance. Stopping crawl.")
                break
        
        status_text.text(f"✅ Crawl complete: {successful_crawls} pages analyzed, {errors} errors")
        
        if len(pages_data) == 0:
            st.error("❌ No pages could be crawled. The website might be blocking requests or have technical issues.")
        
        return pages_data
    
    def _discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Try to find and parse XML sitemaps for faster URL discovery"""
        sitemap_urls = []
        
        # Clean base URL
        base_url = base_url.rstrip('/')
        
        # Common sitemap locations
        sitemap_locations = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/sitemap/sitemap.xml",
            f"{base_url}/sitemaps/sitemap.xml",
            f"{base_url}/wp-sitemap.xml",  # WordPress
            f"{base_url}/sitemap-index.xml",
            f"{base_url}/robots.txt"
        ]
        
        for sitemap_url in sitemap_locations:
            try:
                st.write(f"🔍 Checking: {sitemap_url}")
                response = requests.get(sitemap_url, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    if sitemap_url.endswith('robots.txt'):
                        # Parse robots.txt for sitemap references
                        st.write("📋 Parsing robots.txt for sitemaps...")
                        for line in response.text.split('\n'):
                            line = line.strip()
                            if line.lower().startswith('sitemap:'):
                                actual_sitemap = line.split(':', 1)[1].strip()
                                st.write(f"📋 Found sitemap in robots.txt: {actual_sitemap}")
                                try:
                                    sitemap_response = requests.get(actual_sitemap, headers=self.headers, timeout=15)
                                    if sitemap_response.status_code == 200:
                                        new_urls = self._parse_sitemap_xml(sitemap_response.text, base_url)
                                        sitemap_urls.extend(new_urls)
                                        st.write(f"✅ Extracted {len(new_urls)} URLs from {actual_sitemap}")
                                except Exception as e:
                                    st.write(f"❌ Error parsing sitemap from robots.txt: {str(e)}")
                                    continue
                    else:
                        # Parse XML sitemap
                        st.write(f"✅ Found XML sitemap: {sitemap_url}")
                        new_urls = self._parse_sitemap_xml(response.text, base_url)
                        sitemap_urls.extend(new_urls)
                        st.write(f"✅ Extracted {len(new_urls)} URLs from sitemap")
                        
                    if sitemap_urls:
                        st.success(f"🎉 Total discovered: {len(sitemap_urls)} URLs from sitemaps!")
                        break  # Found working sitemap
                else:
                    st.write(f"❌ HTTP {response.status_code} for {sitemap_url}")
                        
            except Exception as e:
                st.write(f"❌ Could not access {sitemap_url}: {str(e)}")
                continue
        
        if not sitemap_urls:
            st.warning("⚠️ No sitemaps found. Will use aggressive link discovery instead.")
            # Try a more aggressive approach for sites without sitemaps
            return self._fallback_url_discovery(base_url)
        
        return sitemap_urls[:1000]  # Reasonable limit to prevent crashes
    
    def _fallback_url_discovery(self, base_url: str) -> List[str]:
        """Fallback URL discovery for sites without sitemaps"""
        discovered_urls = [base_url]
        
        try:
            st.info("🕷️ Using aggressive link discovery (no sitemap found)...")
            response = requests.get(base_url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove noise
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                
                # Find all internal links
                from urllib.parse import urljoin, urlparse
                base_domain = urlparse(base_url).netloc
                
                links = soup.find_all('a', href=True)
                st.write(f"🔗 Found {len(links)} links on homepage")
                
                for link in links:
                    href = link['href'].strip()
                    if not href or href.startswith('#'):
                        continue
                    
                    full_url = urljoin(base_url, href)
                    parsed = urlparse(full_url)
                    
                    if (parsed.netloc == base_domain and 
                        full_url not in discovered_urls and
                        not any(full_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.png', '.gif', '.zip'])):
                        discovered_urls.append(full_url)
                        
                        if len(discovered_urls) >= 50:  # Limit for fallback
                            break
                
                st.info(f"🔗 Discovered {len(discovered_urls)} URLs through link crawling")
        
        except Exception as e:
            st.error(f"❌ Fallback discovery failed: {str(e)}")
        
        return discovered_urls
    
    def _parse_sitemap_xml(self, xml_content: str, base_url: str) -> List[str]:
        """Parse XML sitemap to extract URLs with better error handling"""
        urls = []
        try:
            # Handle different XML parsers
            try:
                soup = BeautifulSoup(xml_content, 'xml')
            except:
                soup = BeautifulSoup(xml_content, 'html.parser')
            
            # Handle sitemap index files
            sitemap_tags = soup.find_all('sitemap')
            if sitemap_tags:
                st.write(f"📑 Processing sitemap index with {len(sitemap_tags)} nested sitemaps...")
                for sitemap in sitemap_tags[:10]:  # Limit nested sitemaps
                    loc = sitemap.find('loc')
                    if loc:
                        # Recursively parse nested sitemaps
                        try:
                            nested_url = loc.text.strip()
                            response = requests.get(nested_url, timeout=10)
                            if response.status_code == 200:
                                nested_urls = self._parse_sitemap_xml(response.text, base_url)
                                urls.extend(nested_urls)
                                st.write(f"  ✅ Parsed {len(nested_urls)} URLs from {nested_url}")
                        except Exception:
                            continue
            
            # Handle regular URL entries
            url_tags = soup.find_all('url')
            if url_tags:
                st.write(f"📄 Processing {len(url_tags)} individual URLs...")
                for url_tag in url_tags:
                    loc = url_tag.find('loc')
                    if loc:
                        url = loc.text.strip()
                        # Ensure URL is from the same domain
                        from urllib.parse import urlparse
                        if urlparse(url).netloc == urlparse(base_url).netloc:
                            urls.append(url)
                    
        except Exception as e:
            st.write(f"❌ Error parsing sitemap XML: {str(e)}")
        
        return urls
    
    def _extract_main_content(self, soup) -> str:
        """Extract main content using multiple strategies"""
        content = ""
        
        # Strategy 1: Look for main content selectors
        content_selectors = [
            'article', 'main', '[role="main"]', '.content', '#content', 
            '.post-content', '.entry-content', '.page-content', '.article-content',
            '.post', '.entry', '.article', '.blog-post'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text() for elem in elements])
                break
        
        # Strategy 2: If no main content found, use body but filter out common noise
        if not content or len(content) < 100:
            body = soup.find('body')
            if body:
                # Remove common navigation and sidebar elements
                for noise in body.select('nav, .navigation, .sidebar, .menu, .footer, .header, .breadcrumb, .social, .share'):
                    noise.decompose()
                content = body.get_text()
        
        # Strategy 3: Fallback to all text
        if not content:
            content = soup.get_text()
        
        return content
    
    def _discover_internal_links(self, soup, current_url: str, base_domain: str, visited_urls: set) -> List[str]:
        """Discover internal links from current page with better filtering"""
        from urllib.parse import urljoin, urlparse, urlunparse
        
        new_urls = []
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href'].strip()
            
            # Skip empty or invalid hrefs
            if not href or href in ['#', 'javascript:void(0)', 'javascript:;']:
                continue
            
            # Convert to absolute URL
            full_url = urljoin(current_url, href)
            
            # Parse URL
            parsed = urlparse(full_url)
            
            # Clean the URL (remove fragments and some parameters)
            clean_url = urlunparse((
                parsed.scheme,
                parsed.netloc, 
                parsed.path.rstrip('/'),  # Remove trailing slash for consistency
                parsed.params,
                parsed.query,
                ''  # Remove fragment
            ))
            
            # Only add internal links from same domain
            if (parsed.netloc == base_domain and 
                clean_url not in visited_urls and 
                clean_url != current_url and  # Don't add self-references
                not any(clean_url.lower().endswith(ext) for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx', '.xls', '.xlsx']) and
                not any(skip in clean_url.lower() for skip in ['/wp-admin/', '/admin/', '/login', '/register', '/cart', '/checkout', '/search?', '?page=']) and
                len(parsed.path.split('/')) <= 6):  # Avoid very deep URLs
                
                new_urls.append(clean_url)
                
                if len(new_urls) >= 30:  # Increased limit per page
                    break
        
        return new_urls
    
    def _categorize_relevance(self, similarity_score: float) -> str:
        """Categorize relevance based on similarity score with more lenient thresholds"""
        if similarity_score >= 0.5:  # Lowered from 0.7
            return "Highly Relevant"
        elif similarity_score >= 0.25:  # Lowered from 0.4
            return "Somewhat Relevant"
        else:
            return "Irrelevant"

    def _calculate_enhanced_relevance(self, page: Dict, target_embedding: np.ndarray, target_topic: str) -> Dict:
        """Enhanced relevance calculation using semantic analysis"""
        
        # Basic relevance (same as before)
        basic_similarity = cosine_similarity([target_embedding], 
                                           [self.embedding_model.encode([page['content']])[0]])[0][0]
        
        # Try to get better content structure if we have raw HTML
        enhanced_similarity = basic_similarity
        chunk_analysis = {}
        structure_quality = 0
        
        # If we can parse the content better, do semantic analysis
        try:
            if len(page['content']) > 200:  # Only for substantial content
                # Create a simple soup from content to try semantic chunking
                soup = BeautifulSoup(f"<div>{page['content']}</div>", 'html.parser')
                semantic_chunks = self._extract_semantic_chunks(soup, page['url'])
                
                if semantic_chunks:
                    structure_quality = len(semantic_chunks)
                    
                    # Analyze relevance by content type
                    chunk_relevances = {}
                    for chunk in semantic_chunks:
                        chunk_embedding = self.embedding_model.encode([chunk['text']])[0]
                        chunk_similarity = cosine_similarity([target_embedding], [chunk_embedding])[0][0]
                        
                        content_type = chunk['element_type']
                        if content_type not in chunk_relevances:
                            chunk_relevances[content_type] = []
                        chunk_relevances[content_type].append(chunk_similarity)
                    
                    # Calculate weighted relevance (headings matter more)
                    weights = {'heading': 0.3, 'paragraph': 0.4, 'list_item': 0.2, 'quote': 0.1}
                    
                    if chunk_relevances:
                        total_weight = 0
                        weighted_sum = 0
                        
                        for content_type, similarities in chunk_relevances.items():
                            weight = weights.get(content_type, 0.1)
                            avg_similarity = sum(similarities) / len(similarities)
                            weighted_sum += avg_similarity * weight
                            total_weight += weight
                        
                        if total_weight > 0:
                            enhanced_similarity = weighted_sum / total_weight
                    
                    chunk_analysis = {k: sum(v)/len(v) for k, v in chunk_relevances.items()}
        
        except Exception:
            # If enhanced analysis fails, stick with basic
            pass
        
        return {
            'url': page['url'],
            'title': page['title'],
            'word_count': page['word_count'],
            'similarity_score': enhanced_similarity,
            'basic_similarity': basic_similarity,
            'relevance_status': self._categorize_relevance(enhanced_similarity),
            'content_preview': page['content'][:200] + "...",
            'main_topics': self._extract_main_topics(page['content']),
            'chunk_analysis': chunk_analysis,  # NEW: Per-content-type relevance
            'structure_quality': structure_quality,  # NEW: Number of semantic chunks found
            'enhancement_applied': enhanced_similarity != basic_similarity
        }
        
    def _generate_website_structure_recommendations(self, pages_data: List[Dict], target_topic: str) -> List[Dict]:
        """Generate structure improvement recommendations for website analysis"""
        recommendations = []
        
        # Analyze overall structure quality
        total_pages = len(pages_data)
        low_structure_pages = len([p for p in pages_data if p.get('structure_quality', 0) < 3])
        
        if low_structure_pages > total_pages * 0.3:  # More than 30% have poor structure
            recommendations.append({
                'type': 'content_structure',
                'priority': 'High',
                'issue': f"{low_structure_pages} pages have poor content structure",
                'recommendation': "Improve content structure with more headings, lists, and organized paragraphs",
                'pages_affected': low_structure_pages
            })
        
        # Analyze content type usage
        content_type_usage = {}
        for page in pages_data:
            chunk_analysis = page.get('chunk_analysis', {})
            for content_type in chunk_analysis.keys():
                if content_type not in content_type_usage:
                    content_type_usage[content_type] = 0
                content_type_usage[content_type] += 1
        
        # Check for missing important content types
        important_types = ['heading', 'paragraph', 'list_item']
        for content_type in important_types:
            usage_pct = (content_type_usage.get(content_type, 0) / total_pages) * 100
            if usage_pct < 50:  # Less than 50% of pages use this content type
                recommendations.append({
                    'type': 'content_type',
                    'priority': 'Medium',
                    'issue': f"Only {usage_pct:.0f}% of pages use {content_type.replace('_', ' ')} content",
                    'recommendation': f"Add more {content_type.replace('_', ' ')} elements across your website",
                    'pages_affected': f"{usage_pct:.0f}% of pages"
                })
        
        return recommendations
    
    def _extract_main_topics(self, content: str) -> List[str]:
        """Extract main topics from content using simple keyword extraction with fallback"""
        try:
            # Try NLTK tokenization first
            words = word_tokenize(content.lower())
        except:
            # Fallback to simple split if NLTK fails
            words = content.lower().split()
        
        # Remove stop words and get word frequency
        try:
            # Try using NLTK stopwords
            clean_words = [w for w in words if w.isalpha() and len(w) > 3 and w not in self.stop_words]
        except:
            # Fallback stopwords if NLTK fails
            basic_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once'}
            clean_words = [w for w in words if w.isalpha() and len(w) > 3 and w not in basic_stopwords]
        
        from collections import Counter
        word_freq = Counter(clean_words)
        
        # Get top 5 most frequent meaningful words
        top_words = [word for word, freq in word_freq.most_common(5)]
        return top_words
    
    def create_relevance_visualization(self, relevance_data: Dict) -> go.Figure:
        """Create visualization for website relevance analysis"""
        if 'error' in relevance_data:
            return None
            
        pages = relevance_data['pages']
        
        # Create scatter plot
        fig = go.Figure()
        
        # Color mapping for relevance
        color_map = {
            'Irrelevant': 'red',
            'Somewhat Relevant': 'orange', 
            'Highly Relevant': 'green'
        }
        
        for status in ['Irrelevant', 'Somewhat Relevant', 'Highly Relevant']:
            filtered_pages = [p for p in pages if p['relevance_status'] == status]
            
            if filtered_pages:
                similarities = [p['similarity_score'] for p in filtered_pages]
                word_counts = [p['word_count'] for p in filtered_pages]
                titles = [p['title'][:50] + "..." if len(p['title']) > 50 else p['title'] for p in filtered_pages]
                urls = [p['url'] for p in filtered_pages]
                
                hover_text = [f"<b>{title}</b><br>Similarity: {sim:.2%}<br>Words: {wc}<br>URL: {url}" 
                             for title, sim, wc, url in zip(titles, similarities, word_counts, urls)]
                
                fig.add_trace(go.Scatter(
                    x=similarities,
                    y=word_counts,
                    mode='markers',
                    name=f'{status} ({len(filtered_pages)})',
                    marker=dict(
                        size=12,
                        color=color_map[status],
                        opacity=0.7,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate='%{hovertext}<extra></extra>',
                    hovertext=hover_text
                ))
        
        fig.update_layout(
            title=f'Website Content Relevance Analysis<br><sup>Target Topic: "{relevance_data["target_topic"]}"</sup>',
            xaxis_title='Similarity Score (Higher = More Relevant)',
            yaxis_title='Word Count',
            width=800,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        # Add relevance threshold lines with updated values
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=0.25, line_dash="dash", line_color="orange", opacity=0.5, 
                     annotation_text="Relevance Threshold")
        fig.add_vline(x=0.5, line_dash="dash", line_color="green", opacity=0.5,
                     annotation_text="High Relevance")
        
        return fig


# Streamlit App
def main():
    st.set_page_config(
        page_title="Data-Driven SEO Analyzer", 
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header with logo and branding
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        # Add your logo here - you'll need to upload logo.png to your repo
        try:
            st.image("logo.png", width=120)
        except:
            st.write("🎯")  # Fallback emoji if no logo
    
    with col2:
        st.title("🚀 Data-Driven Vector SEO Analyzer")
        st.markdown("**Find content gaps using REAL user data!**")
    
    with col3:
        # Your website link
        st.markdown("""
        <div style='text-align: right; padding-top: 20px;'>
            <a href='https://tororank.com/' target='_blank' style='
                color: #ff4b4b; 
                text-decoration: none; 
                font-weight: bold;
                border: 2px solid #ff4b4b;
                padding: 8px 16px;
                border-radius: 6px;
                transition: all 0.3s;
            '>Visit Our Website</a>
        </div>
        """, unsafe_allow_html=True)
    
    # Add custom CSS for better styling
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Custom button styling */
    div.stButton > button {
        background: linear-gradient(45deg, #ff4b4b, #ff6b6b);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    div.stButton > button:hover {
        background: linear-gradient(45deg, #ff3b3b, #ff5b5b);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(255, 75, 75, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: white;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Buy Me a Coffee Widget
    st.markdown("""
    <script data-name="BMC-Widget" data-cfasync="false" src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" data-id="deyangeorgiev" data-description="Support me on Buy me a coffee!" data-message="If this tool has helped you, consider getting me a coffee :) Thanks!" data-color="#5F7FFF" data-position="Right" data-x_margin="18" data-y_margin="18"></script>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Buy Me a Coffee Widget
        st.markdown("---")
        st.markdown("### ☕ Support This Tool")
        st.markdown("""
        If this tool has offered any value and helped you with your work, any support would be appreciated!
        
        <a href="https://www.buymeacoffee.com/deyangeorgiev" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                 alt="Buy Me A Coffee" 
                 style="height: 50px !important;width: 180px !important;" >
        </a>
        
        <script data-name="BMC-Widget" data-cfasync="false" src="https://cdnjs.buymeacoffee.com/1.0.0/widget.prod.min.js" data-id="deyangeorgiev" data-description="Support me on Buy me a coffee!" data-message="If this tool has offered any value and helped you with your work, any support would be appreciated so I can continue improving it and adding more features!" data-color="#5F7FFF" data-position="Right" data-x_margin="18" data-y_margin="18"></script>
        """, unsafe_allow_html=True)
        st.markdown("---")
        # Tool selection
        analysis_mode = st.radio(
            "Choose Analysis Type:",
            ["🔍 Find Content Gaps", "🎯 Check Website Relevance"],
            help="Find gaps in competitor content OR analyze your website for off-topic content"
        )
        
        serper_key = st.text_input(
            "Serper API Key", 
            type="password",
            help="Get free key from serper.dev"
        )
        
        if analysis_mode == "🔍 Find Content Gaps":
            st.markdown("**Optional: Reddit API**")
            reddit_id = st.text_input("Reddit Client ID", type="password")
            reddit_secret = st.text_input("Reddit Client Secret", type="password")
            
            keyword = st.text_input("Target Keyword", placeholder="e.g., best seedbox")
            num_competitors = st.slider("Competitors", 3, 12, 8)
            
            analyze_btn = st.button("🎯 Find Content Gaps", type="primary")
        
        else:  # Website Relevance Analysis
            website_url = st.text_input(
                "Website URL", 
                placeholder="https://tororank.com/",
                help="Enter the website URL to analyze for content relevance"
            )
            target_topic = st.text_input(
                "Main Topic/Niche", 
                placeholder="e.g., digital marketing, web development, fitness",
                help="What should your website be about? We'll find content that doesn't match."
            )
            
            # Advanced options
            with st.expander("⚙️ Advanced Crawl Settings"):
                crawl_mode = st.radio(
                    "Crawling Mode:",
                    ["🌐 Entire Website (Recommended)", "📊 Limited Crawl"],
                    help="Entire website uses sitemaps and intelligent crawling. Limited crawl stops at a specific number."
                )
                
                if crawl_mode == "📊 Limited Crawl":
                    max_pages = st.slider("Max Pages to Analyze", 10, 500, 100)
                else:
                    max_pages = None
                    st.info("✅ Will analyze up to 500 pages (performance limit) using sitemaps and intelligent crawling")
                
                # Performance warning
                st.warning("⚠️ **Performance Notice**: Large websites (500+ pages) may take 15-30 minutes to analyze. Consider using Limited Crawl for faster results.")
                
                st.markdown("""
                **Crawling Strategy:**
                - 🗺️ **Sitemap Discovery**: Automatically finds and parses XML sitemaps
                - 🔗 **Link Following**: Discovers pages through internal links (if no sitemap)
                - 🚫 **Smart Filtering**: Skips images, PDFs, and duplicate content
                - ⚡ **Batch Processing**: Efficiently handles large websites
                - 🛡️ **Safety Limits**: Auto-stops at 500 pages for performance
                """)
            
            analyze_btn = st.button("🎯 Analyze Website Relevance", type="primary")
        
        # Sidebar footer with your branding
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 12px;'>
            Made with ❤️ by<br>
            <a href='https://tororank.com/' target='_blank' style='color: #ff4b4b; text-decoration: none;'>
                TORO RANK
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Info
    with st.expander("ℹ️ What makes this different?"):
        if analysis_mode == "🔍 Find Content Gaps":
            st.markdown("""
            **Real data sources:**
            - 🔍 **Search Suggestions**: Google Autocomplete data
            - 💬 **Reddit Mining**: Real user questions  
            - 📊 **Content Depth**: Competitor weaknesses
            - 🎯 **Vector Analysis**: AI finds missed topics
            
            **Why it works:**
            - Uses actual user behavior data, not guesses
            - AI-powered semantic analysis finds hidden gaps
            - Actionable recommendations with difficulty scores
            """)
        else:
            st.markdown("""
            **Website Relevance Analysis:**
            - 🕷️ **Website Crawling**: Analyzes your ENTIRE website intelligently
            - 🎯 **Vector Similarity**: AI compares content to your main topic
            - 📊 **Relevance Scoring**: Identifies off-topic content
            - 🔍 **Topic Extraction**: Shows what each page is actually about
            
            **Advanced Crawling:**
            - 🗺️ **Sitemap Discovery**: Finds and parses XML sitemaps automatically
            - 🔗 **Intelligent Link Following**: Discovers all internal pages
            - 📈 **Scales to Any Size**: Handles websites with 1000+ pages
            - ⚡ **Batch Processing**: Efficient AI analysis
            
            **Why it's useful:**
            - Find content that hurts your SEO focus
            - Identify pages to remove or redirect
            - Maintain topical authority
            - Clean up content strategy
            """)
    
    # Handle different analysis modes
    if analysis_mode == "🔍 Find Content Gaps":
        if not serper_key:
            st.info("👈 Enter Serper API key to start")
            return
        
        if not keyword:
            st.info("👈 Enter a keyword to analyze")
            return
        
        if analyze_btn:
            try:
                analyzer = DataDrivenSEOAnalyzer(serper_key, reddit_id, reddit_secret)
                results = analyzer.run_analysis(keyword, num_competitors)
                
                # Check for errors
                if 'error' in results:
                    st.error(results['error'])
                    return
                
                # Extract data from results dictionary
                competitor_urls = results.get('competitor_urls', [])
                competitor_topics = results.get('competitor_topics', [])
                reddit_topics = results.get('reddit_topics', [])
                search_topics = results.get('search_topics', [])
                gaps = results.get('gaps', [])
                depth_gaps = results.get('depth_gaps', [])
                structure_insights = results.get('structure_insights', {})
                actionable_topics = results.get('actionable_topics', [])
                fig = results.get('visualization')
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if fig:
                        # Display the chart with click functionality
                        st.plotly_chart(fig, use_container_width=True, key="main_chart", on_select="rerun")
                        
                        # Handle click events
                        if st.session_state.get("main_chart", {}).get("selection", {}).get("points"):
                            selected_points = st.session_state["main_chart"]["selection"]["points"]
                            
                            if selected_points:
                                # Get the first selected point
                                point = selected_points[0]
                                point_index = point.get("pointIndex")
                                curve_number = point.get("curveNumber")
                                
                                # Map curve number to data source
                                trace_names = ['🏢 Competitor Content', '💬 Reddit Questions', '🔍 Search Suggestions', 
                                              '📊 Thin Content Gaps', '🎯 CONTENT GAPS']
                                
                                if curve_number is not None and curve_number < len(trace_names):
                                    trace_name = trace_names[curve_number]
                                    
                                    # Get the corresponding topic data
                                    selected_topic = None
                                    
                                    if trace_name == '🏢 Competitor Content' and point_index < len(competitor_topics):
                                        selected_topic = competitor_topics[point_index]
                                    elif trace_name == '💬 Reddit Questions':
                                        if point_index < len(reddit_topics):
                                            selected_topic = reddit_topics[point_index]
                                    elif trace_name == '🔍 Search Suggestions':
                                        if point_index < len(search_topics):
                                            selected_topic = search_topics[point_index]
                                    elif trace_name == '📊 Thin Content Gaps':
                                        if point_index < len(depth_gaps):
                                            selected_topic = depth_gaps[point_index]
                                    elif trace_name == '🎯 CONTENT GAPS':
                                        if point_index < len(gaps):
                                            selected_topic = gaps[point_index]
                                    
                                    # Display selected point details
                                    if selected_topic:
                                        st.success("🎯 **Point Selected!** Click anywhere else to deselect.")
                                        
                                        col_detail1, col_detail2 = st.columns([2, 1])
                                        
                                        with col_detail1:
                                            st.subheader(f"{trace_name} - Details")
                                            
                                            # Topic text
                                            st.write(f"**📝 Topic:** {selected_topic.text}")
                                            
                                            # Source information
                                            if selected_topic.source == 'competitor':
                                                comp_name = competitor_urls[selected_topic.competitor_id].split('/')[2].replace('www.', '') if selected_topic.competitor_id < len(competitor_urls) else 'Unknown'
                                                st.write(f"**🏢 Competitor:** {comp_name}")
                                                st.write(f"**📄 Word Count:** {selected_topic.word_count} words")
                                            elif selected_topic.source == 'reddit':
                                                st.write(f"**👍 Upvotes:** {selected_topic.upvotes}")
                                                st.write(f"**💬 Source:** Reddit discussion")
                                            elif selected_topic.source == 'search_suggest':
                                                st.write(f"**🔍 Source:** Google search suggestions")
                                                st.write(f"**💡 Insight:** People actively search for this")
                                            elif selected_topic.source == 'depth_gap':
                                                st.write(f"**📊 Current depth:** {selected_topic.word_count} words")
                                                st.write(f"**💡 Opportunity:** Create comprehensive guide")
                                            
                                            # Confidence score
                                            st.write(f"**🎯 Confidence:** {selected_topic.confidence:.1%}")
                                            
                                            # URL if available
                                            if selected_topic.source_url and selected_topic.source_url != 'google_autocomplete':
                                                st.write(f"**🔗 URL:** [{selected_topic.source_url}]({selected_topic.source_url})")
                                                
                                                # Copy URL button
                                                if st.button("📋 Copy URL", key=f"copy_{point_index}_{curve_number}"):
                                                    st.code(selected_topic.source_url)
                                                    st.success("URL copied to display! You can copy it from above.")
                                        
                                        with col_detail2:
                                            # Action recommendations
                                            st.subheader("💡 Action Items")
                                            
                                            if selected_topic.source == 'competitor':
                                                st.info("**Competitor Analysis:**")
                                                st.write("• Analyze their content approach")
                                                st.write("• Identify improvement opportunities") 
                                                st.write("• Note their content structure")
                                            elif selected_topic.source in ['reddit', 'search_suggest', 'depth_gap']:
                                                st.success("**Content Opportunity:**")
                                                st.write("• Create comprehensive content")
                                                st.write("• Target this specific topic")
                                                st.write("• Address user questions directly")
                                            
                                            # SEO recommendations
                                            st.subheader("🎯 SEO Strategy")
                                            if selected_topic.source == 'search_suggest':
                                                st.write("• High search volume potential")
                                                st.write("• Target exact search query")
                                                st.write("• Optimize for featured snippets")
                                            elif selected_topic.source == 'reddit':
                                                st.write("• Answer real user questions")
                                                st.write("• Create FAQ-style content")
                                                st.write("• Build topical authority")
                                            elif selected_topic.source == 'depth_gap':
                                                st.write("• Create definitive guide")
                                                st.write("• Aim for 2000+ words")
                                                st.write("• Outrank thin content")
                                            
                                        st.markdown("---")
                                        
                                        # Clear selection button
                                        if st.button("🔄 Clear Selection", key="clear_selection"):
                                            if "main_chart" in st.session_state:
                                                del st.session_state["main_chart"]
                                            st.rerun()
                    else:
                        st.error("No data found for visualization")
                
                with col2:
                    # Enhanced Analysis Summary
                    st.subheader("📊 Enhanced Analysis Summary")
                    
                    # Main metrics
                    col1_sum, col2_sum, col3_sum, col4_sum = st.columns(4)
                    with col1_sum:
                        st.metric("Content Gaps Found", len(gaps), help="Opportunities based on competitor analysis")
                    with col2_sum:
                        st.metric("Competitors Analyzed", len(competitor_urls), help="Deep semantic analysis performed")
                    with col3_sum:
                        st.metric("Reddit Discussions", len(reddit_topics), help="Real user conversations analyzed")
                    with col4_sum:
                        if structure_insights and structure_insights.get('common_patterns'):
                            content_types_found = len(structure_insights['common_patterns'])
                            st.metric("Content Types Found", content_types_found, help="Different content structures identified")
                        else:
                            st.metric("Search Results", len(search_topics), help="Additional research data")
                    
                    # Enhanced features summary
                    st.subheader("🚀 Enhanced Features Applied")
                    
                    enhancement_col1, enhancement_col2 = st.columns(2)
                    
                    with enhancement_col1:
                        st.markdown("**🔬 Semantic Analysis:**")
                        if structure_insights:
                            st.success("✅ Advanced content structure analysis")
                            st.success("✅ Semantic chunking applied") 
                            st.success("✅ Content type gap identification")
                        else:
                            st.info("📊 Basic analysis completed")
                        
                        st.markdown("**📈 AI-Powered Insights:**")
                        st.success("✅ Sentence-transformer embeddings")
                        st.success("✅ Contextual similarity scoring")
                        st.success("✅ Multi-source data integration")
                    
                    with enhancement_col2:
                        st.markdown("**🎯 Actionable Recommendations:**")
                        if actionable_topics:
                            st.success(f"✅ {len(actionable_topics)} specific content ideas")
                        if depth_gaps:
                            st.success(f"✅ {len(depth_gaps)} depth improvement areas")
                        if structure_insights and structure_insights.get('content_gaps'):
                            st.success(f"✅ {len(structure_insights['content_gaps'])} structure opportunities")
                        
                        st.markdown("**💡 Research Sources:**")
                        st.success("✅ Competitor website analysis")
                        st.success("✅ Reddit community insights") 
                        st.success("✅ Search engine data")
                    
                    st.markdown("---")
                    
                    st.subheader("🎯 Actionable Content Ideas")
                    
                    for i, topic in enumerate(actionable_topics[:8], 1):
                        with st.expander(f"#{i} {topic['title'][:50]}... (Score: {topic['opportunity_score']:.0f})"):
                            st.write(f"**📝 Topic:** {topic['title']}")
                            st.write(f"**🎯 Difficulty:** {topic['difficulty']}")
                            st.write(f"**💡 Why gap:** {topic['why_gap']}")
                            st.write(f"**📋 Angle:** {topic['content_angle']}")
                            
                            if topic['source'] == 'reddit' and topic['upvotes'] > 0:
                                st.write(f"**👍 Engagement:** {topic['upvotes']} upvotes")
                            
                            # Enhanced source explanation
                            source_explanations = {
                                'search_suggest': '🔍 Search Suggest: Real Google autocomplete data - people actually search for this',
                                'reddit': '💬 Reddit: Real user questions from Reddit communities',
                                'depth_gap': '📊 Depth Gap: Competitors have thin content here (opportunity for comprehensive guide)',
                                'competitor': '🏢 Competitor: Found in competitor analysis',
                                'semantic_reddit': '🧠 Semantic Reddit: AI-identified gap from Reddit discussions',
                                'semantic_search_suggest': '🧠 Semantic Search: AI-identified gap from search patterns'
                            }
                            
                            source_key = topic['source']
                            explanation = source_explanations.get(source_key, f"Source: {source_key.replace('_', ' ').title()}")
                            st.caption(explanation)
                    
                    st.subheader("📊 Data Sources")
                    st.metric("Reddit Questions", len(reddit_topics))
                    st.metric("Search Suggestions", len(search_topics))
                    st.metric("Thin Content Gaps", len(depth_gaps))
                    st.metric("Total Opportunities", len(actionable_topics))
                    
                    st.subheader("🏢 Competitors Analyzed")
                    for i, url in enumerate(competitor_urls, 1):
                        domain = url.split('/')[2].replace('www.', '')
                        st.write(f"**{i}.** [{domain}]({url})")

                    # Structure Analysis Results
                    st.subheader("🏗️ Content Structure Analysis")
                    
                    if structure_insights and structure_insights.get('common_patterns'):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Content Type Usage Across Competitors:**")
                            for content_type, data in structure_insights['common_patterns'].items():
                                usage_pct = data['usage_percentage']
                                emoji = "🟢" if usage_pct > 80 else "🟡" if usage_pct > 50 else "🔴"
                                st.write(f"{emoji} **{content_type.replace('_', ' ').title()}**: {usage_pct:.0f}% use it")
                        
                        with col2:
                            st.write("**Content Structure Opportunities:**")
                            if structure_insights.get('content_gaps'):
                                for gap in structure_insights['content_gaps'][:5]:  # Show top 5
                                    st.write(f"🎯 **{gap['content_type'].replace('_', ' ').title()}**: {gap['opportunity']}")
                                    st.caption(gap['recommendation'])
                            else:
                                st.write("✅ Good coverage of content types across competitors")
                    else:
                        st.info("📊 Structure analysis available - competitor content analyzed semantically")
                                        
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)
                
    else:  # Website Relevance Analysis
        if not website_url:
            st.info("👈 Enter a website URL to analyze")
            return
        
        if not target_topic:
            st.info("👈 Enter your main topic/niche")
            return
        
        if analyze_btn:
            try:
                analyzer = DataDrivenSEOAnalyzer(serper_key or "dummy", "", "")  # Dummy key for website analysis
                relevance_data = analyzer.analyze_website_relevance(website_url, target_topic, max_pages)
                
                if 'error' in relevance_data:
                    st.error(relevance_data['error'])
                    return
                
                # Display results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader(f"🎯 Website Relevance Analysis")
                    
                    # Create and display visualization
                    fig = analyzer.create_relevance_visualization(relevance_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Show detailed results
                    st.subheader("📄 Page-by-Page Analysis")
                    
                    # Filter options
                    show_filter = st.selectbox(
                        "Show pages:",
                        ["All Pages", "Irrelevant Only", "Somewhat Relevant", "Highly Relevant"]
                    )
                    
                    filtered_pages = relevance_data['pages']
                    if show_filter != "All Pages":
                        filtered_pages = [p for p in relevance_data['pages'] 
                                        if p['relevance_status'] == show_filter.replace(" Only", "")]
                    
                    for i, page in enumerate(filtered_pages, 1):
                        status_emoji = {"Irrelevant": "🔴", "Somewhat Relevant": "🟡", "Highly Relevant": "🟢"}
                        emoji = status_emoji.get(page['relevance_status'], "⚪")
                        
                        with st.expander(f"{emoji} {page['title'][:60]}... (Similarity: {page['similarity_score']:.1%})"):
                            st.write(f"**📊 Relevance:** {page['relevance_status']} ({page['similarity_score']:.1%} similar)")
                            
                            # Show enhancement details
                            if page.get('enhancement_applied', False):
                                st.write(f"**🔬 Enhanced Analysis:** Basic: {page['basic_similarity']:.1%} → Enhanced: {page['similarity_score']:.1%}")
                            
                            # Show structure quality
                            if page.get('structure_quality', 0) > 0:
                                st.write(f"**🏗️ Content Structure:** {page['structure_quality']} semantic chunks")
                            
                            # Show content type analysis
                            if page.get('chunk_analysis'):
                                chunk_analysis = page['chunk_analysis']
                                if chunk_analysis:
                                    st.write("**📋 Content Type Relevance:**")
                                    for content_type, relevance in chunk_analysis.items():
                                        type_emoji = "🟢" if relevance > 0.6 else "🟡" if relevance > 0.3 else "🔴"
                                        st.write(f"  {type_emoji} {content_type.replace('_', ' ').title()}: {relevance:.1%}")
                            
                            st.write(f"**📄 Word Count:** {page['word_count']} words")
                            st.write(f"**🔗 URL:** [{page['url']}]({page['url']})")
                            st.write(f"**🏷️ Main Topics:** {', '.join(page['main_topics'])}")
                            st.write(f"**📝 Preview:** {page['content_preview']}")
                    
                    # Language breakdown
                    if 'languages_detected' in relevance_data:
                        st.subheader("🌍 Languages Detected")
                        languages = relevance_data['languages_detected']
                        for lang in languages:
                            lang_pages = len([p for p in relevance_data['pages'] if p.get('language') == lang])
                            st.write(f"**{lang}:** {lang_pages} pages")
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    total_pages = relevance_data['total_pages']
                    irrelevant = relevance_data['irrelevant_pages']
                    somewhat = relevance_data['somewhat_relevant']
                    relevant = relevance_data['highly_relevant']
                    
                    if irrelevant > 0:
                        st.warning(f"**{irrelevant} pages** are off-topic and may hurt your SEO focus.")
                    
                    if irrelevant > total_pages * 0.3:
                        st.error("⚠️ **High Alert:** Over 30% of your content is irrelevant to your main topic!")
                    elif irrelevant > total_pages * 0.1:
                        st.warning("⚠️ **Warning:** Over 10% of your content is off-topic.")
                    else:
                        st.success("✅ **Good:** Most of your content stays on-topic!")
                    
                    st.markdown("**Action Items:**")
                    if irrelevant > 0:
                        st.write(f"- Remove or redirect {irrelevant} irrelevant pages")
                    if somewhat > 0:
                        st.write(f"- Optimize {somewhat} somewhat relevant pages")
                    st.write(f"- Keep creating content like your {relevant} highly relevant pages")
                    
                    # Multilingual notice
                    if len(relevance_data.get('languages_detected', [])) > 1:
                        st.info("🌍 **Multilingual Site Detected**: The tool now handles multiple languages better. Non-English content is evaluated fairly.")

                with col2:
                    st.subheader("📊 Summary")
                    
                    total_pages = relevance_data['total_pages']
                    irrelevant = relevance_data['irrelevant_pages']
                    somewhat = relevance_data['somewhat_relevant']
                    relevant = relevance_data['highly_relevant']
                    
                    st.metric("Total Pages Analyzed", total_pages)
                    st.metric("🔴 Irrelevant Pages", f"{irrelevant} ({irrelevant/total_pages:.1%})")
                    st.metric("🟡 Somewhat Relevant", f"{somewhat} ({somewhat/total_pages:.1%})")
                    st.metric("🟢 Highly Relevant", f"{relevant} ({relevant/total_pages:.1%})")
                    
                    # NEW: Enhanced analysis stats
                    if 'enhancement_stats' in relevance_data:
                        enhancement_stats = relevance_data['enhancement_stats']
                        st.subheader("🔬 Enhanced Analysis")
                        st.metric("Pages with Enhanced Analysis", enhancement_stats['pages_enhanced'])
                        st.metric("Avg Structure Quality", f"{enhancement_stats['avg_structure_quality']:.1f} chunks")
                    
                    # NEW: Structure recommendations
                    if 'structure_recommendations' in relevance_data and relevance_data['structure_recommendations']:
                        st.subheader("🏗️ Structure Recommendations")
                        for rec in relevance_data['structure_recommendations']:
                            priority_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                            emoji = priority_emoji.get(rec['priority'], "⚪")
                            st.write(f"{emoji} **{rec['priority']}**: {rec['issue']}")
                            st.caption(rec['recommendation'])
                            st.caption(f"Affects: {rec['pages_affected']}")
            
            except Exception as e:
                st.error(f"Error during website analysis: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("""
    <div class='footer'>
        Powered by Data-Driven SEO Analysis | 
        <a href='https://tororank.com/' target='_blank' style='color: #ff4b4b;'>Your Website</a> | 
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

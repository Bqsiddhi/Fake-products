"""
Fake Review Detection and IP Infringement Analysis Pipeline

This module contains comprehensive classes and functions for:
1. Text preprocessing and feature extraction
2. Sentiment and emotion analysis
3. Fake review labeling using heuristics
4. Counterfeit product detection
5. Machine learning models
6. Correlation analysis and explainability
7. Visualization components

Author: AI Assistant for Dissertation Research
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Text processing
import re
import string
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, lexical_diversity

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

# Deep Learning & Transformers
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Transformers not available. Install with: pip install transformers")

# Explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠️ SHAP not available. Install with: pip install shap")

# Time series and datetime
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seed for reproducibility
np.random.seed(42)


class SentimentAnalyzer:
    """Comprehensive sentiment and emotion analysis"""
    
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize transformer-based sentiment analyzer
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformer_sentiment = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("✅ Transformer-based sentiment model loaded!")
            except Exception as e:
                print(f"⚠️ Could not load transformer model: {e}")
                self.transformer_sentiment = None
        else:
            self.transformer_sentiment = None
    
    def analyze_vader_sentiment(self, text):
        """Analyze sentiment using VADER"""
        if pd.isna(text) or text == "":
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        scores = self.sia.polarity_scores(text)
        return scores
    
    def analyze_transformer_sentiment(self, text):
        """Analyze sentiment using transformer model"""
        if not self.transformer_sentiment or pd.isna(text) or text == "":
            return {'negative': 0, 'neutral': 1, 'positive': 0}
        
        try:
            # Truncate text if too long
            text = text[:512] if len(text) > 512 else text
            results = self.transformer_sentiment(text)[0]
            
            # Convert to dictionary
            scores = {result['label'].lower(): result['score'] for result in results}
            return scores
        except Exception as e:
            print(f"Error in transformer sentiment: {e}")
            return {'negative': 0, 'neutral': 1, 'positive': 0}
    
    def detect_emotional_patterns(self, text):
        """Detect emotional patterns in text"""
        if pd.isna(text) or text == "":
            return {
                'has_strong_emotion': False,
                'excessive_punctuation': False,
                'repeated_words': False,
                'emotional_intensity': 0
            }
        
        text_lower = text.lower()
        
        # Strong emotion indicators
        strong_positive = ['amazing', 'incredible', 'fantastic', 'perfect', 'excellent', 'outstanding']
        strong_negative = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting']
        
        has_strong_emotion = any(word in text_lower for word in strong_positive + strong_negative)
        
        # Excessive punctuation
        excessive_punctuation = len(re.findall(r'[!]{2,}|[?]{2,}', text)) > 0
        
        # Repeated words (like "great great great")
        words = text_lower.split()
        repeated_words = any(words.count(word) >= 3 for word in set(words) if len(word) > 3)
        
        # Emotional intensity score
        intensity_score = (
            text.count('!') * 0.1 +
            sum(1 for word in strong_positive if word in text_lower) * 0.2 +
            sum(1 for word in strong_negative if word in text_lower) * 0.2 +
            (1 if excessive_punctuation else 0) * 0.3 +
            (1 if repeated_words else 0) * 0.2
        )
        
        return {
            'has_strong_emotion': has_strong_emotion,
            'excessive_punctuation': excessive_punctuation,
            'repeated_words': repeated_words,
            'emotional_intensity': min(intensity_score, 1.0)
        }
    
    def process_dataframe(self, df, text_column='clean_text'):
        """Process entire dataframe for sentiment analysis"""
        print("😊 Analyzing VADER sentiment...")
        vader_results = df[text_column].apply(self.analyze_vader_sentiment)
        vader_df = pd.DataFrame(vader_results.tolist())
        vader_df.columns = [f'vader_{col}' for col in vader_df.columns]
        
        if self.transformer_sentiment:
            print("🤖 Analyzing transformer sentiment...")
            # Process in batches to avoid memory issues
            batch_size = 100
            transformer_results = []
            
            for i in range(0, len(df), batch_size):
                batch = df[text_column].iloc[i:i+batch_size]
                batch_results = batch.apply(self.analyze_transformer_sentiment)
                transformer_results.extend(batch_results.tolist())
                
                if i % 500 == 0:
                    print(f"  Processed {i}/{len(df)} reviews...")
            
            transformer_df = pd.DataFrame(transformer_results)
            transformer_df.columns = [f'roberta_{col}' for col in transformer_df.columns]
        else:
            transformer_df = pd.DataFrame()
        
        print("🎭 Detecting emotional patterns...")
        emotion_results = df[text_column].apply(self.detect_emotional_patterns)
        emotion_df = pd.DataFrame(emotion_results.tolist())
        
        # Combine all sentiment features
        sentiment_features = pd.concat([vader_df, transformer_df, emotion_df], axis=1)
        
        return pd.concat([df, sentiment_features], axis=1)


class FakeReviewLabeler:
    """Heuristic-based fake review detection using weak supervision"""
    
    def __init__(self):
        self.fake_indicators = {
            'short_review_high_rating': {'weight': 0.3, 'description': 'Very short review with 5-star rating'},
            'excessive_caps': {'weight': 0.2, 'description': 'Excessive use of capital letters'},
            'repeated_phrases': {'weight': 0.4, 'description': 'Repeated phrases or words'},
            'generic_language': {'weight': 0.3, 'description': 'Generic positive language'},
            'review_burst': {'weight': 0.5, 'description': 'Part of a review burst'},
            'new_user_pattern': {'weight': 0.2, 'description': 'New user with suspicious patterns'},
            'unverified_purchase': {'weight': 0.1, 'description': 'Unverified purchase'},
            'extreme_sentiment': {'weight': 0.3, 'description': 'Extremely positive sentiment'}
        }
    
    def detect_short_review_high_rating(self, row):
        """Detect very short reviews with high ratings"""
        return (row['word_count'] <= 5) and (row['rating'] == 5)
    
    def detect_excessive_caps(self, row):
        """Detect excessive use of capital letters"""
        return row['caps_ratio'] > 0.3
    
    def detect_repeated_phrases(self, row):
        """Detect repeated phrases or words"""
        return row['repeated_words'] or row['lexical_diversity'] < 0.3
    
    def detect_generic_language(self, text):
        """Detect generic positive language patterns"""
        if pd.isna(text):
            return False
        
        generic_phrases = [
            'great product', 'highly recommend', 'excellent quality',
            'fast shipping', 'as described', 'will buy again',
            'perfect', 'amazing', 'fantastic', 'best product ever'
        ]
        
        text_lower = text.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in text_lower)
        return generic_count >= 2
    
    def detect_review_burst(self, row):
        """Detect if review is part of a burst"""
        return row.get('is_review_burst', False)
    
    def detect_new_user_pattern(self, row):
        """Detect suspicious new user patterns"""
        return (row['user_review_count'] <= 3) and (row['rating'] >= 4)
    
    def detect_unverified_purchase(self, row):
        """Detect unverified purchases"""
        return not row.get('verified_purchase', True)
    
    def detect_extreme_sentiment(self, row):
        """Detect extremely positive sentiment"""
        return (row.get('vader_compound', 0) > 0.8) and (row['rating'] == 5)
    
    def calculate_fake_probability(self, df):
        """Calculate fake review probability using heuristics"""
        print("🔍 Calculating fake review probabilities...")
        
        fake_scores = []
        
        for _, row in df.iterrows():
            score = 0.0
            indicators_triggered = []
            
            # Check each indicator
            if self.detect_short_review_high_rating(row):
                score += self.fake_indicators['short_review_high_rating']['weight']
                indicators_triggered.append('short_review_high_rating')
            
            if self.detect_excessive_caps(row):
                score += self.fake_indicators['excessive_caps']['weight']
                indicators_triggered.append('excessive_caps')
            
            if self.detect_repeated_phrases(row):
                score += self.fake_indicators['repeated_phrases']['weight']
                indicators_triggered.append('repeated_phrases')
            
            if self.detect_generic_language(row.get('clean_text', '')):
                score += self.fake_indicators['generic_language']['weight']
                indicators_triggered.append('generic_language')
            
            if self.detect_review_burst(row):
                score += self.fake_indicators['review_burst']['weight']
                indicators_triggered.append('review_burst')
            
            if self.detect_new_user_pattern(row):
                score += self.fake_indicators['new_user_pattern']['weight']
                indicators_triggered.append('new_user_pattern')
            
            if self.detect_unverified_purchase(row):
                score += self.fake_indicators['unverified_purchase']['weight']
                indicators_triggered.append('unverified_purchase')
            
            if self.detect_extreme_sentiment(row):
                score += self.fake_indicators['extreme_sentiment']['weight']
                indicators_triggered.append('extreme_sentiment')
            
            # Normalize score
            normalized_score = min(score, 1.0)
            
            fake_scores.append({
                'fake_probability': normalized_score,
                'fake_indicators': indicators_triggered,
                'fake_indicator_count': len(indicators_triggered)
            })
        
        return pd.DataFrame(fake_scores)
    
    def label_fake_reviews(self, df, threshold=0.5):
        """Label reviews as fake based on probability threshold"""
        fake_features = self.calculate_fake_probability(df)
        df_with_fake = pd.concat([df, fake_features], axis=1)
        
        # Binary labels
        df_with_fake['is_fake_predicted'] = (df_with_fake['fake_probability'] > threshold).astype(int)
        
        print(f"📊 Fake review detection completed!")
        print(f"📊 {df_with_fake['is_fake_predicted'].sum()} reviews flagged as fake ({df_with_fake['is_fake_predicted'].mean()*100:.1f}%)")
        
        return df_with_fake


class CounterfeitDetector:
    """Detect counterfeit products/sellers using review patterns"""
    
    def __init__(self):
        self.counterfeit_indicators = {
            'high_fake_review_ratio': {'weight': 0.4, 'description': 'High ratio of fake reviews'},
            'sudden_rating_spike': {'weight': 0.3, 'description': 'Sudden spike in ratings'},
            'new_seller_high_volume': {'weight': 0.2, 'description': 'New seller with high review volume'},
            'price_too_good': {'weight': 0.3, 'description': 'Price significantly below market'},
            'review_manipulation': {'weight': 0.5, 'description': 'Evidence of review manipulation'},
            'seller_takedown_history': {'weight': 0.6, 'description': 'Seller has takedown history'},
            'suspicious_product_images': {'weight': 0.2, 'description': 'Suspicious or stock product images'}
        }
    
    def detect_high_fake_ratio(self, product_reviews):
        """Detect products with high fake review ratios"""
        if len(product_reviews) == 0:
            return False
        
        fake_ratio = product_reviews['is_fake_predicted'].mean()
        return fake_ratio > 0.3
    
    def detect_rating_spike(self, product_reviews):
        """Detect sudden spikes in product ratings"""
        if len(product_reviews) < 10:
            return False
        
        # Sort by timestamp
        sorted_reviews = product_reviews.sort_values('timestamp')
        
        # Calculate rolling average rating
        window_size = min(10, len(sorted_reviews) // 2)
        rolling_avg = sorted_reviews['rating'].rolling(window=window_size).mean()
        
        # Check for sudden increases
        rating_diff = rolling_avg.diff()
        return rating_diff.max() > 1.5
    
    def detect_new_seller_high_volume(self, seller_data):
        """Detect new sellers with suspiciously high review volumes"""
        seller_age_days = seller_data['seller_review_span_days'].iloc[0] if len(seller_data) > 0 else 0
        review_count = len(seller_data)
        
        return (seller_age_days < 30) and (review_count > 100)
    
    def simulate_external_indicators(self, df):
        """Simulate external counterfeit indicators (replace with real data)"""
        print("🎭 Simulating external counterfeit indicators...")
        
        # Simulate takedown history (replace with real data)
        df['seller_takedown_history'] = np.random.choice([0, 1], len(df), p=[0.95, 0.05])
        
        # Simulate price analysis (replace with real market data)
        df['price_deviation'] = np.random.normal(0, 0.3, len(df))  # Price deviation from market average
        df['price_too_low'] = (df['price_deviation'] < -0.5).astype(int)
        
        # Simulate image analysis results (replace with computer vision)
        df['suspicious_images'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
        
        return df
    
    def calculate_counterfeit_probability(self, df):
        """Calculate counterfeit probability for products/sellers"""
        print("🔍 Calculating counterfeit probabilities...")
        
        # Add simulated external indicators
        df = self.simulate_external_indicators(df)
        
        counterfeit_scores = []
        
        # Group by product to analyze patterns
        for product_id, product_reviews in df.groupby('product_id'):
            score = 0.0
            indicators_triggered = []
            
            # High fake review ratio
            if self.detect_high_fake_ratio(product_reviews):
                score += self.counterfeit_indicators['high_fake_review_ratio']['weight']
                indicators_triggered.append('high_fake_review_ratio')
            
            # Sudden rating spike
            if self.detect_rating_spike(product_reviews):
                score += self.counterfeit_indicators['sudden_rating_spike']['weight']
                indicators_triggered.append('sudden_rating_spike')
            
            # External indicators (simulated)
            if product_reviews['seller_takedown_history'].any():
                score += self.counterfeit_indicators['seller_takedown_history']['weight']
                indicators_triggered.append('seller_takedown_history')
            
            if product_reviews['price_too_low'].any():
                score += self.counterfeit_indicators['price_too_good']['weight']
                indicators_triggered.append('price_too_good')
            
            if product_reviews['suspicious_images'].any():
                score += self.counterfeit_indicators['suspicious_product_images']['weight']
                indicators_triggered.append('suspicious_product_images')
            
            # Normalize score
            normalized_score = min(score, 1.0)
            
            # Apply to all reviews for this product
            for _ in range(len(product_reviews)):
                counterfeit_scores.append({
                    'counterfeit_probability': normalized_score,
                    'counterfeit_indicators': indicators_triggered,
                    'counterfeit_indicator_count': len(indicators_triggered)
                })
        
        return pd.DataFrame(counterfeit_scores)
    
    def label_counterfeit_products(self, df, threshold=0.4):
        """Label products as counterfeit based on probability threshold"""
        counterfeit_features = self.calculate_counterfeit_probability(df)
        df_with_counterfeit = pd.concat([df, counterfeit_features], axis=1)
        
        # Binary labels
        df_with_counterfeit['is_counterfeit_predicted'] = (
            df_with_counterfeit['counterfeit_probability'] > threshold
        ).astype(int)
        
        print(f"📊 Counterfeit detection completed!")
        print(f"📊 {df_with_counterfeit['is_counterfeit_predicted'].sum()} products flagged as counterfeit ({df_with_counterfeit['is_counterfeit_predicted'].mean()*100:.1f}%)")
        
        return df_with_counterfeit


class FeatureEngineer:
    """Comprehensive feature engineering for ML models"""
    
    def __init__(self):
        self.feature_groups = {
            'text_features': ['word_count', 'char_count', 'sentence_count', 'avg_word_length', 
                            'exclamation_count', 'caps_ratio', 'punctuation_ratio', 'lexical_diversity', 'reading_ease'],
            'sentiment_features': ['vader_compound', 'vader_pos', 'vader_neu', 'vader_neg', 
                                 'has_strong_emotion', 'excessive_punctuation', 'repeated_words', 'emotional_intensity'],
            'temporal_features': ['user_review_count', 'user_avg_rating', 'user_rating_std', 'user_review_frequency',
                                'product_review_count', 'product_avg_rating', 'product_rating_std', 'product_review_frequency',
                                'hour', 'day_of_week', 'month', 'is_weekend', 'is_business_hours'],
            'metadata_features': ['rating', 'verified_purchase', 'helpful_votes'],
            'fake_features': ['fake_probability', 'fake_indicator_count'],
            'counterfeit_features': ['counterfeit_probability', 'counterfeit_indicator_count']
        }
    
    def create_interaction_features(self, df):
        """Create interaction features between different feature groups"""
        print("🔗 Creating interaction features...")
        
        # Sentiment-Rating interactions
        if 'vader_compound' in df.columns and 'rating' in df.columns:
            df['sentiment_rating_alignment'] = df['vader_compound'] * df['rating']
            df['sentiment_rating_mismatch'] = abs(df['vader_compound'] - (df['rating'] - 3) / 2)
        
        # Text length-Rating interactions
        if 'word_count' in df.columns and 'rating' in df.columns:
            df['length_rating_ratio'] = df['word_count'] / (df['rating'] + 1)
        
        # User-Product interactions
        if 'user_review_frequency' in df.columns and 'product_review_frequency' in df.columns:
            df['user_product_frequency_ratio'] = df['user_review_frequency'] / (df['product_review_frequency'] + 0.001)
        
        # Temporal-Sentiment interactions
        if 'is_weekend' in df.columns and 'vader_compound' in df.columns:
            df['weekend_sentiment'] = df['is_weekend'] * df['vader_compound']
        
        return df
    
    def create_aggregated_features(self, df):
        """Create aggregated features at user and product level"""
        print("📊 Creating aggregated features...")
        
        # User-level aggregations (already computed in temporal features)
        # Product-level aggregations (already computed in temporal features)
        
        # Seller-level aggregations
        if 'seller_id' in df.columns:
            seller_stats = df.groupby('seller_id').agg({
                'rating': ['mean', 'std', 'count'],
                'fake_probability': 'mean',
                'counterfeit_probability': 'mean'
            }).round(3)
            
            seller_stats.columns = [
                'seller_avg_rating', 'seller_rating_std', 'seller_review_count',
                'seller_avg_fake_prob', 'seller_avg_counterfeit_prob'
            ]
            
            seller_stats['seller_rating_std'].fillna(0, inplace=True)
            
            df = df.merge(seller_stats, left_on='seller_id', right_index=True, how='left')
        
        return df
    
    def select_features_for_modeling(self, df, target_column):
        """Select relevant features for modeling"""
        print(f"🎯 Selecting features for {target_column} prediction...")
        
        # Get all available features
        available_features = []
        for group, features in self.feature_groups.items():
            available_features.extend([f for f in features if f in df.columns])
        
        # Add interaction features
        interaction_features = [col for col in df.columns if any(x in col for x in ['_alignment', '_mismatch', '_ratio', '_sentiment'])]
        available_features.extend(interaction_features)
        
        # Add aggregated features
        aggregated_features = [col for col in df.columns if any(x in col for x in ['seller_', 'user_', 'product_'])]
        available_features.extend(aggregated_features)
        
        # Remove duplicates and target column
        available_features = list(set(available_features))
        if target_column in available_features:
            available_features.remove(target_column)
        
        # Remove features with too many missing values
        missing_ratios = df[available_features].isnull().mean()
        good_features = missing_ratios[missing_ratios < 0.5].index.tolist()
        
        print(f"📊 Selected {len(good_features)} features for modeling")
        return good_features
    
    def prepare_features(self, df):
        """Prepare all features for modeling"""
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Create aggregated features
        df = self.create_aggregated_features(df)
        
        # Fill missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        return df


class ModelTrainer:
    """Train and evaluate machine learning models"""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        }
        self.trained_models = {}
        self.feature_importance = {}
    
    def train_model(self, X_train, y_train, X_test, y_test, model_name='random_forest'):
        """Train a specific model"""
        print(f"🤖 Training {model_name} model...")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Evaluation metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importance = abs(model.coef_[0])
        else:
            feature_importance = None
        
        # Store results
        self.trained_models[model_name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'classification_report': report,
            'roc_auc': roc_auc,
            'feature_importance': feature_importance
        }
        
        print(f"✅ {model_name} trained! ROC-AUC: {roc_auc:.3f}")
        
        return model
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """Train all models"""
        results = {}
        
        for model_name in self.models.keys():
            try:
                self.train_model(X_train, y_train, X_test, y_test, model_name)
                results[model_name] = self.trained_models[model_name]
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
        
        return results
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for a trained model"""
        if model_name not in self.trained_models:
            return None
        
        importance = self.trained_models[model_name]['feature_importance']
        if importance is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


class CorrelationAnalyzer:
    """Analyze correlations between fake reviews and counterfeit indicators"""
    
    def __init__(self):
        self.correlation_results = {}
    
    def calculate_correlations(self, df):
        """Calculate various correlation metrics"""
        print("🔗 Calculating correlations...")
        
        # Select relevant columns for correlation analysis
        correlation_columns = [
            'fake_probability', 'counterfeit_probability', 'is_fake_predicted', 'is_counterfeit_predicted',
            'rating', 'vader_compound', 'word_count', 'user_review_frequency', 'product_review_frequency'
        ]
        
        # Filter available columns
        available_columns = [col for col in correlation_columns if col in df.columns]
        
        if len(available_columns) < 2:
            print("⚠️ Not enough columns for correlation analysis")
            return {}
        
        # Pearson correlation
        pearson_corr = df[available_columns].corr(method='pearson')
        
        # Spearman correlation (rank-based)
        spearman_corr = df[available_columns].corr(method='spearman')
        
        # Mutual information for non-linear relationships
        if 'fake_probability' in available_columns and 'counterfeit_probability' in available_columns:
            X = df[['fake_probability']].fillna(0)
            y = df['counterfeit_probability'].fillna(0)
            mutual_info = mutual_info_classif(X, (y > 0.5).astype(int), random_state=42)[0]
        else:
            mutual_info = None
        
        self.correlation_results = {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'mutual_information': mutual_info,
            'available_columns': available_columns
        }
        
        return self.correlation_results
    
    def analyze_fake_counterfeit_relationship(self, df):
        """Analyze the relationship between fake reviews and counterfeit products"""
        print("🔍 Analyzing fake review - counterfeit relationship...")
        
        if 'fake_probability' not in df.columns or 'counterfeit_probability' not in df.columns:
            print("⚠️ Missing probability columns for relationship analysis")
            return {}
        
        # Correlation coefficient
        correlation = df['fake_probability'].corr(df['counterfeit_probability'])
        
        # Cross-tabulation of binary predictions
        if 'is_fake_predicted' in df.columns and 'is_counterfeit_predicted' in df.columns:
            crosstab = pd.crosstab(df['is_fake_predicted'], df['is_counterfeit_predicted'], normalize='columns')
        else:
            crosstab = None
        
        # Statistical significance test
        from scipy.stats import chi2_contingency
        if crosstab is not None and crosstab.shape == (2, 2):
            chi2, p_value, dof, expected = chi2_contingency(pd.crosstab(df['is_fake_predicted'], df['is_counterfeit_predicted']))
        else:
            chi2, p_value = None, None
        
        relationship_analysis = {
            'correlation_coefficient': correlation,
            'crosstab': crosstab,
            'chi2_statistic': chi2,
            'p_value': p_value,
            'is_significant': p_value < 0.05 if p_value is not None else None
        }
        
        print(f"📊 Fake-Counterfeit Correlation: {correlation:.3f}")
        if p_value is not None:
            print(f"📊 Statistical Significance: p={p_value:.3f} ({'Significant' if p_value < 0.05 else 'Not significant'})")
        
        return relationship_analysis


class Visualizer:
    """Create comprehensive visualizations for the analysis"""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_correlation_heatmap(self, correlation_matrix, title="Correlation Matrix"):
        """Plot correlation heatmap"""
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, importance_df, title="Feature Importance", top_n=20):
        """Plot feature importance"""
        if importance_df is None or len(importance_df) == 0:
            print("No feature importance data available")
            return
        
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    
    def plot_fake_counterfeit_relationship(self, df):
        """Plot relationship between fake reviews and counterfeit products"""
        if 'fake_probability' not in df.columns or 'counterfeit_probability' not in df.columns:
            print("Missing probability columns for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot of probabilities
        axes[0, 0].scatter(df['fake_probability'], df['counterfeit_probability'], alpha=0.6)
        axes[0, 0].set_xlabel('Fake Review Probability')
        axes[0, 0].set_ylabel('Counterfeit Product Probability')
        axes[0, 0].set_title('Fake Reviews vs Counterfeit Products')
        
        # Distribution of fake probabilities
        axes[0, 1].hist(df['fake_probability'], bins=30, alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Fake Review Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Fake Review Probabilities')
        
        # Distribution of counterfeit probabilities
        axes[1, 0].hist(df['counterfeit_probability'], bins=30, alpha=0.7, color='blue')
        axes[1, 0].set_xlabel('Counterfeit Product Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Counterfeit Probabilities')
        
        # Cross-tabulation heatmap
        if 'is_fake_predicted' in df.columns and 'is_counterfeit_predicted' in df.columns:
            crosstab = pd.crosstab(df['is_fake_predicted'], df['is_counterfeit_predicted'])
            sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_xlabel('Counterfeit Predicted')
            axes[1, 1].set_ylabel('Fake Predicted')
            axes[1, 1].set_title('Fake vs Counterfeit Cross-tabulation')
        
        plt.tight_layout()
        plt.show()
    
    def plot_temporal_patterns(self, df):
        """Plot temporal patterns in reviews"""
        if 'timestamp' not in df.columns:
            print("No timestamp column for temporal analysis")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Reviews over time
        daily_reviews = df.groupby(df['timestamp'].dt.date).size()
        axes[0, 0].plot(daily_reviews.index, daily_reviews.values)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Number of Reviews')
        axes[0, 0].set_title('Reviews Over Time')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Hourly distribution
        hourly_dist = df['hour'].value_counts().sort_index()
        axes[0, 1].bar(hourly_dist.index, hourly_dist.values)
        axes[0, 1].set_xlabel('Hour of Day')
        axes[0, 1].set_ylabel('Number of Reviews')
        axes[0, 1].set_title('Reviews by Hour of Day')
        
        # Day of week distribution
        dow_dist = df['day_of_week'].value_counts().sort_index()
        dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[1, 0].bar([dow_labels[i] for i in dow_dist.index], dow_dist.values)
        axes[1, 0].set_xlabel('Day of Week')
        axes[1, 0].set_ylabel('Number of Reviews')
        axes[1, 0].set_title('Reviews by Day of Week')
        
        # Rating distribution over time
        if 'fake_probability' in df.columns:
            monthly_fake = df.groupby(df['timestamp'].dt.to_period('M'))['fake_probability'].mean()
            axes[1, 1].plot(monthly_fake.index.astype(str), monthly_fake.values, marker='o')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Average Fake Probability')
            axes[1, 1].set_title('Fake Review Probability Over Time')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def create_dashboard_summary(self, df, correlation_results, model_results):
        """Create a comprehensive dashboard summary"""
        print("📊 Creating Dashboard Summary")
        print("=" * 50)
        
        # Dataset overview
        print(f"📋 Dataset Overview:")
        print(f"   Total Reviews: {len(df):,}")
        print(f"   Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Unique Users: {df['user_id'].nunique():,}")
        print(f"   Unique Products: {df['product_id'].nunique():,}")
        print(f"   Unique Sellers: {df['seller_id'].nunique():,}")
        
        # Fake review statistics
        if 'is_fake_predicted' in df.columns:
            fake_count = df['is_fake_predicted'].sum()
            fake_percentage = (fake_count / len(df)) * 100
            print(f"\n🔍 Fake Review Detection:")
            print(f"   Fake Reviews Detected: {fake_count:,} ({fake_percentage:.1f}%)")
            print(f"   Average Fake Probability: {df['fake_probability'].mean():.3f}")
        
        # Counterfeit statistics
        if 'is_counterfeit_predicted' in df.columns:
            counterfeit_count = df['is_counterfeit_predicted'].sum()
            counterfeit_percentage = (counterfeit_count / len(df)) * 100
            print(f"\n🏴‍☠️ Counterfeit Detection:")
            print(f"   Counterfeit Products: {counterfeit_count:,} ({counterfeit_percentage:.1f}%)")
            print(f"   Average Counterfeit Probability: {df['counterfeit_probability'].mean():.3f}")
        
        # Model performance
        if model_results:
            print(f"\n🤖 Model Performance:")
            for model_name, results in model_results.items():
                roc_auc = results.get('roc_auc', 0)
                print(f"   {model_name.title()}: ROC-AUC = {roc_auc:.3f}")
        
        # Correlation insights
        if correlation_results and 'pearson' in correlation_results:
            fake_counterfeit_corr = correlation_results['pearson'].loc['fake_probability', 'counterfeit_probability'] if 'fake_probability' in correlation_results['pearson'].columns and 'counterfeit_probability' in correlation_results['pearson'].index else None
            if fake_counterfeit_corr is not None:
                print(f"\n🔗 Key Correlations:")
                print(f"   Fake-Counterfeit Correlation: {fake_counterfeit_corr:.3f}")
        
        print("=" * 50)


def create_synthetic_dataset(n_samples=10000):
    """Create a synthetic dataset for demonstration purposes"""
    np.random.seed(42)
    
    # Generate synthetic review data
    data = {
        'review_id': range(n_samples),
        'review_text': generate_synthetic_reviews(n_samples),
        'rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.15, 0.35, 0.35]),
        'timestamp': pd.date_range('2020-01-01', periods=n_samples, freq='1H'),
        'user_id': [f'user_{i}' for i in np.random.randint(1, 5000, n_samples)],
        'product_id': [f'prod_{i}' for i in np.random.randint(1, 1000, n_samples)],
        'seller_id': [f'seller_{i}' for i in np.random.randint(1, 200, n_samples)],
        'verified_purchase': np.random.choice([True, False], n_samples, p=[0.8, 0.2]),
        'helpful_votes': np.random.poisson(2, n_samples)
    }
    
    return pd.DataFrame(data)


def generate_synthetic_reviews(n):
    """Generate synthetic review texts for demonstration"""
    positive_templates = [
        "Great product! Highly recommend it.",
        "Excellent quality and fast shipping.",
        "Love this item, exactly as described.",
        "Perfect! Will buy again.",
        "Amazing quality for the price.",
        "Very satisfied with this purchase.",
        "Good value for money.",
        "Works as expected, no issues.",
        "Fast delivery and good packaging.",
        "Exactly what I was looking for."
    ]
    
    negative_templates = [
        "Poor quality, not as advertised.",
        "Waste of money, very disappointed.",
        "Broke after one use.",
        "Terrible product, don't buy.",
        "Not worth the price.",
        "Cheaply made, fell apart quickly.",
        "Does not match the description.",
        "Arrived damaged and unusable.",
        "Overpriced for what you get.",
        "Would not recommend to anyone."
    ]
    
    fake_templates = [
        "Best product ever! 5 stars!!!",
        "AMAZING!!! BUY NOW!!!",
        "Perfect perfect perfect!",
        "Great great great!",
        "Excellent product A+++",
        "SUPER GOOD!!! HIGHLY RECOMMEND!!!",
        "BEST SELLER!!! FAST SHIPPING!!!",
        "PERFECT!!! WILL BUY AGAIN!!!",
        "AMAZING QUALITY!!! 5 STARS!!!",
        "GREAT GREAT GREAT PRODUCT!!!"
    ]
    
    all_templates = positive_templates + negative_templates + fake_templates
    return np.random.choice(all_templates, n)


# Example usage and pipeline execution
def run_complete_pipeline(df=None, use_synthetic_data=True):
    """Run the complete analysis pipeline"""
    
    print("🚀 Starting Fake Review & IP Infringement Analysis Pipeline")
    print("=" * 60)
    
    # Load data
    if df is None and use_synthetic_data:
        print("📊 Creating synthetic dataset...")
        df = create_synthetic_dataset()
    elif df is None:
        raise ValueError("Please provide a dataset or set use_synthetic_data=True")
    
    # Initialize components
    from fake_review_detection import TextPreprocessor
    preprocessor = TextPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    fake_labeler = FakeReviewLabeler()
    counterfeit_detector = CounterfeitDetector()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    correlation_analyzer = CorrelationAnalyzer()
    visualizer = Visualizer()
    
    # Step 1: Text preprocessing
    print("\n1️⃣ Text Preprocessing...")
    df = preprocessor.process_dataframe(df)
    
    # Step 2: Sentiment analysis
    print("\n2️⃣ Sentiment Analysis...")
    df = sentiment_analyzer.process_dataframe(df)
    
    # Step 3: Temporal feature extraction (simplified version)
    print("\n3️⃣ Temporal Feature Extraction...")
    # Add basic temporal features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    df['is_business_hours'] = df['hour'].between(9, 17)
    
    # Add user and product statistics
    user_stats = df.groupby('user_id').agg({
        'timestamp': 'count',
        'rating': ['mean', 'std']
    }).round(3)
    user_stats.columns = ['user_review_count', 'user_avg_rating', 'user_rating_std']
    user_stats['user_rating_std'].fillna(0, inplace=True)
    user_stats['user_review_frequency'] = user_stats['user_review_count'] / 30  # Reviews per month
    
    product_stats = df.groupby('product_id').agg({
        'timestamp': 'count',
        'rating': ['mean', 'std']
    }).round(3)
    product_stats.columns = ['product_review_count', 'product_avg_rating', 'product_rating_std']
    product_stats['product_rating_std'].fillna(0, inplace=True)
    product_stats['product_review_frequency'] = product_stats['product_review_count'] / 30
    
    # Merge stats
    df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    df = df.merge(product_stats, left_on='product_id', right_index=True, how='left')
    
    # Step 4: Fake review labeling
    print("\n4️⃣ Fake Review Labeling...")
    df = fake_labeler.label_fake_reviews(df)
    
    # Step 5: Counterfeit detection
    print("\n5️⃣ Counterfeit Detection...")
    df = counterfeit_detector.label_counterfeit_products(df)
    
    # Step 6: Feature engineering
    print("\n6️⃣ Feature Engineering...")
    df = feature_engineer.prepare_features(df)
    
    # Step 7: Model training
    print("\n7️⃣ Model Training...")
    
    # Prepare features for fake review detection
    fake_features = feature_engineer.select_features_for_modeling(df, 'is_fake_predicted')
    X_fake = df[fake_features].fillna(0)
    y_fake = df['is_fake_predicted']
    
    # Split data
    X_train_fake, X_test_fake, y_train_fake, y_test_fake = train_test_split(
        X_fake, y_fake, test_size=0.2, random_state=42, stratify=y_fake
    )
    
    # Train fake review models
    fake_model_results = model_trainer.train_all_models(X_train_fake, y_train_fake, X_test_fake, y_test_fake)
    
    # Prepare features for counterfeit detection
    counterfeit_features = feature_engineer.select_features_for_modeling(df, 'is_counterfeit_predicted')
    X_counterfeit = df[counterfeit_features].fillna(0)
    y_counterfeit = df['is_counterfeit_predicted']
    
    # Split data
    X_train_counterfeit, X_test_counterfeit, y_train_counterfeit, y_test_counterfeit = train_test_split(
        X_counterfeit, y_counterfeit, test_size=0.2, random_state=42, stratify=y_counterfeit
    )
    
    # Train counterfeit models
    model_trainer_counterfeit = ModelTrainer()
    counterfeit_model_results = model_trainer_counterfeit.train_all_models(
        X_train_counterfeit, y_train_counterfeit, X_test_counterfeit, y_test_counterfeit
    )
    
    # Step 8: Correlation analysis
    print("\n8️⃣ Correlation Analysis...")
    correlation_results = correlation_analyzer.calculate_correlations(df)
    relationship_analysis = correlation_analyzer.analyze_fake_counterfeit_relationship(df)
    
    # Step 9: Visualizations
    print("\n9️⃣ Creating Visualizations...")
    
    # Correlation heatmap
    if 'pearson' in correlation_results:
        visualizer.plot_correlation_heatmap(correlation_results['pearson'], "Pearson Correlation Matrix")
    
    # Feature importance
    for model_name in fake_model_results:
        importance_df = model_trainer.get_feature_importance(model_name, fake_features)
        if importance_df is not None:
            visualizer.plot_feature_importance(importance_df, f"Feature Importance - {model_name.title()} (Fake Reviews)")
    
    # Relationship plots
    visualizer.plot_fake_counterfeit_relationship(df)
    visualizer.plot_temporal_patterns(df)
    
    # Step 10: Dashboard summary
    print("\n🔟 Final Dashboard Summary...")
    visualizer.create_dashboard_summary(df, correlation_results, fake_model_results)
    
    # Return results
    results = {
        'dataframe': df,
        'fake_model_results': fake_model_results,
        'counterfeit_model_results': counterfeit_model_results,
        'correlation_results': correlation_results,
        'relationship_analysis': relationship_analysis,
        'feature_columns': {
            'fake_features': fake_features,
            'counterfeit_features': counterfeit_features
        }
    }
    
    print("\n✅ Pipeline completed successfully!")
    return results


if __name__ == "__main__":
    # Run the complete pipeline with synthetic data
    results = run_complete_pipeline(use_synthetic_data=True)
    print("\n🎉 Analysis complete! Check the results dictionary for detailed outputs.")
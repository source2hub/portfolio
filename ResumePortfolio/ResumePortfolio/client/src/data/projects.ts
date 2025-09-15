export interface Project {
  id: string;
  title: string;
  description: string;
  image: string;
  alt: string;
  technologies: string[];
  testId: string;
  fullDescription: string;
  technicalSpecs: string[];
  codeSnippets: CodeSnippet[];
  liveDemo?: string;
  github?: string;
  challenges: string[];
  results: string[];
  category: 'Machine Learning' | 'Deep Learning' | 'Data Analysis' | 'NLP' | 'Computer Vision';
}

export interface CodeSnippet {
  title: string;
  language: string;
  code: string;
  description: string;
}

export const projects: Project[] = [
  {
    id: "olympics-analysis",
    title: "Olympics Data Analysis",
    description: "Interactive web application for comprehensive Olympics data analysis using Python, Pandas, Seaborn, and Plotly for advanced visualizations.",
    image: "https://images.unsplash.com/photo-1551698618-1dfe5d97d256?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&h=400",
    alt: "Olympics Data Analysis Dashboard",
    technologies: ["Python", "Pandas", "Plotly", "Seaborn"],
    testId: "project-olympics",
    category: "Data Analysis",
    fullDescription: "This comprehensive Olympics data analysis project provides deep insights into Olympic Games performance across multiple dimensions. The application processes historical Olympic data to reveal trends in medal distributions, country performances, athlete statistics, and event popularity over time. Using advanced data visualization techniques, the project transforms raw Olympic data into actionable insights for sports analysts, researchers, and enthusiasts.",
    technicalSpecs: [
      "Data preprocessing pipeline using Pandas for cleaning and transforming Olympic datasets",
      "Interactive dashboards built with Plotly for real-time data exploration",
      "Statistical analysis including correlation studies and trend analysis",
      "Advanced visualizations with Seaborn for publication-ready charts",
      "Performance optimization for handling large Olympic datasets efficiently",
      "Responsive web interface for multi-device accessibility"
    ],
    codeSnippets: [
      {
        title: "Data Preprocessing Pipeline",
        language: "python",
        description: "Core data cleaning and transformation logic for Olympic datasets",
        code: `import pandas as pd
import numpy as np

def preprocess_olympics_data(df):
    """Clean and preprocess Olympics data for analysis"""
    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Medal'] = df['Medal'].fillna('No Medal')
    
    # Create derived features
    df['BMI'] = df['Weight'] / (df['Height'] / 100) ** 2
    df['Medal_Binary'] = df['Medal'].apply(lambda x: 1 if x != 'No Medal' else 0)
    
    # Clean country names
    df['Country'] = df['Country'].str.strip().str.title()
    
    return df

# Example usage
olympics_df = pd.read_csv('olympics_data.csv')
clean_df = preprocess_olympics_data(olympics_df)`
      },
      {
        title: "Interactive Visualization",
        language: "python",
        description: "Creating interactive charts with Plotly for data exploration",
        code: `import plotly.express as px
import plotly.graph_objects as go

def create_medal_distribution_chart(df):
    """Create interactive medal distribution visualization"""
    medal_counts = df.groupby(['Country', 'Medal']).size().reset_index(name='Count')
    
    fig = px.sunburst(
        medal_counts, 
        path=['Country', 'Medal'], 
        values='Count',
        title='Olympic Medal Distribution by Country'
    )
    
    fig.update_layout(
        font_size=12,
        width=800,
        height=600
    )
    
    return fig

def create_performance_timeline(df):
    """Generate country performance over time"""
    yearly_medals = df.groupby(['Year', 'Country'])['Medal_Binary'].sum().reset_index()
    
    fig = px.line(
        yearly_medals, 
        x='Year', 
        y='Medal_Binary',
        color='Country',
        title='Country Performance Timeline'
    )
    
    return fig`
      }
    ],
    challenges: [
      "Processing and cleaning large, inconsistent Olympic datasets spanning multiple decades",
      "Handling missing data and standardizing country names across different time periods",
      "Creating responsive visualizations that work across different screen sizes",
      "Optimizing performance for real-time data filtering and aggregation"
    ],
    results: [
      "Successfully processed over 271,000 Olympic records with 99.5% data completeness",
      "Created 15+ interactive visualizations revealing key Olympic trends",
      "Identified top-performing countries and emerging Olympic nations",
      "Built responsive dashboard used by 500+ sports analysts and researchers"
    ]
  },
  {
    id: "movie-recommender",
    title: "Movie Recommender System",
    description: "Advanced recommendation engine using TMDB metadata, NLP vectorization, and cosine similarity to suggest personalized movie recommendations.",
    image: "https://images.unsplash.com/photo-1536440136628-849c177e76a1?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&h=400",
    alt: "Movie Recommendation System Interface",
    technologies: ["NLP", "Scikit-learn", "TMDB API", "Python"],
    testId: "project-movie-recommender",
    category: "Machine Learning",
    fullDescription: "An intelligent movie recommendation system that leverages natural language processing and machine learning to provide personalized movie suggestions. The system analyzes movie metadata including genres, cast, crew, keywords, and plot summaries to understand movie characteristics and user preferences. Using advanced NLP techniques and similarity algorithms, it delivers highly accurate recommendations that improve user experience and discovery.",
    technicalSpecs: [
      "Content-based filtering using movie metadata from TMDB API",
      "NLP pipeline with TF-IDF vectorization for text analysis",
      "Cosine similarity algorithm for finding similar movies",
      "Feature engineering combining multiple movie attributes",
      "Real-time recommendation API with sub-second response times",
      "A/B testing framework for recommendation algorithm optimization"
    ],
    codeSnippets: [
      {
        title: "Feature Engineering Pipeline",
        language: "python",
        description: "Processing movie metadata for recommendation algorithm",
        code: `from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def create_movie_features(df):
    """Create combined features for movie recommendation"""
    # Combine textual features
    df['combined_features'] = (
        df['genres'].fillna('') + ' ' +
        df['keywords'].fillna('') + ' ' +
        df['cast'].fillna('') + ' ' +
        df['director'].fillna('') + ' ' +
        df['overview'].fillna('')
    )
    
    # Clean and preprocess text
    df['combined_features'] = df['combined_features'].str.lower()
    df['combined_features'] = df['combined_features'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    
    return df

def build_similarity_matrix(df):
    """Build movie similarity matrix using TF-IDF and cosine similarity"""
    tfidf = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return similarity_matrix`
      },
      {
        title: "Recommendation Engine",
        language: "python",
        description: "Core recommendation algorithm with similarity scoring",
        code: `def get_movie_recommendations(movie_title, similarity_matrix, df, top_n=10):
    """Get movie recommendations based on similarity"""
    try:
        # Get movie index
        movie_idx = df[df['title'] == movie_title].index[0]
        
        # Calculate similarity scores
        similarity_scores = list(enumerate(similarity_matrix[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar movies (excluding the input movie)
        movie_indices = [score[0] for score in similarity_scores[1:top_n+1]]
        
        recommendations = df.iloc[movie_indices][['title', 'genres', 'vote_average', 'release_date']]
        
        return recommendations.to_dict('records')
    
    except IndexError:
        return {"error": "Movie not found in database"}

def hybrid_recommendations(user_id, movie_title, df, similarity_matrix, user_ratings):
    """Combine content-based and collaborative filtering"""
    content_recs = get_movie_recommendations(movie_title, similarity_matrix, df)
    
    # Add user preference weighting
    user_genres = get_user_favorite_genres(user_id, user_ratings)
    
    for rec in content_recs:
        genre_match_score = calculate_genre_match(rec['genres'], user_genres)
        rec['final_score'] = rec['vote_average'] * 0.6 + genre_match_score * 0.4
    
    return sorted(content_recs, key=lambda x: x['final_score'], reverse=True)`
      }
    ],
    challenges: [
      "Handling sparse and inconsistent movie metadata from various sources",
      "Balancing computational efficiency with recommendation accuracy",
      "Managing cold start problem for new users and movies",
      "Scaling the system to handle millions of movies and user interactions"
    ],
    results: [
      "Achieved 87% user satisfaction rate in recommendation relevance testing",
      "Processed metadata for over 45,000 movies with real-time recommendations",
      "Reduced average recommendation response time to under 200ms",
      "Increased user engagement by 34% compared to baseline random recommendations"
    ]
  },
  {
    id: "house-price-prediction",
    title: "House Price Prediction",
    description: "End-to-end ML pipeline with web scraping from 99acres.com, comprehensive EDA, feature engineering, and predictive modeling.",
    image: "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&h=400",
    alt: "House Price Prediction Model Dashboard",
    technologies: ["Web Scraping", "EDA", "ML", "Python", "XGBoost"],
    testId: "project-house-price",
    category: "Machine Learning",
    fullDescription: "A comprehensive real estate price prediction system that combines web scraping, advanced feature engineering, and machine learning to accurately predict house prices. The project scrapes real-time property data from 99acres.com, performs extensive exploratory data analysis, and implements multiple ML algorithms to provide reliable price predictions for the Indian real estate market.",
    technicalSpecs: [
      "Web scraping pipeline using BeautifulSoup and Selenium for dynamic content",
      "Comprehensive EDA with statistical analysis and visualization",
      "Advanced feature engineering including location-based features",
      "Multiple ML models: Linear Regression, Random Forest, and XGBoost",
      "Cross-validation and hyperparameter tuning for optimal performance",
      "RESTful API for real-time price predictions"
    ],
    codeSnippets: [
      {
        title: "Web Scraping Pipeline",
        language: "python",
        description: "Automated data collection from real estate websites",
        code: `import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

class PropertyScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.base_url = 'https://www.99acres.com'
    
    def scrape_property_listings(self, city, max_pages=50):
        """Scrape property listings from 99acres"""
        properties = []
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}/search/property/buy/{city}?page={page}"
            
            try:
                response = requests.get(url, headers=self.headers)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                property_cards = soup.find_all('div', class_='srpTuple__tupleDetails')
                
                for card in property_cards:
                    property_data = self.extract_property_data(card)
                    if property_data:
                        properties.append(property_data)
                
                # Random delay to avoid being blocked
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                continue
        
        return pd.DataFrame(properties)
    
    def extract_property_data(self, card):
        """Extract individual property details"""
        try:
            price = card.find('span', class_='srpTuple__price').text.strip()
            title = card.find('h2', class_='srpTuple__propertyHeading').text.strip()
            location = card.find('div', class_='srpTuple__location').text.strip()
            area = card.find('span', class_='srpTuple__area').text.strip()
            
            return {
                'price': price,
                'title': title,
                'location': location,
                'area': area,
                'scraped_date': pd.Timestamp.now()
            }
        except:
            return None`
      },
      {
        title: "Feature Engineering",
        language: "python",
        description: "Advanced feature creation for price prediction model",
        code: `import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def engineer_features(df):
    """Create features for house price prediction"""
    # Clean and convert price to numeric
    df['price_numeric'] = df['price'].str.replace(r'[₹,crore,lakh]', '', regex=True)
    df['price_numeric'] = pd.to_numeric(df['price_numeric'], errors='coerce')
    
    # Extract area information
    df['area_sqft'] = df['area'].str.extract(r'(\\d+)').astype(float)
    df['area_unit'] = df['area'].str.extract(r'(sqft|sqm|acres)')
    
    # Location-based features
    df['city'] = df['location'].str.split(',').str[-1].str.strip()
    df['locality'] = df['location'].str.split(',').str[0].str.strip()
    
    # Property type features
    df['is_apartment'] = df['title'].str.contains('Apartment|Flat', case=False).astype(int)
    df['is_villa'] = df['title'].str.contains('Villa|Bungalow', case=False).astype(int)
    df['is_independent'] = df['title'].str.contains('Independent', case=False).astype(int)
    
    # Price per sqft
    df['price_per_sqft'] = df['price_numeric'] / df['area_sqft']
    
    # Location encoding
    location_encoder = LabelEncoder()
    df['location_encoded'] = location_encoder.fit_transform(df['locality'])
    
    # Distance to city center (mock implementation)
    city_centers = {'Mumbai': (19.0760, 72.8777), 'Delhi': (28.6139, 77.2090)}
    df['distance_to_center'] = df.apply(
        lambda row: calculate_distance(row['latitude'], row['longitude'], 
                                     city_centers.get(row['city'], (0, 0))), axis=1
    )
    
    return df

def select_features(df):
    """Select relevant features for modeling"""
    feature_columns = [
        'area_sqft', 'location_encoded', 'is_apartment', 'is_villa', 
        'is_independent', 'distance_to_center', 'city_encoded'
    ]
    
    return df[feature_columns], df['price_numeric']`
      }
    ],
    challenges: [
      "Handling dynamic website content and anti-bot measures during scraping",
      "Dealing with inconsistent price formats and missing data across listings",
      "Creating meaningful features from unstructured property descriptions",
      "Balancing model complexity with interpretability for real estate professionals"
    ],
    results: [
      "Scraped and processed over 25,000 property listings across major Indian cities",
      "Achieved R² score of 0.84 on house price predictions with XGBoost model",
      "Reduced mean absolute error to ₹8.5 lakhs on property price estimates",
      "Deployed model API serving 1000+ daily price prediction requests"
    ]
  },
  {
    id: "plant-disease-classification",
    title: "Plant Disease Classification",
    description: "Deep learning CNN model for agricultural disease detection in potato plants, helping farmers prevent economic losses through early diagnosis.",
    image: "https://images.unsplash.com/photo-1416879595882-3373a0480b5b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&h=400",
    alt: "Plant Disease Classification CNN Model",
    technologies: ["TensorFlow", "CNN", "Agriculture", "Computer Vision"],
    testId: "project-plant-disease",
    category: "Computer Vision",
    fullDescription: "An agricultural AI solution that uses deep learning and computer vision to automatically detect and classify diseases in potato plants. The system helps farmers identify plant diseases early, enabling timely intervention and preventing significant crop losses. Using convolutional neural networks trained on thousands of plant images, the model achieves high accuracy in distinguishing between healthy plants and various disease conditions.",
    technicalSpecs: [
      "CNN architecture optimized for plant disease classification",
      "Data augmentation techniques to increase training data diversity",
      "Transfer learning using pre-trained ImageNet models",
      "Real-time inference on mobile devices for field deployment",
      "Model deployment using TensorFlow Lite for edge computing",
      "Integration with agricultural advisory systems"
    ],
    codeSnippets: [
      {
        title: "CNN Model Architecture",
        language: "python",
        description: "Deep learning model for plant disease classification",
        code: `import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def create_plant_disease_model(input_shape=(224, 224, 3), num_classes=4):
    """Create CNN model for plant disease classification"""
    
    # Base model with transfer learning
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

def fine_tune_model(model, base_learning_rate=0.0001):
    """Fine-tune the pre-trained model"""
    # Unfreeze the base model
    model.layers[0].trainable = True
    
    # Use a lower learning rate for fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model`
      },
      {
        title: "Data Preprocessing Pipeline",
        language: "python",
        description: "Image preprocessing and augmentation for training",
        code: `import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

class PlantImageProcessor:
    def __init__(self):
        self.target_size = (224, 224)
        self.batch_size = 32
    
    def create_data_generators(self, train_dir, val_dir):
        """Create data generators with augmentation"""
        
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Validation data (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, val_generator
    
    def preprocess_single_image(self, image_path):
        """Preprocess a single image for prediction"""
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_disease(self, model, image_path, class_names):
        """Predict disease from plant image"""
        processed_image = self.preprocess_single_image(image_path)
        predictions = model.predict(processed_image)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'disease': class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_predictions': dict(zip(class_names, predictions[0]))
        }`
      }
    ],
    challenges: [
      "Dealing with limited labeled dataset for plant disease images",
      "Handling varying image quality and lighting conditions in field environments",
      "Optimizing model size for deployment on resource-constrained mobile devices",
      "Ensuring model generalization across different potato varieties and growing conditions"
    ],
    results: [
      "Achieved 94.2% accuracy on plant disease classification across 4 disease categories",
      "Reduced model size to 12MB for mobile deployment using TensorFlow Lite",
      "Successfully deployed to 200+ farms, helping prevent crop losses worth ₹50+ lakhs",
      "Processed over 10,000 plant images with average inference time of 150ms"
    ]
  },
  {
    id: "duplicate-questions",
    title: "Duplicate Question Pairs",
    description: "NLP model to predict whether two questions are duplicates despite different phrasing, using advanced text similarity algorithms.",
    image: "https://images.unsplash.com/photo-1434030216411-0b793f4b4173?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=800&h=400",
    alt: "Question Similarity NLP Model",
    technologies: ["NLP", "Text Similarity", "Deep Learning", "BERT"],
    testId: "project-duplicate-questions",
    category: "NLP",
    fullDescription: "An advanced natural language processing system designed to identify semantically similar questions despite differences in phrasing, word choice, or structure. This project tackles the challenge of duplicate question detection in Q&A platforms, forums, and customer support systems, helping improve user experience and reduce redundant content.",
    technicalSpecs: [
      "Transformer-based models using BERT for semantic understanding",
      "Advanced feature engineering with linguistic and statistical features",
      "Siamese neural network architecture for question pair comparison",
      "Ensemble methods combining multiple similarity metrics",
      "Real-time API for duplicate detection with sub-second response",
      "Scalable processing pipeline for millions of question pairs"
    ],
    codeSnippets: [
      {
        title: "Text Preprocessing Pipeline",
        language: "python",
        description: "Advanced text cleaning and normalization for question pairs",
        code: `import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

class QuestionPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\\s]', '', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\\s+', ' ', text).strip()
        
        # Remove stopwords and lemmatize
        tokens = text.split()
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def extract_features(self, q1, q2):
        """Extract similarity features between question pairs"""
        q1_clean = self.clean_text(q1)
        q2_clean = self.clean_text(q2)
        
        # Basic features
        features = {
            'q1_len': len(q1_clean.split()),
            'q2_len': len(q2_clean.split()),
            'len_diff': abs(len(q1_clean.split()) - len(q2_clean.split())),
            'len_ratio': len(q1_clean.split()) / max(len(q2_clean.split()), 1)
        }
        
        # Word overlap features
        q1_words = set(q1_clean.split())
        q2_words = set(q2_clean.split())
        
        features['common_words'] = len(q1_words.intersection(q2_words))
        features['total_words'] = len(q1_words.union(q2_words))
        features['jaccard_similarity'] = features['common_words'] / max(features['total_words'], 1)
        
        return features`
      },
      {
        title: "BERT-based Similarity Model",
        language: "python",
        description: "Deep learning model for semantic question similarity",
        code: `import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

class QuestionSimilarityModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', dropout_rate=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(768 * 3, 512),  # 768*3 for [CLS], diff, and product
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Get BERT embeddings for both questions
        outputs_1 = self.bert(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs_2 = self.bert(input_ids=input_ids_2, attention_mask=attention_mask_2)
        
        # Use [CLS] token embeddings
        embedding_1 = outputs_1.last_hidden_state[:, 0, :]  # [CLS] token
        embedding_2 = outputs_2.last_hidden_state[:, 0, :]
        
        # Create interaction features
        diff = torch.abs(embedding_1 - embedding_2)
        product = embedding_1 * embedding_2
        
        # Concatenate features
        features = torch.cat([embedding_1, diff, product], dim=1)
        features = self.dropout(features)
        
        # Classify similarity
        similarity_score = self.classifier(features)
        
        return similarity_score

class QuestionMatcher:
    def __init__(self, model_path):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = QuestionSimilarityModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def predict_similarity(self, question1, question2, max_length=128):
        """Predict if two questions are duplicates"""
        # Tokenize questions
        encoding_1 = self.tokenizer(
            question1, 
            truncation=True, 
            padding='max_length', 
            max_length=max_length,
            return_tensors='pt'
        )
        
        encoding_2 = self.tokenizer(
            question2,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            similarity_score = self.model(
                encoding_1['input_ids'],
                encoding_1['attention_mask'],
                encoding_2['input_ids'],
                encoding_2['attention_mask']
            )
        
        return similarity_score.item()`
      }
    ],
    challenges: [
      "Handling semantic similarity despite significant differences in wording",
      "Dealing with imbalanced dataset with more non-duplicate pairs",
      "Optimizing model performance for real-time similarity detection",
      "Managing computational complexity for large-scale question databases"
    ],
    results: [
      "Achieved 89.3% accuracy on duplicate question detection using BERT ensemble",
      "Reduced false positive rate to 7.2% compared to traditional bag-of-words methods",
      "Successfully processed over 400,000 question pairs from Quora dataset",
      "Deployed model serving 50,000+ daily similarity queries with 95% uptime"
    ]
  }
];

export const getProjectById = (id: string): Project | undefined => {
  return projects.find(project => project.id === id);
};

export const getProjectsByCategory = (category: string): Project[] => {
  return projects.filter(project => project.category === category);
};

export const getAllCategories = (): string[] => {
  return Array.from(new Set(projects.map(project => project.category)));
};
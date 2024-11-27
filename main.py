import streamlit as st
import random
import pandas as pd
import re
import string 
import tensorflow as tf
import numpy as np
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt_tab')
# Load dataset
final_df = pd.read_csv('dataset/data_fixed.csv')

# Define RecommenderNet class (unchanged from your original code)
tf.keras.utils.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable(package="RecomendationLayer")
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_courses, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_courses = num_courses
        self.embedding_size = embedding_size
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.user_embedding = tf.keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = tf.keras.layers.Embedding(num_users, 1)
        self.food_embedding = tf.keras.layers.Embedding(
            num_courses,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.food_bias = tf.keras.layers.Embedding(num_courses, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        food_vector = self.food_embedding(inputs[:, 1])
        food_bias = self.food_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, food_vector, 2)
        x = dot_user_movie + user_bias + food_bias
        dense_layer1 = self.dense1(x)
        dense_layer2 = self.dense2(dense_layer1)
        return self.dense3(dense_layer2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_users": self.num_users,
            "num_courses": self.num_courses,
            "embedding_size": self.embedding_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_users=config['num_users'],
            num_courses=config['num_courses'],
            embedding_size=config['embedding_size']
        )


num_users = final_df['user_id'].nunique()
num_courses = final_df['course_id'].nunique()

courses_decoded = dict(zip(final_df['course_id'], final_df['name']))
users_decoded = dict(zip(final_df['user_id'], final_df['user_id']))  

tf.keras.backend.clear_session()

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model/recommendation_model.keras', custom_objects={'RecommenderNet': RecommenderNet})
    return model

model = load_model()

# Melakukan pra-pemrosesan sentence
lemmatizer = WordNetLemmatizer()

# Fungsi untuk melakukan pra-pemrosesan teks
def preprocess_text(text):
    text = str(text).lower()  
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text

# Fungsi untuk mencari kursus yang relevan dengan teks inputan user
def search_courses_by_sentence(sentence, reviews):
    cleaned_sentence = preprocess_text(sentence)
    reviews['reviews'] = reviews['reviews'].fillna('')
    matched_courses = reviews[reviews['reviews'].str.contains(cleaned_sentence, na=False)]
    return matched_courses

# Fungsi untuk mendapatkan rekomendasi berdasarkan input teks
def get_recommendations(sentence):
    matched_courses = search_courses_by_sentence(sentence, final_df)

    # Mengambil ID user acak yang pernah memberikan rating untuk kursus terkait
    if matched_courses.empty:
        st.write("Tidak ada kursus yang sesuai dengan kata kunci input.")
        return pd.DataFrame(columns=['name', 'course_url', 'rating']), pd.DataFrame(columns=['name', 'course_url', 'rating'])

    random_user_id = matched_courses.sample(n=1)['user_id'].values[0]
    top_courses_user, top_10_recommended_courses = get_user_recommendations(random_user_id)
    
    return top_courses_user, top_10_recommended_courses

# Fungsi untuk mendapatkan rekomendasi berdasarkan user ID
def get_user_recommendations(user_id):
    if user_id < 1 or user_id >= num_users:  
        raise ValueError("User ID is out of bounds.")

    reviewed_course_by_user = final_df[final_df.user_id == user_id]

    courses_not_reviewed = final_df[~(final_df.name.isin(reviewed_course_by_user.name.values))]['course_id']
    courses_not_reviewed = list(set(courses_not_reviewed).intersection(set(courses_decoded.keys())))

    if not courses_not_reviewed:
        return reviewed_course_by_user, pd.DataFrame(columns=['name', 'course_url', 'rating'])

    courses_not_reviewed = [[x] for x in courses_not_reviewed]
    courses_not_reviewed = [x for x in courses_not_reviewed if x[0] < num_courses]
    
    user_courses_array = np.hstack(
        ([[user_id]] * len(courses_not_reviewed), courses_not_reviewed)
    )

    if np.any(user_courses_array[:, 0] >= num_users) or np.any(user_courses_array[:, 1] >= num_courses):
        raise ValueError("One or more course IDs are out of bounds.")

    ratings = model.predict(user_courses_array, verbose=0).flatten()
    top_ratings_indices = ratings.argsort()[-10:][::-1]

    # Fetch the courses reviewed by the user
    top_courses_user = reviewed_course_by_user.sort_values(by='rating', ascending=False).head(10)
    
    # Select top recommended courses and ensure unique names
    recommended_courses = final_df[final_df['course_id'].isin(top_ratings_indices)]
    top_10_recommended_courses = recommended_courses[['name', 'course_url', 'rating']].drop_duplicates(subset='name').head(10)

    return top_courses_user, top_10_recommended_courses

# Fungsi untuk menampilkan kursus yang pernah di-review oleh user_id 
def display_reviewed_courses(user_id):
    try:
        user_id = int(user_id)  
    except ValueError:
        st.write("User ID harus berupa angka.")
        return
    
    user_reviews = final_df[final_df['user_id'] == user_id]
    reviewed_courses = user_reviews[['name', 'course_url', 'rating', 'reviews']]
    
    # Tampilkan hasil
    if reviewed_courses.empty:
        st.write(f"User ID {user_id} belum mereview kursus apapun.")
    else:
        st.write(f"Kursus yang pernah di-review oleh User ID {user_id}:")
        st.dataframe(reviewed_courses)


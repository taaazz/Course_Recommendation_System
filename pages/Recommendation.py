import os
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers  
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Concatenate
from tensorflow.keras.models import Model
import random
from main import sidebar 

final_df = pd.read_csv('dataset/data_prep.csv')
final_rating_df = pd.read_csv('dataset/final_rating_df.csv')


class RecommenderNet(tf.keras.Model):

    def __init__(self, num_users, num_courses, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_courses = num_courses
        self.embedding_size = embedding_size

        # Matrix Factorization (MF) Embeddings
        self.user_embedding_mf = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.user_bias_mf = layers.Embedding(num_users, 1)
        self.courses_embedding_mf = layers.Embedding(
            num_courses,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.courses_bias_mf = layers.Embedding(num_courses, 1)

        # Neural Network (NN) Embeddings
        self.user_embedding_nn = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )
        self.courses_embedding_nn = layers.Embedding(
            num_courses,
            embedding_size,
            embeddings_initializer='he_normal',
            embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
        )

        # Neural Network Layers
        self.dense1 = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        self.dense2 = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)

        # Output Layer
        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_vector_mf = self.user_embedding_mf(inputs[:, 0])
        user_bias_mf = self.user_bias_mf(inputs[:, 0])
        courses_vector_mf = self.courses_embedding_mf(inputs[:, 1])
        courses_bias_mf = self.courses_bias_mf(inputs[:, 1])

        dot_user_courses_mf = tf.tensordot(user_vector_mf, courses_vector_mf, axes=2)
        x_mf = dot_user_courses_mf + user_bias_mf + courses_bias_mf

        user_vector_nn = self.user_embedding_nn(inputs[:, 0])
        courses_vector_nn = self.courses_embedding_nn(inputs[:, 1])

        user_vector_nn = tf.keras.layers.Flatten()(user_vector_nn)
        courses_vector_nn = tf.keras.layers.Flatten()(courses_vector_nn)

        concat_nn = tf.keras.layers.Concatenate()([user_vector_nn, courses_vector_nn])

        x_nn = self.dense1(concat_nn)
        x_nn = self.batch_norm1(x_nn)
        x_nn = self.dropout1(x_nn)
        x_nn = self.dense2(x_nn)
        x_nn = self.batch_norm2(x_nn)
        x_nn = self.dropout2(x_nn)

        concat_mf_nn = tf.keras.layers.Concatenate()([x_mf, x_nn])
        output = self.output_layer(concat_mf_nn)

        return output

    def get_config(self):
        config = super(RecommenderNet, self).get_config()
        config.update({
            'num_users': self.num_users,
            'num_courses': self.num_courses,
            'embedding_size': self.embedding_size,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            num_users=config['num_users'],
            num_courses=config['num_courses'],
            embedding_size=config['embedding_size']
        )


tf.keras.utils.get_custom_objects().update({'RecommenderNet': RecommenderNet})

# def load_model(model_dir): 
#     model_path = os.path.abspath("model/recommender_model.keras")
#     model= tf.keras.models.load_model(model_path)
#     return model


num_users = final_df['user_id'].nunique()
num_courses = final_df['course_id'].nunique()

embedding_size = 50  
model = RecommenderNet(num_users, num_courses, embedding_size)
# # model.save('recommender_model.keras', save_format='keras', include_optimizer=False)
tf.keras.backend.clear_session()

@st.cache_resource
def load_model_cached():
    model = tf.keras.models.load_model('model/recommender_model.keras', custom_objects={'RecommenderNet': RecommenderNet})
    return model
model = load_model_cached()

courses_decoded = dict(zip(final_df['course_id'], final_df['name']))
users_decoded = dict(zip(final_df['user_id'], final_df['user_id']))  


#Recommendation function
def get_recommendations(user_id):
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

    top_courses_user = reviewed_course_by_user.sort_values(by='rating', ascending=False).head(10)
    recommended_courses = final_rating_df[final_rating_df['course_id'].isin(top_ratings_indices)]
    top_10_recommended_courses = recommended_courses[['name', 'course_url', 'rating']].head(10)

    return top_courses_user, top_10_recommended_courses


col1, col2 = st.columns([1, 4])  

with col1:
    st.image('logo/12.png', use_column_width=True)  

with col2:
    st.title('Sistem Rekomendasi Kursus')  

sidebar()

user_id = st.number_input('Masukkan User ID (1 - 281001):', min_value=1, max_value=num_users, step=1)

if st.button('Search'):
    if user_id <= num_users:
        top_courses_user, top_10_recommended_courses = get_recommendations(user_id)
        
        st.subheader(f'Riwayat Kursus yang pernah diikuti oleh {users_decoded.get(user_id)}')
        st.table(top_courses_user[['name', 'rating']].head(10))
        
        st.subheader('Rekomendasi Kursus untuk Anda')
        for idx, row in top_10_recommended_courses.iterrows():
            st.write(f"[{row['name']}]({row['course_url']}) - Rating: {row['rating']}")
        top_courses_user, top_10_recommended_courses = get_recommendations(user_id)

       
    else:
        st.error("ID user tidak ditemukan.")

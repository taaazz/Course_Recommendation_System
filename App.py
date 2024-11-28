import streamlit as st
from main import * 
import tensorflow as tf
from main import get_recommendations
import nltk
nltk.data.path.append('./nltk_data')


# st.set_page_config(page_title='Course Recommender System', page_icon='📚')
col1, col2 = st.columns([1, 4])  

with col1:
    st.image('logo/12.png', use_column_width=True)  

with col2:
    st.title('Course Recommendation System')  

st.header('About this app')

st.subheader('🤖What can this app do?')
st.info(
    'Course recommender was designed to help users find the most relevant online courses that match their interests and needs. '
    'This app uses a Neural Collaborative Filtering (NCF) model. Built with the Coursera dataset, it provides personalized recommendations '
    'based on users’ past interactions and course preferences.'
)

st.subheader('📝How to use the app?')
st.warning(
    "To use the app, simply enter a keyword related to the type of course you're interested in (e.g., 'machine learning') "
    "in the text box provided. When you click 'Get Recommendations,' the app will analyze your input and suggest the most relevant courses based on similarity to other user interactions. "
    "You’ll receive a list of recommended courses with links to access them."
)

st.subheader('📁Datasets')
st.text('Coursera Dataset from Kaggle')

sidebar()

input_text = st.text_input('Masukkan kata kunci atau deskripsi minat kursus:')

if st.button('Get Recommendation'):
    if input_text:
        top_courses_user, top_10_recommended_courses = get_recommendations(input_text)
        
        st.subheader('Rekomendasi Kursus untuk Anda')
        for idx, row in top_10_recommended_courses.iterrows():
            st.write(f"[{row['name']}]({row['course_url']}) - Rating: {row['rating']}")
        
        st.subheader('Riwayat Kursus yang mirip dengan minat Anda')
        st.table(top_courses_user[['name', 'rating']].head(10))
        
    else:
        st.error("Harap masukkan kata kunci atau deskripsi minat.")
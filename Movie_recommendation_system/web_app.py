import streamlit as st
import pickle

movies = pickle.load(open('movies_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))
movies_list = movies['title'].values

st.header(" Movie Recommender System")
select_value = st.selectbox("Select a movie name ", movies_list)

def recommend_movies(title):
    index = movies[movies['title'] == title].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key = lambda vector: vector[1])
    recommended_movies = []
    for i in distance[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

if st.button("Show Recommended"):
    movie_names = recommend_movies(select_value)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(movie_names[0])
        st.text("\n")
    with col2:
        st.text(movie_names[1])
    with col3:
        st.text(movie_names[2])
    with col4:
        st.text(movie_names[3])
    with col5:
        st.text(movie_names[4]) 

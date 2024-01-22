import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

st.title('Movie Recommender System')

# Load data
movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')
tags_df = pd.read_csv('data/tags.csv')


##########################################################


# Preprocessing
## Calculate average rating and rating count for each movie
ratings_mean_count = ratings_df.groupby('movieId').agg({'rating': ['mean', 'count']})
ratings_mean_count.columns = ['average_rating', 'rating_count']

## Merge with movies_df
movies_df = pd.merge(movies_df, ratings_mean_count, on='movieId', how='left')

# Replace NaN values in average_rating and rating_count with 0
movies_df['average_rating'].fillna(0, inplace=True)
movies_df['rating_count'].fillna(0, inplace=True)

# Calculate the 90th percentile of the number of ratings
rating_count_percentile = movies_df['rating_count'].quantile(0.95)

# Preprocess genres
movies_df['genres'] = movies_df['genres'].str.replace('|', ' ')

# Preprocess tags
tags_df = tags_df.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x))
tags_df = pd.DataFrame(tags_df)

# Merge movies and tags dataframes
movies_with_tags = pd.merge(movies_df, tags_df, on='movieId', how='left')
movies_with_tags['tag'] = movies_with_tags['tag'].fillna('')

# Combine genres and tags to form a unified content description
movies_with_tags['description'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag']

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_with_tags['description'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


##########################################################


# Popular Movies Section in Streamlit
st.header('Popular Films')
# Filter movies with rating_count >= rating_count_90th_percentile and sort by average_rating
popular_movies = movies_df[movies_df['rating_count'] >= rating_count_percentile].sort_values(by='average_rating', ascending=False).head(10)
st.write('Top 10 Popular Movies (Top 10% by Number of Ratings):', popular_movies[['title', 'average_rating', 'rating_count']])


##########################################################


# Item-based Recommendation Section in Streamlit using Content-Based Filtering
st.header('Similar Film Recommendations Based on Content')
selected_movie = st.selectbox('Select a Movie that you like', movies_with_tags['title'].values)
if selected_movie:
    # Get the index of the movie that matches the selected title
    idx = movies_with_tags.index[movies_with_tags['title'] == selected_movie].tolist()[0]
    
    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    similar_movies = movies_with_tags['title'].iloc[movie_indices]
    
    st.write('Movies similar to', selected_movie, 'based on content:')
    st.table(similar_movies)


##########################################################


# User-based Recommendation Section in Streamlit
st.header('Personal Recommendations')
user_id = st.number_input('Enter your User ID', min_value=1)

user_movie_matrix = ratings_df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(csr_matrix(user_movie_matrix))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

if user_id in user_similarity_df.index:
    similar_users = list(enumerate(user_similarity[user_id]))
    sorted_similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[1:]
    top_similar_user = sorted_similar_users[0][0]
    user_based_recommendations = user_movie_matrix.loc[top_similar_user].sort_values(ascending=False).head(10).index
    recommended_movies = movies_df[movies_df['movieId'].isin(user_based_recommendations)]['title']
    st.write('Recommended Movies for User ID', user_id, ':')
    st.write(recommended_movies)
else:
    st.write('User ID not found in the dataset.')


##########################################################


# Hidden Treasures Section in Streamlit
st.header('Hidden Treasures')

# Filter movies with rating_count between 10 and 100 and sort by average_rating
hidden_treasures = movies_df[(movies_df['rating_count'] >= 10) & (movies_df['rating_count'] <= 100)].sort_values(by='average_rating', ascending=False).head(10)
st.write('Top 10 Hidden Treasures (Between 10 and 100 Ratings):', hidden_treasures[['title', 'average_rating', 'rating_count']])

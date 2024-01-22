import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD

# Streamlit General Configuration from movie_recommender_WBSFLIX.py
st.set_page_config(
    page_title="WBSFLIX Movie Store", 
    page_icon="🎬", 
    layout="wide"
)

page_bg_img = '''
<style>
    .stApp {
    background-image: url("https://blog.filmustage.com/content/images/2023/08/Decoding-cinema---A-deep-dive-into-film-studies-and-its-language.png");
    background-size: cover;
    }
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("🎬 The WBSFLIX Movie Store")

# Load Data from group_recommender.py
movies_path = "data/movies.csv"
ratings_path = "data/ratings.csv"
movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)
ratings_df1 = ratings_df[['userId', 'movieId', 'rating']]
movies_df['movieId'] = movies_df['movieId'].astype(str)
ratings_df['movieId'] = ratings_df['movieId'].astype(str)

# Setup for SVD and Training
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df1, reader)
full_train = data.build_full_trainset()
n = 10

# Popularity Recommender from group_recommender.py with styling from movie_recommender_WBSFLIX.py
mean_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movies_df = movies_df.merge(mean_ratings, on='movieId', how='left')
movies_df.rename(columns={'mean': 'average_rating', 'count': 'rating_count'}, inplace=True)
weight_for_mean_rating = 0.7
weight_for_number_of_ratings = 0.3
mean_ratings['weighted_score'] = (weight_for_mean_rating * mean_ratings['mean']) + (weight_for_number_of_ratings * mean_ratings['count'])
ranked_movies = mean_ratings.sort_values(by='weighted_score', ascending=False).head(10)
ranked_movies_df = ranked_movies.merge(movies_df, on='movieId', how='left')
ranked_movies_df['ranking'] = range(1, 11)
ranked_movies_df['Title'] = ranked_movies_df['title'].str.split('(').str[0]
ranked_movies_df['Year of Release'] = ranked_movies_df['title'].str.extract(r'\((\d{4})\)')
ranked_movies_df['Genres'] = ranked_movies_df['genres'].str.split('|').str[:2].str.join(', ')
ranked_movies_df = ranked_movies_df[['ranking', 'Title', 'Year of Release', 'Genres']]

st.header('Most Popular Top 10 Movies')
st.dataframe(ranked_movies_df, hide_index=True, column_config={'ranking': 'Ranking'}, width=1000)

# Item-based Recommender from group_recommender.py with styling from movie_recommender_WBSFLIX.py
st.header("👍 Enjoyed the Movie? Let's Recommend Something Similar just for You!")
chosen_item = st.selectbox(
    "Choose a Favorite Movie and We'll Suggest One for You:",
    key='item-based',
    options=movies_df['title'].unique(),
    index=0,
    placeholder="Browse our library..."
)

def get_top_n_recommendations(chosen_movieId, n):
    # User movie matrix
    user_movie_matrix = pd.pivot_table(data=ratings_df,
                                       values='rating',
                                       index='userId',
                                       columns='movieId',
                                       fill_value=0)
    # Cosine correlation matrix
    movies_df_cosines_matrix = pd.DataFrame(
        cosine_similarity(user_movie_matrix.T),
        columns=user_movie_matrix.columns,
        index=user_movie_matrix.columns
    )
    
    # Retrieve the movie ID based on the input movie title
    cosine_sim_movies = movies_df_cosines_matrix[chosen_movieId].sort_values(ascending=False)
    
    # Get the top N similar movies
    top_n_movieIds = cosine_sim_movies.iloc[1:n+1].index
    top_n_recommendations = movies_df[movies_df['movieId'].isin(top_n_movieIds)]
    
    return top_n_recommendations

if chosen_item:
    chosen_movieId = movies_df.loc[movies_df['title'] == chosen_item, 'movieId'].values[0]
    top_n_recommendations = get_top_n_recommendations(chosen_movieId, n)
    st.write(f'Because You Liked "{chosen_item}", You Might Also Like...')
    st.dataframe(
        top_n_recommendations[['title', 'genres']].assign(genres=top_n_recommendations['genres'].apply(lambda x: x.replace("|", ", "))),
        hide_index=True
    )

# User-based Recommender from group_recommender.py with styling from movie_recommender_WBSFLIX.py
st.header("🌟 Discover Your Personalized Recommendation 🍿")
user_id = st.text_input("Enter User ID:")
if st.button("Get Recommendations"):
    # Prepare data
    flix_df = ratings_df[['userId', 'movieId', 'rating']]
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(flix_df, reader)
    trainset, testset = train_test_split(data, test_size=0.25, random_state=1972)

    # Train the model
    sim_options = {'name': 'cosine', 'user_based': True}
    knn = KNNBasic(sim_options=sim_options)
    knn.fit(trainset)

    # Make predictions and get top recommendations
    full_train = data.build_full_trainset()
    testset = full_train.build_anti_testset()
    top_n_predictions_df = get_top_n(testset, int(user_id), n)
    reduced_top_n_df = top_n_predictions_df.loc[:, ["iid"]].rename(columns={"iid": "movieId"})

    # Merge with movies_df to get movie details
    user_based_df = reduced_top_n_df.merge(movies_df, how='left', on='movieId')

    st.write(f"Top {n} Recommendations for User {user_id}:")
    st.dataframe(
        user_based_df[['title', 'genres']].assign(genres=user_based_df['genres'].apply(lambda x: x.replace("|", ", "))),
        hide_index=True
    )

# Hidden Treasures Section from group_recommender.py with styling from movie_recommender_WBSFLIX.py
st.header('Hidden Treasures')
hidden_treasures = movies_df[(movies_df['rating_count'] >= 10) & (movies_df['rating_count'] <= 100)].sort_values(by='average_rating', ascending=False).head(10)
st.write('Top 10 Hidden Treasures (Between 10 and 100 Ratings):', hidden_treasures[['title', 'average_rating', 'rating_count']])

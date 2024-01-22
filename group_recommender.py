import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD



# Load Data
movies_path = "movies.csv"
ratings_path = "ratings.csv"

# Load the data into pandas DataFrames
movies_df = pd.read_csv(movies_path)
ratings_df = pd.read_csv(ratings_path)

# Select the required columns and create a new DataFrame
ratings_df1 = ratings_df[['userId', 'movieId', 'rating']]

# Convert 'movieId' to string in both DataFrames
movies_df['movieId'] = movies_df['movieId'].astype(str)
ratings_df['movieId'] = ratings_df['movieId'].astype(str)


# Setup for SVD and Training
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings_df1, reader)
full_train = data.build_full_trainset()

# Number of recommendations
n = 10

##########################################################

# Streamlit General Configuration
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
    /* Add custom styles below */
    .stApp, .stApp .css-1d391kg, .stApp .css-145kmo2, .stApp .css-xq1lnh-EmotionIconBase, .stApp .css-10trblm {
        color: white; /* This will make the text color white */
    }
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Header
st.title("🎬 The WBSFLIX Movie Store")

##########################################################


# Popularity Recommender

# Group the combined dataframe by 'movieId' and calculate the mean of ratings
mean_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
movies_df = movies_df.merge(mean_ratings, on='movieId', how='left')
movies_df.rename(columns={'mean': 'average_rating', 'count': 'rating_count'}, inplace=True)


# Define your weighting factors
weight_for_mean_rating = 0.7
weight_for_number_of_ratings = 0.3

# Calculate the weighted score for each movie
mean_ratings['weighted_score'] = (weight_for_mean_rating * mean_ratings['mean']) + (weight_for_number_of_ratings * mean_ratings['count'])

# Sort the movies based on their weighted scores in descending order to get the ranked list
ranked_movies = mean_ratings.sort_values(by='weighted_score', ascending=False).head(10)

# Merge the dataframes on the 'movieId' column
ranked_movies_df = ranked_movies.merge(movies_df, on='movieId', how='left')

# Some formatting
ranked_movies_df['ranking'] = range(1,11)
ranked_movies_df['Title'] = ranked_movies_df['title'].str.split('(').str[0]
ranked_movies_df['Year of Release'] = ranked_movies_df['title'].str.extract(r'\((\d{4})\)')
ranked_movies_df['Genres'] = ranked_movies_df['genres'].str.split('|').str[:2].str.join(', ')
ranked_movies_df = ranked_movies_df[['ranking', 'Title', 'Year of Release','Genres']]


# Create a Streamlit app
st.title('Movie Recommendations')

st.header('Most Popular Top 10 Movies')

# Display the DataFrame using the global theme settings (dark mode)
st.dataframe(ranked_movies_df,hide_index=True,column_config={'ranking' : 'Ranking'}, width=1000)


##########################################################


# Item based Recommender

def get_top_n_recommendations(chosen_movieId, n):
    
    movie_info_columns = ['title', 'genres']

    user_movie_matrix = pd.pivot_table(data=ratings_df,
                                  values='rating',
                                  index='userId',
                                  columns='movieId',
                                  fill_value=0)

    # Calculate cosine similarity matrix
    movies_cosines_matrix = pd.DataFrame(cosine_similarity(user_movie_matrix.T),
                                        columns=user_movie_matrix.columns,
                                        index=user_movie_matrix.columns)

    # Create a DataFrame using the values from 'books_cosines_matrix' for the input book.
    input_movie_cosines_df = movies_cosines_matrix[[chosen_movieId]]
    
    # Rename the column to match the input book
    input_movie_cosines_df = input_movie_cosines_df.rename(columns={chosen_movieId: 'input_movie_cosine'})

    # Remove the row with the index of the input book
    input_movie_cosines_df = input_movie_cosines_df[input_movie_cosines_df.index != chosen_movieId]

    # Sort the DataFrame by the 'input_book_cosine' column in descending order.
    input_movie_cosines_df = input_movie_cosines_df.sort_values(by="input_movie_cosine", ascending=False)

    # Find out the number of users who rated both the input book and the other books
    no_of_users_rated_both_movies = [sum((user_movie_matrix[chosen_movieId] > 0) & (user_movie_matrix[movieId] > 0)) for movieId in input_movie_cosines_df.index]
    
    # Create a column for the number of users who rated both the input book and the other books
    input_movie_cosines_df['users_who_rated_both_movies'] = no_of_users_rated_both_movies

    # Remove recommendations that have less than 10 users who rated both books.
    input_movie_cosines_df = input_movie_cosines_df[input_movie_cosines_df["users_who_rated_both_movies"] > 10]

    # Get the top 'n' recommendations
    top_n_recommendations = (input_movie_cosines_df
                            .head(n)
                            .reset_index()
                            .merge(movies_df.drop_duplicates(subset='movieId'),
                                  on='movieId',
                                  how='left')
                           [movie_info_columns + ['input_movie_cosine', 'users_who_rated_both_movies']]
                           )

    
    return top_n_recommendations


# Create a Streamlit app
st.header('Top10 movies for you based on your favorite movie')

chosen_item = st.selectbox("Choose a favorite movie and we'll suggest your Top10:",
             key='item-based',
             options=movies_df['title'].unique(),
             index=0,
             placeholder="Browse our library...")

chosen_movieId = movies_df.loc[movies_df['title'] == chosen_item, 'movieId'].values[0]

#Generate recommendations
top_n_recommendations = get_top_n_recommendations(chosen_movieId, n)

# Filter the DataFrame to include only the 'title' column
titles_df = top_n_recommendations[['title', 'genres']]
titles_df['ranking'] = range(1,11)
titles_df['Title'] = titles_df['title'].str.split('(').str[0]
titles_df['Year of Release'] = titles_df['title'].str.extract(r'\((\d{4})\)')
titles_df['Genres'] = titles_df['genres'].str.split('|').str[:2].str.join(', ')
titles_df = titles_df[['ranking', 'Title', 'Year of Release','Genres']]

# Display the DataFrame
st.dataframe(
    titles_df,
    hide_index=True,
    column_config={'ranking' : 'Ranking'},
    width = 1000
)


##########################################################


# User based Recommender

#Building the full trainset
algo = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
algo.fit(full_train)

testset = full_train.build_anti_testset()

def get_top_n(testset, user_id, n):
    # Filter the testset to include only rows with the specified user_id
    filtered_testset = [row for row in testset if row[0] == user_id]

    # Make predictions on the filtered testset
    predictions = algo.test(filtered_testset)

    # Create a DataFrame from the predictions and return the top n predictions based on the estimated ratings ('est')
    top_n_predictions_df = pd.DataFrame(predictions).nlargest(n, 'est')

    return top_n_predictions_df

# Create a Streamlit app
st.header('Personalized Recommendations for You')

# Assuming ratings_df is your DataFrame
user_ids = ratings_df['userId'].unique()

if len(user_ids) > 0:
    chosen_item = st.selectbox("Put in your User ID and we'll suggest your Top10 based on your preferences and similar choices made by users like you",
        key='user-based',
        options=user_ids
    )
else:
    st.write("PLease choose a UserId")

user_id = ratings_df.loc[ratings_df['userId'] == chosen_item, 'userId'].values[0]

#Generate recommendations
top_n_df = get_top_n(testset, user_id, n)

# Creating a DataFrame from the top_n with columns 'movieId' and 'estimated_rating'
reduced_top_n_df = top_n_df.loc[:, ["iid", "est"]].rename(columns={"iid": "movieId", "est": "rating"})

# Merging the DataFrames based on 'movieId', retaining only the matching rows
reduced_top_n_df['movieId'] = reduced_top_n_df['movieId'].astype(str)
final_df = reduced_top_n_df.merge(movies_df, on="movieId", how='left')

# Filter the DataFrame to include only the 'title' column
titles_df = final_df[['title', 'genres']]
titles_df['ranking'] = range(1,11)
titles_df['Title'] = titles_df['title'].str.split('(').str[0]
titles_df['Year of Release'] = titles_df['title'].str.extract(r'\((\d{4})\)')
titles_df['Genres'] = titles_df['genres'].str.split('|').str[:2].str.join(', ')
titles_df = titles_df[['ranking', 'Title', 'Year of Release','Genres']]

# Display the DataFrame
st.dataframe(
    titles_df,
    hide_index=True,
    column_config={'ranking' : 'Ranking'},
    width = 1000
)


##########################################################


# Hidden Treasures Section in Streamlit
st.header('Hidden Treasures')

# Filter movies with rating_count between 10 and 100 and sort by average_rating
hidden_treasures = movies_df[(movies_df['rating_count'] >= 10) & (movies_df['rating_count'] <= 100)].sort_values(by='average_rating', ascending=False).head(10)
st.write('Top 10 Hidden Treasures (Between 10 and 100 Ratings):', hidden_treasures[['title', 'average_rating', 'rating_count']])

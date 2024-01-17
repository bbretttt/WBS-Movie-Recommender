{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 1. Data Loading and Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from scipy.sparse import csr_matrix\n",
    "\n",
    "# Load datasets\n",
    "ratings = pd.read_csv('/Users/bv/Documents/WBS Data Science/Project 8 - Recommender Systems/Data/ratings.csv')\n",
    "movies = pd.read_csv('/Users/bv/Documents/WBS Data Science/Project 8 - Recommender Systems/Data/movies.csv')\n",
    "tags = pd.read_csv('/Users/bv/Documents/WBS Data Science/Project 8 - Recommender Systems/Data/tags.csv')\n",
    "\n",
    "# Preprocessing\n",
    "movies['genres'] = movies['genres'].str.replace('|', ' ')\n",
    "movies_with_tags = pd.merge(movies, tags, on='movieId', how='left')\n",
    "movies_with_tags['metadata'] = movies_with_tags[['genres', 'tag']].apply(lambda x: ' '.join(x.dropna()), axis=1)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 2. Collaborative Filtering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating a user-movie matrix\n",
    "user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)\n",
    "\n",
    "# Computing cosine similarity between users\n",
    "user_similarity = cosine_similarity(user_movie_matrix)\n",
    "user_similarity = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 3. Content-Based Filtering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Creating TF-IDF matrix for movie metadata\n",
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "tfidf_matrix = tfidf.fit_transform(movies_with_tags['metadata'])\n",
    "movie_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 4. Hybrid Recommender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def hybrid_recommender(userId, movieId):\n",
    "    # Get top 10 similar users\n",
    "    similar_users = user_similarity[userId].sort_values(ascending=False)[1:11]\n",
    "    \n",
    "    # Get movies rated by similar users\n",
    "    similar_users_movies = user_movie_matrix.loc[similar_users.index]\n",
    "    recommended_movies = similar_users_movies.mean(axis=0).sort_values(ascending=False).head(20).index.tolist()\n",
    "    \n",
    "    # Content-based recommendation\n",
    "    movie_idx = movies_with_tags.index[movies_with_tags['movieId'] == movieId].tolist()[0]\n",
    "    content_similar_movies = list(enumerate(movie_similarity[movie_idx]))\n",
    "    content_similar_movies = sorted(content_similar_movies, key=lambda x: x[1], reverse=True)[1:11]\n",
    "    \n",
    "    # Combine recommendations\n",
    "    hybrid_recommendations = set([movies_with_tags.iloc[i[0]]['movieId'] for i in content_similar_movies])\n",
    "    hybrid_recommendations.update(recommended_movies)\n",
    "    \n",
    "    return movies[movies['movieId'].isin(hybrid_recommendations)]['title']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 5. Recommendation Function Usage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0                                        Toy Story (1995)\n",
      "277                      Shawshank Redemption, The (1994)\n",
      "314                                   Forrest Gump (1994)\n",
      "514                                   Pretty Woman (1990)\n",
      "1706                                          Antz (1998)\n",
      "1757                                 Bug's Life, A (1998)\n",
      "2355                                   Toy Story 2 (1999)\n",
      "2809       Adventures of Rocky and Bullwinkle, The (2000)\n",
      "3000                     Emperor's New Groove, The (2000)\n",
      "3194                                         Shrek (2001)\n",
      "3287                                Legally Blonde (2001)\n",
      "3568                                Monsters, Inc. (2001)\n",
      "3638    Lord of the Rings: The Fellowship of the Ring,...\n",
      "4356                                Bruce Almighty (2003)\n",
      "4360                                  Finding Nemo (2003)\n",
      "4427    Pirates of the Caribbean: The Curse of the Bla...\n",
      "4644                                 Love Actually (2003)\n",
      "4800    Lord of the Rings: The Return of the King, The...\n",
      "5227                                 Notebook, The (2004)\n",
      "6062           Harry Potter and the Goblet of Fire (2005)\n",
      "6194                                     Wild, The (2006)\n",
      "6220                        Devil Wears Prada, The (2006)\n",
      "6710                              Dark Knight, The (2008)\n",
      "7039                                            Up (2009)\n",
      "7075                          (500) Days of Summer (2009)\n",
      "7355                                   Toy Story 3 (2010)\n",
      "7372                                     Inception (2010)\n",
      "7466                            King's Speech, The (2010)\n",
      "Name: title, dtype: object\n"
     ]
    }
   ],
   "source": [
    "# Example usage: Recommend movies for user with ID 10 based on their rating of movie with ID 1\n",
    "recommended_movies = hybrid_recommender(10, 1)\n",
    "print(recommended_movies)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

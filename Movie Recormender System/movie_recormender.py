import pandas as pd
import numpy as np
desired_width = 320
pd.set_option('display.width', desired_width)

r_cols = ['user_id', 'movie_id', 'rating']

ratings = pd.read_csv('ml-100K/u.data', sep='\t', encoding="ISO-8859-1", names=r_cols, usecols=range(3))

m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100K/u.item', sep='|', encoding="ISO-8859-1", names=m_cols, usecols=range(2))

ratings = pd.merge(movies, ratings)

print(ratings.head())

movieRatings = ratings.pivot_table('rating', index=['user_id'], columns=['title'])
print(movieRatings.head())

starWarsRatings = movieRatings["Star Wars (1977)"]
print(starWarsRatings.head())

similarMovies = movieRatings.corrwith(starWarsRatings)
similarMovies = similarMovies.dropna()
similarMovies = similarMovies.order(ascending=False)
df = pd.DataFrame(similarMovies)
print(df.head(20))

movieStats = ratings.groupby('title').agg({'rating': [np.size, np.mean]})

# print(movieStats.head(20))

popularMovies = movieStats['rating']['size'] >= 100

print(movieStats[popularMovies].sort([('rating', 'mean')], ascending=False)[:30])


df = movieStats[popularMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))
df = df.sort(['similarity'], ascending=False)[:15]
print(df.head(30))
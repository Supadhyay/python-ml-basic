import pandas as pd
import numpy as np
desired_width = 320
pd.set_option('display.width', desired_width)
from scipy import spatial

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100K/u.data', sep='\t', encoding="ISO-8859-1", names=r_cols, usecols=range(3))

movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})

# print(movieProperties.head())

movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
print(movieNormalizedNumRatings.head())

movieDict = {}
with open(r'ml-100k/u.item') as f:
    temp = ''
    for line in f:
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = list(map(int, genres))
        movieDict[movieID] = (name, genres, movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))

print(movieDict[1])


def ComputeDistance(a, b):
    genresA = a[1]
    genresB = b[1]
    genreDistance = spatial.distance.cosine(genresA, genresB)
    popularityA = a[2]
    popularityB = b[2]
    popularityDistance = abs(popularityA - popularityB)
    return genreDistance + popularityDistance


print(ComputeDistance(movieDict[2], movieDict[4]))
print(movieDict[2])
print(movieDict[4])


import os
import io
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Reading the files, the following code should be changes according to your data

def readFiles(path):
    for root, dirname, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)

            lines = []
            f = io.open(path, 'r', encoding='latin1')
            f.readline()

            for line in f:
                lines.append(line)
            f.close()

            message = '\n'.join(lines)
            yield path, message


# Creating data frame with filepath / message / class

def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for fileName, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(fileName)

    return DataFrame(rows, index=index)

# Fitting and running the data
data = DataFrame({'message': [], 'class': []})

# Please remember to change the following to the path on your PC, you can find a lot of training data online

# data = data.append(dataFrameFromDirectory('c:/Users/ss/Downloads/enron1/enron1/spam', 'spam'))
# data = data.append(dataFrameFromDirectory('c:/Users/ss/Downloads/enron1/enron1/ham', 'ham'))

data = data.append(dataFrameFromDirectory('', 'spam'))
data = data.append(dataFrameFromDirectory('', 'ham'))

print(data.head())

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data["message"].values)

classifier = MultinomialNB()

targets = data['class'].values
classifier.fit(counts, targets)

example = ["Free viagra now 1!!! ", 'Hi Bob how are you doing ?', 'make money free !']
example_counts = vectorizer.transform(example)
predictions = classifier.predict(example_counts)
print(predictions)

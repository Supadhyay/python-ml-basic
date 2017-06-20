import pandas as pd
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six  import  StringIO
import pydotplus

# Read the input CSV file
input_csv_file = "PastHires.csv"
df = pd.read_csv(input_csv_file, header=0)
print(df.head())

# Massaging the data to convert strings to numbers
d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
df['Employed?'] = df['Employed?'].map(d)

d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
print(df.head())

# fitting the decision tree
features = list(df.columns[:6])
print(features)

y = df['Hired']
x = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

# printing the decision tree
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png('graph.png'))


import pandas as pd
import matplotlib as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn import tree

# import data
df = pd.read_csv("DataSet_Titanic.csv")

# save variables
X = df.drop("Sobreviviente", axis=1)
y = df.Sobreviviente

# create decision tree
dtree = DecisionTreeClassifier(max_depth=3, random_state=42)

# machine training
dtree.fit(X, y)

# predictions
y_pred = dtree.predict(X)
print("Precision: ", accuracy_score(y_pred, y))

# create confusion matrix
confusion_matrix(y, y_pred)

# graphic representation confusion matrix
plot_confusion_matrix(dtree, X, y, cmap=plt.cm.Blues, values_format=".0f")

# normalized graphic representation confusion matrix
plot_confusion_matrix(dtree, X, y, cmap=plt.cm.Blues, values_format=".0f", normalize="true")

# show tree
plt.figure(figsize=(10, 8))
tree.plot_tree(dtree, filled=True, feature_names=X.columns)
plt.show()

# importance graphic
# variables
importance = dtree.feature_importances_
columns = X.columns

# create graphic
sns.barplot(columns, importance)
plt.title("Attribute importance")
plt.show()

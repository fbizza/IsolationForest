import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy.random
import matplotlib.pyplot as plt

numpy.random.seed(29)

# Get the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'
df = pd.read_csv(url, header=None)

# Split features and labels
X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

# Convert 'good' to 1 and 'bad' to -1 (bad represent the outliers)
Y = Y.replace({'g': 1, 'b': -1})

# Compute outliers/inliers counts
num_outliers = (Y == -1).sum()
num_inliers = (Y == 1).sum()

# Use isolation forest
isolation_forest = IsolationForest(contamination=0.30)
isolation_forest.fit(X)
scores = isolation_forest.decision_function(X)
anomalies = isolation_forest.predict(X)

X['score'] = scores
X['anomaly'] = anomalies
X['true'] = Y
acc = accuracy_score(Y, anomalies)

# Print results
TARGET_CLASSES = ["Outliers", "Inliers"]
print(X)
print("\nNumber of true outliers:", num_outliers)
print("Number of true inliers:", num_inliers)
print(f"\nThe accuracy of isolation forest is: {round(acc, 2)}%\n")
report = classification_report(Y, anomalies, target_names=TARGET_CLASSES)
print(report)
ConfusionMatrixDisplay.from_predictions(y_true=Y, y_pred=anomalies, cmap="BuGn", colorbar=False, display_labels=TARGET_CLASSES)
plt.show()



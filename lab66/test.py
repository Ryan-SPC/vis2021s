from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


breast = load_breast_cancer()
datacsv = pd.DataFrame(breast.data, columns=breast.feature_names) # 轉換為 data frame
datacsv.loc[:, "result"] = breast.target # 將品種代號加入 data frame

columns = datacsv.columns.tolist()
print(columns)

# 資料內容
features = datacsv.columns[:-1]
# print(features)

# 資料類別
label = "result"
# print(label)

# 資料索引
index = list(range(datacsv.shape[0]))

data = datacsv.to_numpy()
X = data[:, 0:-1]
y = data[:, -1]
# print(len(X))
# print(len(y))


# 73分拆
X_train, X_test, y_train, y_test, train_index, test_index = \
    train_test_split(X, y, index, train_size=0.7, random_state=42)


unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(y_test, return_counts=True)
print(dict(zip(unique, counts)))

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
result = clf.predict(X_train)

from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
# pred = np.argmax(pred, axis=1)
# y_true = np.argmax(y_test, axis=1)
CM = confusion_matrix(y_test, pred)
print(CM)

from sklearn.metrics import plot_confusion_matrix
disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=breast.target_names,
                                 cmap=plt.cm.Blues,
                                 values_format='d')


plt.savefig('./media/CM.png')
plt.show()
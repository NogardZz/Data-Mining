# Nạp các gói thư viện cần thiết
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# Đọc dữ liệu iris từ UCI (https://archive.ics.uci.edu/ml/datasets/Iris)
# hoặc từ thư viện scikit-learn
# Tham khảo https://scikitlearn.org/stable/auto_examples/datasets/plot_iris_dataset.html
from sklearn import datasets
from sklearn.model_selection import train_test_split
iris = datasets.load_iris()
columns = ["Petal length","Petal Width","Sepal Length","Sepal Width"]
df = pd.DataFrame(iris.data, columns=columns)
y = iris.target
print(df.describe())
print("\n")
print("Kiem tra xem du lieu co bi thieu (NULL) khong?")
print(df.isnull().sum())
# Sử dụng nghi thức kiểm tra hold-out
# Chia dữ liệu ngẫu nhiên thành 2 tập dữ liệu con:
# training set và test set theo tỷ lệ 70/30
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print(X_train.head())
# Xây dựng mô hình với k = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
# Dự đoán nhãn tập kiểm tra
prediction = model.predict(X_test)
#print(prediction)
# Tính độ chính xác
print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" % model.score(X_test, y_test))
# Sử dụng nghi thức kiểm tra chéo k-fold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Thực hiện nghi thức kiểm tra 5 fold
nFold = 5
model = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(model, df, y, cv=nFold)
print("Do chinh xac cua mo hinh voi nghi thuc kiem tra %d-fold %.3f" % (nFold, np.mean(scores)))
# Sử dụng nghi thức kiểm tra chéo k-fold
# Thực hiện nghi thức kiểm tra 5 fold
print("\nDataset::Breast Cancer Wisconsin")
for i in range(0, 4):
    nFold = i+1
    model = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(model, df, y, cv=nFold)
    print("Do chinh xac cua mo hinh voi nghi thuc kiem tra %d-fold %.3f" % (nFold, np.mean(scores)))
BCW = datasets.load_breast_cancer()
BCW_data = pd.DataFrame(BCW.data)
print(BCW_data.describe().round(4))

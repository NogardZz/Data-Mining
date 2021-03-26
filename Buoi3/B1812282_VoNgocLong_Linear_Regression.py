# Các thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np

# Nạp dữ liệu
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[ 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Xây dựng biểu đồ
plt.plot(X, y, 'ro', color="blue")
plt.axis([140, 190, 45, 100])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Xây dựng mô hình hồi quy với thư viện sklearn
import sklearn
from sklearn import linear_model
lm = linear_model.LinearRegression()
lm.fit(X, y)

X1 = np.array([[145, 185]]).T
y1 = lm.predict(X1)
plt.plot(X, y, "ro", color="blue")
plt.plot(X1, y1, color="violet")

plt.axis([140, 190, 45, 100])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

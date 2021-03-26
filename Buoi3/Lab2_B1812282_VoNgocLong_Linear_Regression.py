# Các thư viện cần thiết
import matplotlib.pyplot as plt
import numpy as np

# Nạp dữ liệu
X = np.array([[150, 147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
y = np.array([[90, 49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Xây dựng biểu đồ
plt.figure("BIỂU ĐỒ 1 ")
plt.plot(X, y, 'ro', color="red")
plt.axis([140, 190, 45, 100])  # Gioi_han_truc_X_Y
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Xây dựng mô hình hồi quy với thư viện sklearn
import sklearn
from sklearn import linear_model

lm = linear_model.LinearRegression()
lm.fit(X, y)

plt.figure("BIỂU ĐỒ DỰ BÁO BẰNG HỒI QUY TUYẾN TÍNH")
X1 = np.array([[145, 185]]).T  # Du_doan_ke_doan
y1 = lm.predict(X1)
plt.plot(X, y, "ro", color="red")
plt.plot(X1, y1, color="black")

plt.axis([140, 190, 45, 100])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

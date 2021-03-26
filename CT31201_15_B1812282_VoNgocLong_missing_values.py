# -*- coding: utf-8 -*-

"""1. Bài tập về xử lý các trường hợp dữ liệu bị thiếu

Trong bài tập này, chúng ta sẽ làm việc với bộ dữ liệu Housing Prices. 

Tập dữ liệu này có 79 giá trị mô tả (gần như) mọi khía cạnh của các căn nhà ở tại Ames, Iowa.
Với tập dữ liệu này, chúng ta cần xây dựng mô hình để dự đoán giá của mỗi ngôi nhà.

"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Đọc dữ liệu
X_full = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/housing-prices/train.csv', index_col='Id')
X_test_full = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/housing-prices/test.csv', index_col='Id')
print(X_full)

# Loại bỏ các căn nhà không có giá
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Để đơn giản bài toán, ở đây chúng ta chỉ chọn các thuộc tính số
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])
# Chia tập dữ liệu thành 2 tập dữ liệu con là training set và validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Xem các căn nhà đầu tiên trong tập dữ liệu huấn luyện. Chú ý các giá trị bị thiếu.
X_train.head()
#Dung X_train để xác định số nhà
"""
# Bước 1: Làm quen với dữ liệu
"""

# Xem mô tả tập dữ liệu huấn luyện
print(X_train.shape)

# Số lượng dữ liệu bị thiếu trong các cột
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# Hãy trả lời các câu hỏi sau đây?

# Có bao nhiêu căn nhà trong tập dữ liệu huấn luyện?
num_rows = len(X_train)
print("So can nha: ", num_rows)
# Có bao nhiêu cột dữ liệu bị thiếu?
num_cols_with_missing = len([col for col in X_train.columns
                        if X_train[col].isna().any()])
print("So cot du lieu bi mat: ", num_cols_with_missing)
# How many missing entries are contained in all of the training data?
tot_missing = 0
for col in X_train:
    tot_missing = tot_missing + sum(X_train[col].isna())
print("Tot_missing: ", tot_missing)

# Theo bạn, phương pháp nào thích hợp nhất để xử lý trường hợp bị thiếu dữ liệu này?


"""Bước 2: định nghĩa hàm để đo chất lượng của từng phương pháp

Để so sánh chất lượng của các phương pháp, chúng ta cần định nghĩa hàm score_dataset() . 
Hàm được sử dụng trong ví dụ này là hàm Trung bình của sai biệt tuyệt đối (the mean absolute error (MAE)) 
dành cho mô hình rừng ngẫu nhiên (RandomForest).
(https://en.wikipedia.org/wiki/Mean_absolute_error) 
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)


"""
# Bước 3: 

Trong các phương pháp xử lý dữ liệu bị thiếu bạn đã học,
phương pháp nào cho kết quả dự báo chính xác nhất.
Hãy cho biết giá trị MAE của từng mô hình.
"""

"Phương pháp 1"
# Lấy tên các cột có dữ liệu bị thiếu
cols_with_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any()]
# Xóa các cột này trong tập training và validation
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
print("MAE from Approach 1: ")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

"Phương pháp 2"
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("MAE from Approach 2: ")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

"Phương pháp 3"
# Để đảm bảo an toàn, tạo bản sao dữ liệu
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()
# Tạo các cột mới để đánh dấu các vị trí dữ liệu được thay thế
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()
# Phương pháp thay thế cải tiến
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns
print("MAE from Approach 3: ")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))


print("Phương pháp Loại Bỏ Cột cho dự báo chính xác nhất trong 3 phương pháp")


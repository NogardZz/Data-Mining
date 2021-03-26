# -*- coding: utf-8 -*-
"""1. Ví dụ về xử lý các trường hợp dữ liệu bị thiếu
Trong ví dụ này, chúng ta sẽ làm việc với bộ dữ liệu Melbourne Housing.
Mô hình của chúng ta sẽ sử dụng thông tin như số lượng căn phòng và kích thước miếng đất để
dự đoán giá nhà.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
# Đọc dữ liệu
data = pd.read_csv('https://raw.githubusercontent.com/ltdaovn/dataset/master/melbourne-housing-snapshot/melb_data.csv')
# Chọn chỉ số cần dự báo
y = data.Price
# Để đơn giản bài toán, ở đây chúng ta chỉ chọn các thuộc tính số
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])
# Chia tập dữ liệu thành 2 tập dữ liệu con là training set và validation set
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
"""2. Định nghĩa hàm để đo chất lượng của từng phương pháp
Chúng ta định nghĩa hàm score_dataset() để so sánh chất lượng của các phương pháp. 
Hàm được sử dụng trong ví dụ này là hàm Trung bình của sai biệt tuyệt đối (the mean absolute 
error (MAE)) 
dành cho mô hình rừng ngẫu nhiên (RandomForest).
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
"""3.a. Phương pháp 1: Xóa cột có dữ liệu bị thiếu
"""
# Lấy tên các cột có dữ liệu bị thiếu
cols_with_missing = [col for col in X_train.columns
                    if X_train[col].isnull().any()]
# Xóa các cột này trong tập training và validation
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
""" 3.b. Phương pháp 2: Thay thế
Thư viện Scikit-Learn cung cấp một lớp hiệu quả để chúng ta giải quyết các giá trị thiếu: 
SimpleImputer.
"""
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
"""# Đánh giá mô hình """
print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
"""# Câu hỏi: Hàm fit_transform thực hiện tác vụ gì? """
""" 3.c. Phương pháp 3: cải tiến phương pháp thay thế """
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
"""# Đánh giá mô hình """
print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
""" Dựa trên kết quả, chúng ta thấy phương pháp 2 kém hiệu quả hơn phương pháp 2?
"""
""" 4. Đánh giá
Câu hỏi: tại sao phương pháp phương pháp thay thế lại hiệu quả hơn phương pháp lại bỏ cột?
Gợi ý trả lời: Tập training có 10.864 dòng và 12 cột, trong đó có 03 cột chứa dữ liệu bị 
thiếu. 
Đối với mỗi cột, ít hơn một nửa số mục bị thiếu. 
Do đó, việc bỏ các cột sẽ loại bỏ rất nhiều thông tin hữu ích. 
"""
# Mô tả tập dữ liệu training
print(X_train.shape)
# Số lượng dữ liệu bị thiếu trong từng cột dữ liệu trong tập dữ liệu training
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
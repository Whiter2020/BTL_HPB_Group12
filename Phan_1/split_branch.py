import pandas as pd
import numpy as np
import os

# 1. Đọc bảng user features
user_df = pd.read_csv('user_features.csv')

# 2. Xáo trộn để chia random
user_df = user_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 3. Gán branch 1..4 lần lượt
user_df['branch'] = (np.arange(len(user_df)) % 4) + 1

# 4. Lưu ra từng file cho mỗi branch
for b in range(1, 5):
    branch_df = user_df[user_df['branch'] == b].drop(columns='branch')

    # Tạo folder nếu chưa có, nếu có rồi thì thôi
    folder_path = os.path.join('data', f'branch_{b}')
    os.makedirs(folder_path, exist_ok=True)

    # Ghi đè file customer_log.parquet nếu đã tồn tại
    file_path = os.path.join(folder_path, 'customer_log.parquet')
    branch_df.to_parquet(file_path, index=False)

print("Saved data/branch_1..4/customer_log.parquet")

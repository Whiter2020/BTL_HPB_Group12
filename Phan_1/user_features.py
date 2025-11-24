import pandas as pd
import numpy as np

# 1. Đọc dữ liệu đã xử lý
df = pd.read_csv('processed_data.csv')

# 2. Gộp theo từng khách hàng (CustomerID)
user_df = df.groupby('CustomerID').agg(
    total_spent=('TotalPrice', 'sum'),          # tổng tiền đã chi
    num_invoices=('InvoiceNo', 'nunique'),      # số hóa đơn
    num_items=('Quantity', 'sum'),              # tổng số lượng mua
    num_products=('StockCode', 'nunique'),      # số sản phẩm khác nhau
    first_purchase=('InvoiceDate', 'min'),      # lần mua đầu
    last_purchase=('InvoiceDate', 'max'),       # lần mua gần nhất
    country=('Country', lambda x: x.mode().iloc[0])  # quốc gia phổ biến nhất
).reset_index()

# 3. Thêm vài feature thời gian
user_df['purchase_span_days'] = (
    pd.to_datetime(user_df['last_purchase']) - pd.to_datetime(user_df['first_purchase'])
).dt.days

# (optional) bỏ 2 cột ngày nếu thấy thừa
# user_df = user_df.drop(columns=['first_purchase', 'last_purchase'])

# Lưu lại
user_df.to_csv('user_features.csv', index=False)

print("Done! Saved user_features.csv")

import pandas as pd

# 1. Đọc dữ liệu gốc
df = pd.read_csv('data.csv', encoding='ISO-8859-1')

# 2. Tạo schema (cast kiểu dữ liệu)
df['InvoiceNo']   = df['InvoiceNo'].astype(str)
df['StockCode']   = df['StockCode'].astype(str)
df['Description'] = df['Description'].astype(str)
df['Quantity']    = df['Quantity'].astype(int)
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['UnitPrice']   = df['UnitPrice'].astype(float)
df['CustomerID']  = df['CustomerID'].astype('float')

# 3. Xử lý missing values
df['Description'] = df['Description'].fillna('Unknown')
df = df.dropna(subset=['CustomerID'])           # bỏ dòng không có customer
df['CustomerID'] = df['CustomerID'].astype(int).astype(str)

# 4. Bỏ outlier cơ bản
df = df[~df['InvoiceNo'].str.startswith('C')]   # bỏ hoá đơn hủy
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

# 5. Tạo thêm vài feature
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['Year']       = df['InvoiceDate'].dt.year
df['Month']      = df['InvoiceDate'].dt.month
df['DayOfWeek']  = df['InvoiceDate'].dt.dayofweek
df['Hour']       = df['InvoiceDate'].dt.hour

# 6. Lưu file đã xử lý
df.to_csv('processed_data.csv', index=False)

print("Done! Processed data saved to processed_data.csv")

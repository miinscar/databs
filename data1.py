import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from google.colab import drive
from matplotlib import font_manager, rc

# Google Drive 마운트
drive.mount('/content/drive')

# 파일 경로 설정
sales2_file_path = '/content/drive/MyDrive/sales1_data.xlsx'
public_file_path = '/content/drive/MyDrive/public_data1.xlsx'

# 2021, 2022 판매 데이터를 불러오기 (종속 변수 - 판매량 데이터)
sheets_dict = pd.read_excel(sales2_file_path, sheet_name=['2021', '2022', '2023'])

# 2021, 2022, 2023 판매 데이터를 결합
sales_data_2021 = sheets_dict['2021']
sales_data_2022 = sheets_dict['2022']
sales_data_2023 = sheets_dict['2023']
train_sales_data = pd.concat([sales_data_2021, sales_data_2022, sales_data_2023], ignore_index=True)

# 판매일을 월별로 변환 (일별 데이터를 월별로 집계)
train_sales_data['판매일'] = pd.to_datetime(train_sales_data['판매일']).dt.to_period('M')

# 결측값을 0으로 대체 (판매량 데이터의 결측값)
train_sales_data['판매량'] = train_sales_data['판매량'].fillna(0)

# **판매 데이터를 월별로 집계 (판매량 합계)**
monthly_sales = train_sales_data.groupby('판매일')['판매량'].sum().reset_index()

# 공공 데이터를 불러오기 (2021년 1월부터 2024년 12월까지의 독립 변수 데이터)
public_data = pd.read_excel(public_file_path)
public_data['Date'] = pd.to_datetime(public_data['Date']).dt.to_period('M')

# 열 이름의 공백을 밑줄로 변환하여 일관성 유지
public_data.columns = public_data.columns.str.replace(' ', '_')

# 선택한 열 이름 수정
selected_columns = ['Single_Household_Growth_Rate', 'CSI', 'Consumer_Price_Index_(Average)', 
                    '소상공인_경기동향', '코로나_영향', 'HMR_매출지표']

# 2021년부터 2023년까지의 월별 데이터를 공공 데이터와 결합
public_data_train = public_data[public_data['Date'].dt.year.isin([2021, 2022, 2023])]

# 월별 판매 데이터와 공공 데이터를 'Date'와 '판매일'을 기준으로 병합
merged_data = pd.merge(monthly_sales, public_data_train, left_on='판매일', right_on='Date', how='left')

# 독립 변수와 종속 변수 설정
X_train = merged_data[selected_columns].fillna(0)  # 결측값을 0으로 채움
y_train = merged_data['판매량'].fillna(0)  # 결측값을 0으로 채움

# X_train 크기 확인
print("\nX_train 크기:", X_train.shape)
print("y_train 크기:", y_train.shape)

# XGBoost 모델을 사용하여 학습
xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
xg_reg.fit(X_train, y_train)

# 2024년 데이터를 생성하기 위해 2023년 데이터를 복사하여 만듦
public_data_2024 = public_data[public_data['Date'].dt.year == 2023].copy()
public_data_2024['Date'] = public_data_2024['Date'] + 1  # 2024년으로 설정

# 2024년 1월부터 6월까지의 데이터를 예측
X_test = public_data_2024[selected_columns][:6]

# 2024년 1월~6월 예측 (월별 데이터 예측)
y_pred = xg_reg.predict(X_test)

# 2024년 예측 값 출력
print("2024년 예측 판매량 (XGBoost, 1월~6월):", y_pred)

# 예측 결과 시각화
months_2024 = pd.date_range(start='2024-01', periods=6, freq='M')

# 2024년 예측 값을 그래프로 시각화
plt.figure(figsize=(10, 6))
plt.plot(months_2024, y_pred, marker='o', linestyle='-', color='blue', label='Predicted Sales (2024, XGBoost)')
plt.title('Predicted Monthly Sales for Jan-Jun 2024 (XGBoost)')
plt.xlabel('Month')
plt.ylabel('판매량')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

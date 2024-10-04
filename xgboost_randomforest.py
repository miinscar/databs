import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

# 데이터 불러오기
sales_data = pd.read_excel('sales_data.xlsx')  # 판매량 데이터
public_data = pd.read_excel('public_data.xlsx')  # 공공 데이터

# 데이터 병합
data = pd.merge(sales_data, public_data, on='MONTH')

# 날짜 형식 변환
data['MONTH'] = pd.to_datetime(data['MONTH'])

# 월, 연도 정보 추출
data['YEAR'] = data['MONTH'].dt.year
data['MONTH_NUM'] = data['MONTH'].dt.month

# 2023년 예측을 위한 데이터프레임 생성
predicted_sales_2023_xgb = []
predicted_sales_2023_rf = []

# 각 월별 예측 수행
for month in range(1, 13):
    # 2021년과 2022년 데이터 필터링
    monthly_data = data[data['MONTH_NUM'] == month]
    
    # 피처 및 타겟 변수 정의
    X = monthly_data[['CSI', 'CPI', 'SINGLE']]
    y = monthly_data['SALES']

    # XGBoost 모델 학습
    xgb_model = XGBRegressor()
    xgb_model.fit(X, y)

    # 랜덤 포레스트 모델 학습
    rf_model = RandomForestRegressor()
    rf_model.fit(X, y)

    # 2023년 공공 데이터 값 가져오기 (2021년과 2022년 평균 사용)
    public_data_month = public_data[public_data['MONTH'].dt.month == month]
    
    # 2023년의 공공 데이터 평균 값 계산
    if not public_data_month.empty:
        avg_csi = public_data_month['CSI'].mean()
        avg_cpi = public_data_month['CPI'].mean()
        avg_single = public_data_month['SINGLE'].mean()
    else:
        avg_csi, avg_cpi, avg_single = 0, 0, 0  # 기본값 설정

    # 2023년 예측 데이터 생성 (CPI 증가 시 판매량 감소 반영)
    X_2023 = pd.DataFrame({
        'CSI': [avg_csi],
        'CPI': [avg_cpi],
        'SINGLE': [avg_single]
    })

    # XGBoost 예측
    predicted_sales_xgb = xgb_model.predict(X_2023)
    predicted_sales_2023_xgb.append(predicted_sales_xgb[0])

    # 랜덤 포레스트 예측
    predicted_sales_rf = rf_model.predict(X_2023)
    predicted_sales_2023_rf.append(predicted_sales_rf[0])

# 결과 데이터프레임 생성
predicted_sales_2023_df = pd.DataFrame({
    'MONTH_NUM': range(1, 13),
    'SALES_XGB': predicted_sales_2023_xgb,
    'SALES_RF': predicted_sales_2023_rf,
})

# 두 모델의 평균 예측값 계산
predicted_sales_2023_df['SALES_AVERAGE'] = (predicted_sales_2023_df['SALES_XGB'] + predicted_sales_2023_df['SALES_RF']) / 2

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(data[data['YEAR'] == 2021]['MONTH_NUM'], data[data['YEAR'] == 2021]['SALES'], marker='o', label='Actual Sales 2021')
plt.plot(data[data['YEAR'] == 2022]['MONTH_NUM'], data[data['YEAR'] == 2022]['SALES'], marker='o', label='Actual Sales 2022')
plt.plot(predicted_sales_2023_df['MONTH_NUM'], predicted_sales_2023_df['SALES_AVERAGE'], marker='o', label='Predicted Sales 2023 (Average)')

# x축, y축 및 제목 설정
plt.xticks(range(1, 13), ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'])
plt.title('Sales Prediction for 2021, 2022, and 2023')
plt.xlabel('월')
plt.ylabel('판매량')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# 표 출력
final_sales_df = pd.DataFrame({
    'MONTH': ['1월', '2월', '3월', '4월', '5월', '6월', '7월', '8월', '9월', '10월', '11월', '12월'],
    'SALES_2021': data[data['YEAR'] == 2021]['SALES'].values,
    'SALES_2022': data[data['YEAR'] == 2022]['SALES'].values,
    'SALES_2023_XGB': predicted_sales_2023_xgb,
    'SALES_2023_RF': predicted_sales_2023_rf,
    'SALES_2023_AVERAGE': predicted_sales_2023_df['SALES_AVERAGE'].values
})

# 2023년 예측값 합 계산
total_predicted_sales_2023 = sum(predicted_sales_2023_df['SALES_AVERAGE'])

# 표 출력
print("\n판매량 데이터:\n")
print(final_sales_df)
print(f"\n2023년 예측 판매량 총합: {total_predicted_sales_2023}")

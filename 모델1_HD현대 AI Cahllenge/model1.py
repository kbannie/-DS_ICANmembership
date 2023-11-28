# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm 
import statsmodels.api as sm
from ISLP.models import (summarize, 
                         poly, 
                         ModelSpec as MS)


# ## 1. 전처리 작업
# ### (전처리 방법 수집)
# #### 1) 참고사이트 : 
# https://velog.io/@munhui/10.-%EB%8B%A4%EC%96%91%ED%95%9C-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%A0%84%EC%B2%98%EB%A6%AC-%EA%B8%B0%EB%B2%95
# ### 거. 데이터 info 확인 
# 범주형 데이터 유무 체크
# ### 나. 결측치 확인
# 결측치 데이터 제거 or 특정 값으로 대체
# ### 다. 중복 데이터 제거
# ### 라. 이상치 제거
# Z-Score 활용
# ### 마. 변수변환
# 제곱승
#
# ### 바. 정규화 작업
# 표준화(Standardization) 활용
#

# 가. train 데이터 불러오기
ship_train=pd.read_csv('./data/train.csv', encoding='utf8')
ship_train.head(30)

#데이터 info 확인
ship_train.info()

#나. 결측치 확인 -> null인 곳에는 평균값으로 채우기
len(ship_train)-ship_train.count()

#U_WIND 평균값 구해서 null인 곳에 해당 값 넣기
U_mean=ship_train['U_WIND'].mean()
ship_train['U_WIND'] = ship_train['U_WIND'].fillna(U_mean)

#V_WIND 평균값 구해서 null인 곳에 해당 값 넣기
V_mean=ship_train['V_WIND'].mean()
ship_train['V_WIND'] = ship_train['V_WIND'].fillna(V_mean)

#AIR_TEMPERATURE 평균값 구해서 null인 곳에 해당 값 넣기
A_mean=ship_train['AIR_TEMPERATURE'].mean()
ship_train['AIR_TEMPERATURE'] = ship_train['AIR_TEMPERATURE'].fillna(A_mean)

#BN 평균값 구해서 null인 곳에 해당 값 넣기
B_mean=ship_train['BN'].mean()
ship_train['BN'] = ship_train['BN'].fillna(B_mean)

#BREADTH 평균값 구해서 null인 곳에 해당 값 넣기
BR_mean=ship_train['BREADTH'].mean()
ship_train['BREADTH'] = ship_train['BREADTH'].fillna(BR_mean)

#DEPTH 평균값 구해서 null인 곳에 해당 값 넣기
D_mean=ship_train['DEPTH'].mean()
ship_train['DEPTH'] = ship_train['DEPTH'].fillna(D_mean)

#DRAUGHT 평균값 구해서 null인 곳에 해당 값 넣기
DR_mean=ship_train['DRAUGHT'].mean()
ship_train['DRAUGHT'] = ship_train['DRAUGHT'].fillna(DR_mean)

#LENGTH 평균값 구해서 null인 곳에 해당 값 넣기
L_mean=ship_train['LENGTH'].mean()
ship_train['LENGTH'] = ship_train['LENGTH'].fillna(L_mean)

len(ship_train)-ship_train.count()

#다. 중복된 데이터 제거
ship_train.duplicated()
ship_train[ship_train.duplicated()]

# +
#라. 이상치 제거

from scipy.stats import zscore

cols=['DIST','BREADTH','BUILT','DEADWEIGHT','LENGTH','U_WIND','V_WIND','AIR_TEMPERATURE','BN','ATA_LT','PORT_SIZE','CI_HOUR']

for column in cols:
    z_score_column = column + '_z_score'
    ship_train[z_score_column] = zscore(ship_train[column])
    ship_train = ship_train[abs(ship_train[z_score_column]) <= 3].drop(z_score_column, axis=1)

len(ship_train)
# -

#마. 변수변환
y=ship_train.CI_HOUR
plt.subplot(121)
plt.hist(y)
plt.subplot(122)
plt.hist(y**0.25)
ship_train.CI_HOUR

#음수값을 Nan
ship_train['y_CI_HOUR'] = np.where(ship_train['CI_HOUR'] >= 0, ship_train['CI_HOUR']**0.25,0)
ship_train.y_CI_HOUR
plt.hist(ship_train.y_CI_HOUR)
zero_count = (ship_train['y_CI_HOUR'] == 0).sum()
zero_count

ship_train = ship_train[ship_train['y_CI_HOUR'] != 0]

# 표준 정규 분포로 변환
ship_train['y_CI_HOUR'] = (ship_train['y_CI_HOUR'] - np.mean(ship_train['y_CI_HOUR']))/np.std(ship_train['y_CI_HOUR'], ddof=1)
plt.hist(ship_train.y_CI_HOUR)
ship_train.y_CI_HOUR

# +
#마. 정규화 작업
cols=['DIST','BREADTH','BUILT','DEADWEIGHT','LENGTH','U_WIND','V_WIND','AIR_TEMPERATURE','BN','ATA_LT','PORT_SIZE','CI_HOUR']

ship_train[cols]= (ship_train[cols]-ship_train[cols].mean())/ship_train[cols].std()
ship_train.head()
# -

# ## 2. 모델링
# ### 가. 변수 선택
# 1) 요소들을 영향 정도를 파악하기 위해 pariplot을 이용하여 관계 확인
#
# 2) 영향이 큰 변수 나열
#
# ### 나. 모델링
#
# 1) RandomForestRegressor :test데이터에 종속변수인 CI_HOUR가 없어서 RandomForestRegressor 모델링 후 Xtest를 넣어 ytest를 예측하기
#
# 2) 기존 train값들에서 train, test로 나눈 후 모델링 수행하기

ship_train.columns

data=ship_train[['DIST', 'ATA',
       'ID', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT',
       'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE',
       'BN', 'ATA_LT', 'PORT_SIZE','y_CI_HOUR']]
data.describe()

# +

# import seaborn as sns
# columns_to_plot = ['y_CI_HOUR', 'DIST', 'ATA', 'ID', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT',
#                    'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN', 'ATA_LT', 'PORT_SIZE']

# sns.pairplot(data[columns_to_plot])
# -

#상관계수 값에 따라 영향이 큰 특성들 추출하기
cor = data[['y_CI_HOUR', 'DIST', 'ATA', 'ID', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT',
                   'LENGTH', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE', 'BN', 'ATA_LT', 'PORT_SIZE']].corr()['y_CI_HOUR']
colhigh = cor[np.abs(cor) > 0.12].index
colhigh

ship_train[colhigh].head()

new_ship=ship_train[colhigh]
new_ship.head()

# +
# new_ship_test=ship_test[['DIST', 'BREADTH', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT',
#        'PORT_SIZE']]
# new_ship_test.head()

# +
# # RandomForestRegressor 활용
# ytrain=new_ship.y_CI_HOUR
# Xtrain=new_ship.iloc[:,1:]

# Xtest=new_ship_test

# +
#시간 관계상 해당 예측값은 중지시킴
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# model = RandomForestRegressor()
# model.fit(Xtrain, ytrain)

# predictions=model.predict(Xtest) #예측값 산출
# predictions

# +
#GAM 활용
from sklearn.model_selection import train_test_split

y = new_ship.y_CI_HOUR
X = new_ship.iloc[:,1:]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.1,random_state = 1)
Xtrain.columns
# -

from pygam import LinearGAM, s, f, l
gam = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)).fit(Xtrain,ytrain)

gam.summary

# +
plt.figure(figsize=(20, 10))

for i in range(X.shape[1]):
    plt.subplot(3, 4, i + 1)
    XX = gam.generate_X_grid(term=i)
    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c="r", ls="--")
    plt.title(X.columns[i])
    plt.ylabel("popular")

plt.tight_layout()
plt.show()
# -

np.mean((ytest - gam.predict(Xtest))**2)

# ## 추가작업

ship_train.nunique()

ship_train.dtypes

ship_train.shape

# ARI_CO(도착항의 소속국가)와 FLAG(선박의 국적)를 사용해볼 거임

ship_train['ARI_CO'].value_counts()

# +
pd.set_option('display.max_rows', None)  # 모든 행을 출력하도록 설정

flag_counts = ship_train['FLAG'].value_counts()
print(flag_counts)

pd.reset_option('display.max_rows')
# -

ship_train['FLAG']=ship_train['FLAG'].str.split(' & ').str[0] #flag의 & 데이터 처리

ari_co_counts = ship_train['ARI_CO'].value_counts()
ship_train['ari_co_counts'] = ship_train['ARI_CO'].map(ari_co_counts)

flag_counts=ship_train['FLAG'].value_counts()
ship_train['flag_counts'] = ship_train['FLAG'].map(flag_counts)

ship_train['flag_counts']

ship_train.columns

ship_train.head()

# +
data=ship_train[['y_CI_HOUR','DIST', 'ATA',
       'ID', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT',
       'LENGTH', 'SHIPMANAGER', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE',
       'BN', 'ATA_LT', 'PORT_SIZE','ari_co_counts',
       'flag_counts']]
# data.describe()

cor = data[['y_CI_HOUR','DIST', 'ATA',
       'ID', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH', 'DRAUGHT', 'GT',
       'LENGTH', 'SHIPMANAGER', 'U_WIND', 'V_WIND', 'AIR_TEMPERATURE',
       'BN', 'ATA_LT', 'PORT_SIZE','ari_co_counts',
       'flag_counts']].corr()['y_CI_HOUR']
colhigh = cor[np.abs(cor) > 0.05].index
colhigh
# -

new_ship2=ship_train[colhigh]
new_ship2.head()

# +
y = new_ship2.y_CI_HOUR
X = new_ship2.iloc[:,1:]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.1,random_state = 1)
Xtrain.columns
# -

gam = LinearGAM(s(0)+s(1)+s(2)+s(3)+s(4)+s(5)+s(6)+s(7)+s(8)+s(9)+s(10)).fit(Xtrain, ytrain)

# +
plt.figure(figsize=(20, 20))

for i in range(Xtrain.shape[1]):
    plt.subplot(3, 4, i + 1)
    XX = gam.generate_X_grid(term=i)
    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))
    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX, width=0.95)[1], c="r", ls="--")
    plt.title(Xtrain.columns[i])
    plt.ylabel("CI_HOUR")

plt.tight_layout()
plt.show()
# -

np.mean((ytest - gam.predict(Xtest))**2)

gam.predict(Xtest)

Xtest

# ## Test 제출하기

# +
#test 결측치 처리하기
ship_test=pd.read_csv('./data/test.csv', encoding='utf8')
len(ship_test)-ship_test.count()

#U_WIND 평균값 구해서 null인 곳에 해당 값 넣기
U_mean=ship_test['U_WIND'].mean()
ship_test['U_WIND'] = ship_test['U_WIND'].fillna(U_mean)

#V_WIND 평균값 구해서 null인 곳에 해당 값 넣기
V_mean=ship_test['V_WIND'].mean()
ship_test['V_WIND'] = ship_test['V_WIND'].fillna(V_mean)

#AIR_TEMPERATURE 평균값 구해서 null인 곳에 해당 값 넣기
A_mean=ship_test['AIR_TEMPERATURE'].mean()
ship_test['AIR_TEMPERATURE'] = ship_test['AIR_TEMPERATURE'].fillna(A_mean)

#BN 평균값 구해서 null인 곳에 해당 값 넣기
B_mean=ship_test['BN'].mean()
ship_test['BN'] = ship_test['BN'].fillna(B_mean)

len(ship_test)-ship_test.count()

#마. 정규화 작업
cols=['DIST','BREADTH','BUILT','DEADWEIGHT','LENGTH','U_WIND','V_WIND','AIR_TEMPERATURE','BN','ATA_LT','PORT_SIZE']

ship_test[cols]= (ship_test[cols]-ship_test[cols].mean())/ship_test[cols].std()
ship_test


# +
ship_test['ARI_CO'].value_counts()

pd.set_option('display.max_rows', None)  # 모든 행을 출력하도록 설정

flag_counts = ship_test['FLAG'].value_counts()

pd.reset_option('display.max_rows')

ship_test['FLAG']=ship_test['FLAG'].str.split(' & ').str[0] #flag의 & 데이터 처리

ari_co_counts = ship_test['ARI_CO'].value_counts()
ship_test['ari_co_counts'] = ship_test['ARI_CO'].map(ari_co_counts)

flag_counts=ship_test['FLAG'].value_counts()
ship_test['flag_counts'] = ship_test['FLAG'].map(flag_counts)

ship_test.columns

# +
# 예측값 산출
new_ship_test=ship_test[['DIST', 'BREADTH', 'BUILT', 'DEADWEIGHT', 'DEPTH',
       'DRAUGHT', 'GT', 'LENGTH', 'PORT_SIZE', 'ari_co_counts', 'flag_counts']]

new_ship_test.dtypes
predictions = gam.predict(new_ship_test)

# -

predictions

new_ship_test

Xtest

ship_test

# +
# Xtest에 있는 'sample_ID'를 가져오기 (가정)
sample_IDs = ship_test['SAMPLE_ID']

# 'sample_ID'와 'predictions'를 포함하는 DataFrame 생성
top_predictions = pd.DataFrame({'SAMPLE_ID': sample_IDs, 'CI_HOUR': predictions})
# -

top_predictions

top_predictions['CI_HOUR'] = top_predictions['CI_HOUR'].apply(lambda x: max(0, x))
top_predictions

# CSV 파일로 저장
top_predictions.to_csv('./sample_submission.csv', index=False)



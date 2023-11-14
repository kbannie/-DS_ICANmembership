import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm 
from ISLP.models import (summarize, poly, ModelSpec as MS)
import statsmodels.api as sm

# train 데이터 불러오기
ship_train=pd.read_csv('./data/train.csv', encoding='utf8')
ship_train.info()

# 변수 선언
dist=ship_train['DIST']
breadth=ship_train['BREADTH']
built=ship_train['BUILT']
deadweight=ship_train['DEADWEIGHT']
depth=ship_train['DEPTH']
gt=ship_train['GT']
length=ship_train['LENGTH']
postSize=ship_train['PORT_SIZE']

# Anova 변수 선택
y=ship_train['CI_HOUR']
models = [MS([poly('dist', degree=d), poly('breadth', degree=d)]) for d in range(1, 5)]
XEs = [model.fit_transform(ship_train) for model in models]

# ANOVA 실행
anova_results = anova_lm(*[sm.OLS(y, X_).fit() for X_ in XEs])
print(anova_results)

# GAM 고차회귀모형
X = pd.DataFrame({'dist' : ship_train.DIST, 'dist2' : ship_train.DIST**2, 'breadth':ship_train.BREADTH})
X = sm.add_constant(X)
reg = sm.OLS(y, X)
reg_fit = reg.fit()
reg_fit.summary()
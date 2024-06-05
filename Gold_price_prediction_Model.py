import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.pipeline import Pipeline


data = pd.read_csv("gold_price.csv").dropna()
X = data.drop("Price",axis=1)
y = data["Price"]

trans = ColumnTransformer([
    ("encode",OneHotEncoder(),["Date","Vol"]),
    ("scale",RobustScaler(),["Open","High","Low","Change"])
],remainder='passthrough')

model = Pipeline([
    ("t",trans),
    ("clf",ElasticNet(alpha=0.01,max_iter=1000))
])

model.fit(X,y)
y_pred = model.predict(X)
print(r2_score(y,y_pred))
print(mean_squared_error(y,y_pred))

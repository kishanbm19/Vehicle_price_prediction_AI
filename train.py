import pandas as pd
import joblib as j
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.model_selection import train_test_split
from preprocess import pre_process
df=pd.read_csv("Dataset/used_car_dataset_10000.csv")\

df=pre_process(df)


x=df.drop(['car_price_in_rupees'],axis=1)
y=df['car_price_in_rupees']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

models={
    "Linear Regression":LinearRegression(),
    "Decision tree":DecisionTreeRegressor(random_state=42),
    "Random forest":RandomForestRegressor(n_estimators=100,random_state=42)
}
best_sc=-1
best_model=float("-1")
best_name=""
for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    r2=r2_score(y_test,y_pred)
    mae=mean_absolute_error(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)**0.5

    print("\n",name)
    print(f"r2 : {r2:.2f}")
    print(f"mae: {mae:.2f}")
    print(f"mse: {mse:.2f} ")

    if r2>best_sc:
        best_sc=r2
        best_model=model
        best_name=name


print("best model",best_name)
j.dump(best_model,"model/vehicle_price_model.pkl")
print("Models are trained and saved successfully")


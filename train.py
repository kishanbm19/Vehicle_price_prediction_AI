import pandas as pd
import joblib as j
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

from sklearn.model_selection import train_test_split
from preprocess import pre_process



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

df=pd.read_csv("Dataset/used_car_dataset_5000_realistic.csv")\

df=pre_process(df)


x=df.drop(['car_price_in_rupees'],axis=1)
y=df['car_price_in_rupees']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# caterogerical and numerical transform
cat_features=["car_name","fuel_type","city"]
num_features=["kms_driven","car_age"]

preprocessor=ColumnTransformer(
transformers=[
    (
        "cat",
        OneHotEncoder(handle_unknown="ignore"),
        cat_features

    ),
    (
        "num",
        "passthrough",
        num_features
    )
])

models={
    "Linear Regression":Pipeline([
        ("preprocessor",preprocessor),
        ("regressor",LinearRegression())
        ]),
    "Decision tree":Pipeline([
        ("preprocessor",preprocessor),
        ("regressor",DecisionTreeRegressor(random_state=42))
    ]),
    "Random forest":Pipeline([
        ("preprocessor",preprocessor),
        ("regressor",RandomForestRegressor(n_estimators=100,random_state=42))
    ])
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




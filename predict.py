import joblib
import pandas as pd
model=joblib.load("model/vehicle_price_model.pkl")





car=str(input("Enter Car name:"))
kms=float(input("Enter kilometres driven: "))
fuel=str(input("Enter the Fuel type "))
city=str(input("Enter the city: "))
yr=int(input("Enter the year of manufacture"))

car_price=pd.DataFrame({
    "car_name":[car],
    "kms_driven":[kms],
    "fuel_type":[fuel],
    "city":[city],
    "year_of_manufacture":[yr]


})

prediction=model.predict(car_price)


print(f"\nCar Price is-> ₹{prediction[0]:.2f}" )
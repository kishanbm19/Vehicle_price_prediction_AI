import joblib
import pandas as pd
model=joblib.load("model/vehicle_price_model.pkl")





car=str(input("Enter Car name:")).strip().lower()
kms=float(input("Enter kilometres driven: "))
fuel=str(input("Enter the Fuel type: ")).strip().lower()
city=str(input("Enter the city: ")).strip().lower
yr=int(input("Enter the car age: "))

car_price=pd.DataFrame({
    "car_name":[car],
    "kms_driven":[kms],
    "fuel_type":[fuel],
    "city":[city],
    "car_age":[yr]


})

prediction=model.predict(car_price)


print(f"\nCar estimated selling Price is-> ₹{prediction[0]:.2f}" )
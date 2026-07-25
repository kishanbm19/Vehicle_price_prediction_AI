# 🚗 Vehicle Price Predictor

A Machine Learning project that predicts the selling price of a used vehicle based on its specifications. The model is trained using Scikit-learn with a preprocessing pipeline, enabling efficient handling of both numerical and categorical features.

---

## 📌 Features

* Predicts the estimated selling price of a used vehicle.
* Uses Scikit-learn Pipeline for preprocessing and model training.
* Handles categorical features using One-Hot Encoding.
* Supports unseen categories using `handle_unknown="ignore"`.
* Provides predictions through a simple command-line interface.
* Saves the trained model using Joblib for future predictions.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Git & GitHub

---

## 📂 Project Structure

```text
Vehicle-Price-Predictor/
│
├── Dataset/
│   └── used_car_dataset_5000_realistic.csv
│
├── model/
│   └── vehicle_price_model.pkl
│
├── preprocess.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

The dataset contains information about used vehicles, including:

* Car Name
* Selling Price
* Kilometres Driven
* Fuel Type
* City
* Manufacturing Year

During preprocessing, the manufacturing year is converted into **Car Age**, which is used as a feature for training.

---

## ⚙️ Machine Learning Pipeline

The project uses **Scikit-learn Pipeline** and **ColumnTransformer**.

### Numerical Features

* Kilometres Driven
* Car Age

### Categorical Features

* Car Name
* Fuel Type
* City

The pipeline performs:

* One-Hot Encoding for categorical features.
* Pass-through for numerical features.
* Model training using multiple regression algorithms.

---

## 🤖 Models Trained

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

The best-performing model is saved automatically as:

```text
model/vehicle_price_model.pkl
```

---

## ▶️ Training the Model

```bash
python train.py
```

This trains the model and stores the trained model inside the `model/` directory.

---

## 🔮 Making Predictions

Run:

```bash
python predict.py
```

Example:

```text
Enter Car name: Hyundai Grand i10 Magna
Enter kilometres driven: 39664
Enter Fuel type: CNG
Enter City: Delhi
Enter Car Age: 3
```

Example Output:

```text
Estimated Selling Price:
₹6,50,000
```

---

## 📈 Future Improvements

* Train the model on a larger real-world dataset.
* Include additional features such as transmission, ownership, engine capacity, mileage, and seller type.
* Develop a web interface using Flask or Django.
* Deploy the model for online predictions.
* Compare additional regression models and improve prediction accuracy.

---

## 📦 Requirements

```text
pandas
numpy
scikit-learn
joblib
```

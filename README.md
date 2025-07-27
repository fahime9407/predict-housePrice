
# 🏠 Tehran House Price Predictor

This project predicts house prices in Tehran using a polynomial regression model. It includes data preprocessing, feature selection, model training, and a GUI for real-time predictions.

---

## 📂 Features

- Data cleaning and outlier removal
- Feature scaling and encoding
- Polynomial regression (degree = 2)
- Evaluation with R², MAE, MSE
- User-friendly GUI built with Tkinter

---

## 📊 Dataset

The dataset (`housePrice.csv`) contains:

- `Area` (m²)
- `Room` (number of rooms)
- `Parking`, `Warehouse`, `Elevator` (booleans)
- `Address` (Tehran neighborhood)
- `Price(USD)` – Target variable

> ✅ **The dataset and all required libraries are included in this repository.**

---

## 🚀 How to Run

### Option 1: Run the Jupyter Notebook
If you want to view the full pipeline step-by-step:

```bash
jupyter notebook HousePricePredictor.ipynb
````

### Option 2: Run the Python Script

If you just want to use the trained model with GUI:

```bash
python house_price_predictor.py
```

This will launch a **Tkinter GUI** where you can input house features and receive a predicted price in USD.

---

## 🧠 Model Info

* Model: Polynomial Linear Regression
* Encoding: Target Encoding for `Address`
* Input features: `Area`, `Room`, `Parking`, `Address_Encoded`
* Normalized using `StandardScaler`

---

## 🖥 GUI Demo

The GUI allows you to enter:

* House area (m²)
* Number of rooms
* Parking availability (Yes/No)
* Neighborhood (Tehran)

🧮 It returns the predicted price based on the trained model.

---

## 📈 Sample Output

* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **R² Score**

(*Displayed in the terminal after evaluating on test data*)

---

## 📌 Author

**Fahimeh** – IT graduate & Python developer
📫 Feel free to connect or give feedback!

---

## ✅ Future Ideas

* Add more features (e.g., year built, floor number)
* Export trained model using `joblib`
* Build a web app version with Streamlit or Flask

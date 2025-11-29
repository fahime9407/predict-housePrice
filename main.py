import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tkinter import *
from tkinter import ttk


# Define Dataset
df = pd.read_csv("housePrice.csv") # Price(USD) is the label.
df = df.dropna()

# preprocessing
# convert the value of "Area" from object to int64 and if its non-convertible the value will be nan.
df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df = df.dropna()

# this cell drops outlier datas, I determined that if value of "Area" is more than 400 It is outlier.
Q1 = df["Area"].quantile(0.25)
Q3 = df["Area"].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 6.2 * IQR
df["Area"] = df["Area"].where(df["Area"] <= upper_bound, np.nan) # value if less than 480 else nan.
df = df.dropna()

# convert boolean values to int64.
df["Parking"] = df["Parking"].astype(int)
df["Warehouse"] = df["Warehouse"].astype(int)
df["Elevator"] = df["Elevator"].astype(int)

# split data to train and test.
msk = np.random.rand(len(df)) < 0.8
train, test = df[msk], df[~msk]

# I used target encoding to encode "Address".
train, test = train.copy(), test.copy() # because of warning message.
mean_prices = train.groupby("Address")["Price(USD)"].mean()
train["Address_Encoded"] = train["Address"].map(mean_prices)
test["Address_Encoded"] = test["Address"].map(mean_prices)# encode address for test data with the means of train data.
global_price = train["Price(USD)"].mean()
train.drop("Address", axis=1, inplace=True) # we dont need "Address" anymore.
test.drop("Address", axis=1, inplace=True)
test = test.dropna()

# to see correlation between features and label, it is important to use train data for this process.
corr = train.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# we use features with hight correlations for training.
x_train = np.asanyarray(train[["Area", "Room", "Parking", "Address_Encoded"]])
y_train = np.asanyarray(train[["Price(USD)"]])

x_test = np.asanyarray(test[["Area", "Room", "Parking", "Address_Encoded"]])
y_test = np.asanyarray(test[["Price(USD)"]])

# this cell normalizes train  and test datas.
x_scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
y_scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
x_scaler.fit_transform(x_train); x_scaler.transform(x_test) 
y_scaler.fit_transform(y_train); y_scaler.transform(y_test)


# Polynomial Regression
# Model Definition

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test) # transform test data for polynomial model with train metrics.

mymodel = LinearRegression()
mymodel.fit(x_train_poly, y_train)

# Model Evaluation
y_pred = mymodel.predict(x_test_poly)

mse = np.mean((y_test - y_pred) ** 2)
mae = np.mean(np.absolute(y_test - y_pred))
r2 = r2_score(y_test, y_pred)

# show evaluation result with tkinter
r = Tk()
r.title("Evaluation Result")
r.geometry("280x180")
l1 = Label(r, text=f"mean squared error : ".title()).place(x=10, y=20)
l1_result = Label(r)
l1_result.place(x=180, y=20)
l1_result.config(text=f"{mse:.5f}")
l2 = Label(r, text="mean absolute error : ".title()).place(x=10, y=60)
l2_result = Label(r)
l2_result.place(x=180, y=60)
l2_result.config(text=f"{mae:.5f}")
l3 = Label(r, text="r2 score : ".title()).place(x=10, y=100)
l3_result = Label(r)
l3_result.place(x=180, y=100)
l3_result.config(text=f"{r2:.5f}")
b = Button(r, text="OK", command=r.destroy, width=35).place(x=13, y=138)
r.mainloop()

# I used tkinter to recieve information from user and show the result to user.
def encode_address(address):
    return mean_prices.get(address, global_price)

def predict():
    try:
        area, room, parking, address = float(entry1.get()), int(entry2.get()), combo_box.get(), entry4.get()
        parking = 1 if parking == "Yes" else 0
        address_encoded = encode_address(address)
        x = np.asanyarray([[area, room, parking, address_encoded]])
        x_scaler.transform(x)
        x_poly = poly.transform(x)
        y = mymodel.predict(x_poly)
        y_scaler.inverse_transform(y)
        label_result.config(text=f"my prediction : {y[0][0]:.3f} $".title())
    except ValueError:
        label_result.config(text="invalid input!".title())


root = Tk()
root.title("Predictor")
root.geometry("350x300")
label1 = Label(root, text= "House area (m²) : ").place(x= 10, y= 20) # Enter Area as a number.
label2 = Label(root, text= "Number of rooms : ").place(x= 10, y= 60) # Enter number of rooms as e integer.
label3 = Label(root, text= "Parking available: (Yes / No) : ").place(x= 10, y= 100) # Yes if it has parking else no.
label4 = Label(root, text= "Neighborhood (in Tehran) : ").place(x= 10, y= 140) # Enter the neghborhood.
entry1 = Entry(root, width= 21)
entry1.place(x= 185, y= 20)
entry2 = Entry(root, width= 21)
entry2.place(x= 185, y= 60)
combo_box = ttk.Combobox(root, values= ["Yes", "No"], width= 18)
combo_box.place(x= 185, y= 100)
entry4 = Entry(root, width= 21)
entry4.place(x= 185, y= 140)
button1 = Button(root, text= "Predict", width= 45, command= predict, activebackground= "green", activeforeground= "red")
button1.place(x= 15, y= 180)
label_result = Label(root, text= "result .....", bg= "lightgray", font='Helvetica 15 bold')
label_result.place(x= 10, y= 218)
button2 = Button(root, text= "Done", command= root.destroy, width= 45, activebackground= "black", activeforeground= "white").place(x= 15, y= 260)
root.mainloop()

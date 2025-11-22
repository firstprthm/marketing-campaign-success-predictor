# Marketing Campaign Success Predictor - Predicts marketing success using XGBoost method.
# Copyright (C) 2025  Pratham Prabhakar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import pandas as pd

# Load trained model (update path if necessary)
model = joblib.load("./xgb_marketing_model.pkl")

def predict():
    ad_intensity = ad_intensity_var.get()
    product_type = product_type_var.get()
    income_range = income_var.get()
    price_range = price_var.get()
    discount = discount_var.get()
    credit = credit_var.get()
    age = age_var.get()

    # Updated ranges with finer product price granularity
    income_ranges = {
        "₹1,00,000 - ₹3,00,000": (100000, 300000),
        "₹3,00,001 - ₹5,00,000": (300001, 500000),
        "₹5,00,001 - ₹10,00,000": (500001, 1000000),
        "₹10,00,001 - ₹25,00,000": (1000001, 2500000),
        "₹25,00,001 - ₹50,00,000": (2500001, 5000000)
    }

    price_ranges = {
        "₹10,000 - ₹20,000": (10000, 20000),
        "₹20,001 - ₹50,000": (20001, 50000),
        "₹50,001 - ₹1,00,000": (50001, 100000),
        "₹1,00,001 - ₹5,00,000": (100001, 500000),
        "₹5,00,001 - ₹10,00,000": (500001, 1000000)
    }

    ad_map = {"Low": 0, "Medium": 1, "High": 2}
    prod_map = {"Necessity": 0, "Luxury": 1}

    income = sum(income_ranges[income_range]) / 2
    price = sum(price_ranges[price_range]) / 2

    # Derived features (keep consistent with trained model)
    afford_ratio = price / income
    ad_val = {0: 100, 1: 200, 2: 350}[ad_map[ad_intensity]]
    log_price = np.log1p(price)
    log_income = np.log1p(income)
    disc_to_aff = discount * (1 - afford_ratio)
    cred_aff_inter = credit * (1 - afford_ratio)
    income_to_credit = income / credit
    disc_intensity = discount * ad_val
    price_to_income = price / income

    X_input = pd.DataFrame([{
        "Age": age,
        "Annual_Income": income,
        "Credit_Score": credit,
        "Product_Type_enc": prod_map[product_type],
        "Product_Price": price,
        "Discount_Offered(%)": discount,
        "Affordability_Ratio": afford_ratio,
        "Ad_Calls": 10, "Ad_SMS": 100, "Ad_Social": 50, "Ad_Display": 10,
        "Ad_Intensity_Num": ad_map[ad_intensity],
        "Ad_Intensity_Value": ad_val,
        "Normalized_Ad_Intensity": ad_val / 400,
        "Log_Price": log_price, "Log_Income": log_income,
        "Discount_to_Afford": disc_to_aff,
        "Credit_Afford_Interaction": cred_aff_inter,
        "Income_to_Credit": income_to_credit,
        "Discount_Intensity": disc_intensity,
        "Price_to_Income": price_to_income
    }])

    prob = model.predict_proba(X_input)[:, 1][0] * 100
    msg = (
        f"YES ✅\nSuccess Probability: {prob:.2f}%"
        if prob >= 50
        else f"NO ❌\nSuccess Probability: {prob:.2f}%"
    )
    messagebox.showinfo("Prediction", msg)


# ---- UI Layout ----
root = tk.Tk()
root.title("Marketing Campaign Success Predictor")

tk.Label(root, text="📢 Ad Intensity").grid(row=0, column=0)
ad_intensity_var = ttk.Combobox(root, values=["Low", "Medium", "High"])
ad_intensity_var.grid(row=0, column=1)

tk.Label(root, text="🛍️ Product Type").grid(row=1, column=0)
product_type_var = ttk.Combobox(root, values=["Necessity", "Luxury"])
product_type_var.grid(row=1, column=1)

tk.Label(root, text="💰 Annual Income").grid(row=2, column=0)
income_var = ttk.Combobox(
    root,
    values=[
        "₹1,00,000 - ₹3,00,000",
        "₹3,00,001 - ₹5,00,000",
        "₹5,00,001 - ₹10,00,000",
        "₹10,00,001 - ₹25,00,000",
        "₹25,00,001 - ₹50,00,000",
    ],
)
income_var.grid(row=2, column=1)

tk.Label(root, text="🏷 Product Price").grid(row=3, column=0)
price_var = ttk.Combobox(
    root,
    values=[
        "₹10,000 - ₹20,000",
        "₹20,001 - ₹50,000",
        "₹50,001 - ₹1,00,000",
        "₹1,00,001 - ₹5,00,000",
        "₹5,00,001 - ₹10,00,000",
    ],
)
price_var.grid(row=3, column=1)

tk.Label(root, text="Discount (%)").grid(row=4, column=0)
discount_var = tk.Scale(root, from_=5, to=45, orient="horizontal")
discount_var.grid(row=4, column=1)

tk.Label(root, text="Credit Score").grid(row=5, column=0)
credit_var = tk.Scale(root, from_=300, to=900, orient="horizontal")
credit_var.grid(row=5, column=1)

tk.Label(root, text="Age").grid(row=6, column=0)
age_var = tk.Scale(root, from_=18, to=60, orient="horizontal")
age_var.grid(row=6, column=1)

tk.Button(root, text="Predict", command=predict, bg="#06d6a0").grid(
    row=7, column=0, columnspan=2, pady=10
)

root.mainloop()

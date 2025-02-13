import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
print("\n Veri Seti Oluşturuluyor")

data = {
    "TV": [230,44,17,151,180,8,57,120,100,200],
    "Radio": [37,39,45,41,10,2,20,35,15,23],
    "Newspaper": [69,45,78,20,15,10,25,14,50,20],
    "Sales": [22,10,9,18,19,5,8,15,12,21],
}
df = pd.DataFrame(data)

print("Lineer Regresyon Veri Seti:")
print(df.head())

print("\n Veri Eğitim ve TEst Setine Ayrtılıyor....")
X = df[["TV","Radio","Newspaper"]]
y = df["Sales"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print(f"Eğitim Seti Boyutu: {X_train.shape}")
print(f"Test Seti Boyutu: {X_test.shape}")

print("\n Lineer Regresyon Modeli Eğitiliyor...\n")

model = LinearRegression()

model.fit(X_train, y_train)

print("\n Model Test Verisi ile Tahmin Yapıyor....\n")
y_pred = model.predict(X_test)

print("\n Gerçek vs Tahmin Edilen Değerler:")
for gerçek, tahmin in zip(y_test, y_pred):
    print(f"Gerçek: {gerçek:.2f} -> Tahmin: {tahmin:.2f}")


print("\n Model Performansı Değerlendiriliyor...\n")

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test,y_pred)

print("\nModel Performansı Çıktıları:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"r2 Skoru:: {r2:.4f}")



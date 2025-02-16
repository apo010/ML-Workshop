import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

print("\nModel Tahminleri Görselleştiriliyor...\n")
plt.scatter(y_test,y_pred, color='blue')
plt.xlabel("Gerçek Değer")
plt.ylabel("Tahmin Edilen Değerler")
plt.title("Lineer Regresyon: Gerçek vs Tahmin")
plt.grid(True)

plt.savefig("lineer_regresyon_sonucu.png")
print("\nLineer Rekresyon Grafiği 'lineer_regresyon_sonucu.png' olarak kaydedildi.")

print("\n Lojistik Regresyon İçin Veri Seti Oluşturuluyor")
data_classification = {
    "Age": [22,25,47,52,46,56,55,60,62,61],
    "Income": [25000,32000,47000,52000,46000,58000,60000,62000,64000, 63000],
    "Purchased": [0,0,1,1,1,1,1,1,1,1],
}

df_classification = pd.DataFrame(data_classification)

print("\nLojistik Regresyon Veri Seti:")
print(df_classification.head())

print("\nLojistik Regresyon İçin Veri Eğitim ve Test Setine Ayrılıyor...\n")
x_cls = df_classification[["Age","Income"]]
y_cls = df_classification["Purchased"]

X_train_cls, X_test_cls, y_train_cls,y_test_cls = train_test_split(x_cls,y_cls,test_size=0.2,random_state=42)

print((f"Eğitim Seti Boyutu: {X_train_cls.shape}"))
print((f"TEst Seti Boyutu: {X_test_cls.shape}"))

print("\n Lojistik Regresyon Modeli Eğitiliyor...\n")

model_lojistik = LogisticRegression()

model_lojistik.fit(X_train_cls,y_train_cls)

print("\nLojistik Regresyon Modeli Eğitildi!")
print("Katsayılar (Coefficients):")
print(model_lojistik.coef_)
print(f"Intercept (b0): {model_lojistik.intercept_}")

print("\n Lojistik Regresyon Modeli Test Verisi ile Tahmin Yapıyor...\n")
y_pred_lojistik = model_lojistik.predict(X_test_cls)

print("\n Gerçek vs Tahmin Edilen Değerler:")
for gercek, tahmin in zip(y_test_cls, y_pred_lojistik):
    print(f"Gerçek: {gercek} -> Tahmin: {tahmin}")

print("\n Lojistik Regresyon Modeli Performansı Ölçülüyor...\n")
accuracy_lojistic = accuracy_score(y_test_cls,y_pred_lojistik)

print("\n Lojistik Regresyon Modeli Performans Sonuçları:")
print(f"Doğruluk Skoru: {accuracy_lojistic}")

print("\n KNN Modeli Eğitiliyor....\n")

model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn.fit(X_train_cls,y_train_cls)
print("\n KNN Modeli Eğitildi")

print("\n KNN Modeli Test Verisi ile Tahmin Yapıyor...\n")

y_pred_knn = model_knn.predict(X_test_cls)

print("\n Gerçek vs Tahmin Edilen Değerler:")
for gercek, tahmin in zip(y_test_cls,y_pred_knn):
    print(f"Gerçek: {gercek} -> Tahmin: {tahmin}")
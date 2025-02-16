import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

customer_data = pd.read_csv('dataset/customer_data.csv')

print("Veri Setinin İlk 5 Satırı:")
print(customer_data.head())

print("\n Eksi veri Sayısı:")
print(customer_data.isnull().sum())

plt.figure(figsize=(6,4))
sns.histplot(customer_data['age'],bins=30,kde=True)
plt.title("Yaş Dağılımı")
plt.xlabel("Yaş")
plt.ylabel("Frekans")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=customer_data['income'], y=customer_data['speding_score'],hue=customer_data['category'])
plt.title("Gelir vs Harcama Grafiği")
plt.xlabel("Gelir")
plt.ylabel("Harcama Scoru")
plt.show()

plt.figure(figsize=(6,4))
sns.scatterplot(x=customer_data['loyalty_score'], y=customer_data['speding_score'],hue=customer_data['category'])
plt.title("Sadakat Puanı vs Harcama Puanı")
plt.xlabel("Sadakat Skoru")
plt.ylabel("Harcama Scoru")
plt.show()





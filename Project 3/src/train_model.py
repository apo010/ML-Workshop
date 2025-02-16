import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

customer_data = pd.read_csv('dataset/customer_data.csv')

label_encoder = LabelEncoder()
customer_data['category_encoded'] = label_encoder.fit_transform(customer_data['category'])

x = customer_data[['age','income','speding_score','loyalty_score']]
y = customer_data['category_encoded']

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=5, random_state=42)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print(f"\nKarar ağacı model Doğruluğu: {accuracy:.2f}")

print("\n Sınıflama Raporu:")
print(classification_report(y_test,y_pred,target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap='Blues',xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen Değer')
plt.ylabel('Gerçek Değer')
plt.title('Karşılık Matrisi')
plt.show()
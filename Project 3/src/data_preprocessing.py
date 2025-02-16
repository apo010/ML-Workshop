import pandas as pd
import numpy as np

np.random.seed(42)
num_sales = 1000

custumer_data = pd.DataFrame({
    'age': np.random.randint(18,70,num_sales),
    'income': np.random.randint(20000,150000,num_sales),
    'speding_score': np.random.randint(1,100,num_sales),
    'loyalty_score': np.random.randint(1,10,num_sales),
    'category': np.random.choice(['low','Medium','High'],num_sales)
})

custumer_data.to_csv('dataset/customer_data.csv',index=False)

print("\n Veri seti başarıyla oluşturuldu ve dataset/customer_data.csv içine kaydedildi.")
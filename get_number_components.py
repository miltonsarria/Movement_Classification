import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

#cargar datos y organizar la matriz X y el vector de etiquetas y
len_window = 150
f = open('data_time_'+str(len_window)+'.pickle', 'rb')
dic_data = pickle.load(f)
f.close()    

keys = dic_data.keys()
X = []
y = []
i = 0
for key in keys:
    X.append(dic_data[key])
    n_vec = dic_data[key].shape[0]
    y.append(np.ones(n_vec)*i)
    i=i+1
X= np.vstack(X)
y=np.hstack(y).astype(int)
print(f'[INFO] Data loaded: X  {X.shape}, y  {y.shape}')
target_names = list(keys)
###### separar train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=True)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca= PCA()
pca.fit(X_train)
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.plot(np.arange(X_train.shape[1]),0.98*np.ones(X_train.shape[1]))
plt.show()

### usar solo las componentes necesarias
pca= PCA(n_components = 10)
pca.fit(X_train)
Xtrain= pca.transform(X_train)
Xtest=pca.transform(X_test)

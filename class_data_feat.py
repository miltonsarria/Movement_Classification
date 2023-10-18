import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import pickle

#cargar datos y organizar la matriz X y el vector de etiquetas y
len_window = 150
n_components = 40
f = open('data_freq_'+str(len_window)+'.pickle', 'rb')
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

pca= PCA(n_components=n_components)
pca.fit(X_train)


Xtrain= pca.transform(X_train)
Xtest=pca.transform(X_test)
    
#scaler = Normalizer()
#scaler.fit(X_train)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
    

#crear un clasficador
clf = LogisticRegression(solver='lbfgs', max_iter=1000,multi_class='ovr')
#entrenar clasificador
clf.fit(X_train,y_train)
#evaluar
y_pred=clf.predict(X_test) 

acc = np.sum(y_pred == y_test)/y_test.size*100
print(f'[INFO] Accuracy LR = {acc}')
print(classification_report(y_test, y_pred, target_names=target_names))

################
clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test) 

acc = np.sum(y_pred == y_test)/y_test.size*100
print(f'[INFO] Accuracy knn = {acc}')
print(classification_report(y_test, y_pred, target_names=target_names))


clf = RandomForestClassifier(max_depth=10, random_state=0)
clf.fit(X_train,y_train)
#evaluar
y_pred=clf.predict(X_test) 

acc = np.sum(y_pred == y_test)/y_test.size*100
print(f'[INFO] Accuracy RF = {acc}')
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', activation='relu', max_iter=1000, alpha=1e-5, hidden_layer_sizes=(50, 40, 40, 20, 7))
clf.fit(X_train,y_train) 
y_pred=clf.predict(X_test) 
acc = np.sum(y_pred == y_test)/y_test.size*100
print(f'[INFO] Accuracy ANN = {acc}')
print(classification_report(y_test, y_pred, target_names=target_names)) 
#####svm 
C = 1000  # parametro de regularizacion de la svm
gamma=3

scaler = Normalizer()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
    
clf = svm.SVC(kernel='rbf', gamma=gamma, C=C)
clf.fit(X_train,y_train) 
y_pred=clf.predict(X_test) 

acc = np.sum(y_pred == y_test)/y_test.size*100
print(f'[INFO] Accuracy SVM = {acc}')
print(classification_report(y_test, y_pred, target_names=target_names)) 
    



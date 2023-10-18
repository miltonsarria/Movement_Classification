import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn import svm
import pickle

#cargar datos y organizar la matriz X y el vector de etiquetas y
f = open('data.pickle', 'rb')
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

pca= PCA(n_components=50)
pca.fit(X_train)


#####svm 
tuned_parameters =[

                {'kernel': ['linear'],
                 'C': [0.001, 0.10, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]},
                 
                { 'kernel': ['poly'],
                 'gamma': [1e-3, 1e-2, 1, 3,5,10],
                 'C': [0.001, 0.10, 0.1, 1, 10, 100, 1e3, 1e4, 1e5, 1e6]},

                {'kernel': ['rbf'],
                 'gamma': [1e-3, 1e-2, 1, 3,5,10],
                 'C': [0.001, 0.10, 0.1,1, 10, 100, 1e3, 1e4, 1e5, 1e6]},
                ]


scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

svc = svm.SVC()
   
clf = GridSearchCV(svc, param_grid=tuned_parameters, n_jobs=2)
clf= clf.fit(X_train,y_train)
print("Best Hyper Parameters:\n",clf.best_params_)



y_pred=clf.predict(X_test) 

acc = np.sum(y_pred == y_test)/y_test.size*100
print(f'[INFO] Accuracy SVM = {acc}')
print(classification_report(y_test, y_pred, target_names=target_names)) 
    





import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 读取一个路径下所有的.dat文件，默认路径设为..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\pre2")
os.chdir("DATA\\pre2")


data = pd.read_csv(r'20210708pre2.csv')

X = data.iloc[:, 5:]
y = data.iloc[:, data.columns == "bin_type"]

from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y.values.ravel(), test_size=0.2)


svc = SVC()

svc = svc.fit(Xtrain, Ytrain)
score_ = svc.score(Xtest, Ytest)

print("\nscore")
print(score_)

score = cross_val_score(svc, X, y.values.ravel(), cv=10).mean()
print("\nscore mean")
print(score)

"""
print("\n\nTIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")
print("TIME WARNING  TIME WARNING  TIME WARNING  TIME WARNING  \n")

def svc_param(X, y, nfolds):
    parameters = {
        'C': [*np.logspace(-1, 3, 10, base=2)]
        , 'kernel': ('rbf', 'linear', 'poly', 'sigmoid')
        , "gamma": [*np.logspace(-9, 1, 50, base=2)]
    }
    grid_search = GridSearchCV(SVC(), parameters, cv=nfolds)
    grid_search.fit(X, y)
    grid_search.best_params_
    return grid_search

model = svc_param(Xtrain,Ytrain, 4)

print(model.best_params_)
# """

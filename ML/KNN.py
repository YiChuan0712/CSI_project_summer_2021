import numpy as np
import pandas as pd
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 读取一个路径下所有的.dat文件，默认路径设为..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\pre3")
os.chdir("DATA\\pre3")


data = pd.read_csv(r'20210708pre3.csv')

linknum = 5

scores = []
for i in range(10):
    list = range(1, 75)
    randomlist = random.sample(list, 15)
    print(randomlist)
    print()
    data1 = data[data['th'].isin(randomlist)]
    # print(data1)
    # print()
    data2 = data[~data['th'].isin(randomlist)]
    # print(data2)
    # print()


    Xtrain = data1.iloc[:, 5:linknum*30*7+5-1]
    Ytrain = data1.iloc[:, data.columns == "bin_type"]


    Xtest = data2.iloc[:, 5:linknum*30*7+5-1]
    Ytest = data2.iloc[:, data.columns == "bin_type"]


    knn = neighbors.KNeighborsClassifier()

    knn = knn.fit(Xtrain, Ytrain.values.ravel())
    score_ = knn.score(Xtest, Ytest.values.ravel())
    scores.append(score_)
    print(score_)
    print()

    sns.set()
    f, ax = plt.subplots()
    Ytrue = Ytest
    Ypred = knn.predict(Xtest)
    C2 = confusion_matrix(Ytrue, Ypred)
    print(C2)  # 打印出来看看



print("\n10 fold score")
print(np.mean(scores))


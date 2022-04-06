import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from sklearn.utils.multiclass import unique_labels


def plot_Matrix(cm, classes, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='10')  # 设置字体样式、大小

    # 按行进行归一化
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
    str_cm = cm.astype(np.str).tolist()
    for row in str_cm:
        print('\t'.join(row))
    # 占比1%以下的单元格，设为0，防止在最后的颜色中体现出来
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax) # 侧边的颜色条带

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # 标注百分比信息
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig('cm.jpg', dpi=300)
    plt.show()

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='10')  # 设置字体样式、大小
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.ylim(len(cm) - 0.5, -0.5)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# 读取一个路径下所有的.dat文件，默认路径设为..\\DATA\\xxxx
os.chdir(os.path.realpath(__file__) + "\\..")
train_path_list = os.listdir("DATA\\pre3")
os.chdir("DATA\\pre3")


data = pd.read_csv(r'20210708pre3.csv')

linknum = 5

scores = []
for i in range(1):
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


    svc = SVC()

    svc = svc.fit(Xtrain, Ytrain.values.ravel())
    score_ = svc.score(Xtest, Ytest.values.ravel())
    scores.append(score_)
    print(score_)
    print()

    sns.set()
    f, ax = plt.subplots()
    Ytrue = Ytest
    Ypred = svc.predict(Xtest)
    CM = confusion_matrix(Ytrue, Ypred)
    print(CM)

    plot_Matrix(CM, ['00011',
                     '00101',
                     '00110',
                     '01001',
                     '01010',
                     '01100',
                     '10001',
                     '10010',
                     '10100',
                     '11000'], title=None)

    plot_confusion_matrix(CM, ['00011',
                     '00101',
                     '00110',
                     '01001',
                     '01010',
                     '01100',
                     '10001',
                     '10010',
                     '10100',
                     '11000'], title=None)

# print("\n10 fold score")
# print(np.mean(scores))



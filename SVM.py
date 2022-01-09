from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

svm = SVC()


def SV(filename):
    df = pd.read_csv(filename)

    Y = df['RainTomorrow']
    # print(df['Default'])
    X = df.drop(columns=['RainTomorrow'])
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

    svm.fit(x_train, y_train)
    # print(svm.score(x_test,y_test))
    y_pred = svm.predict(x_test)
    print(confusion_matrix(y_test, y_pred))
    tp, fp, fn, tn = confusion_matrix(y_test, y_pred).ravel()
    print((tn + tp) / (tp + fp + fn + tn))


filename = ['smote_oversample.csv','random_oversample.csv']

for i in filename:
    print(i,SV(i))
'''
[[21650   364]
 [ 6250  7997]]
0.8176001764981661
smote_oversample.csv None
[[21642   382]
 [ 6138  8099]]
0.8201924933123742
random_oversample.csv None
'''
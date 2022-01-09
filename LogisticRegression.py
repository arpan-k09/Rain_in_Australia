from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd

lr = LogisticRegression(max_iter=1000000000)

def LR(filename):
    df = pd.read_csv(filename)
    Y = df['RainTomorrow']
    X = df.drop(columns=['RainTomorrow'])
    x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=42,train_size=0.25)
    lr.fit(x_train,y_train)
    y_pred = lr.predict(x_test)
    tp,tn,fn,fp = confusion_matrix(y_test,y_pred).ravel()
    print('Accuracy = ', ((tp+tn)/(tp+tn+fp+fn))*100)
    print(confusion_matrix(y_test,y_pred))

filename = ['smote_oversample.csv','random_oversample.csv']

for i in filename:
    print(i,LR(i))
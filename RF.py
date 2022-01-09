from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np

rf = RandomForestClassifier()

def RF(filename):
    grid = GridSearchCV(param_grid={'n_estimators':np.arange(990,1010,10),'max_features':np.arange(15,16,1)},estimator=rf,cv=10)
    df = pd.read_csv(filename)
    Y = df['RainTomorrow']
    X = df.drop(columns=['RainTomorrow'])
    x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=42,train_size=0.25)
    grid.fit(x_train,y_train)
    print(grid.score(x_test,y_test))
    # rf.fit(x_train,y_train)
    # y_pred = rf.predict(x_test)
    # tp,tn,fn,fp = confusion_matrix(y_test,y_pred).ravel()
    # print('Accuracy = ', ((tp+tn)/(tp+tn+fp+fn))*100)
    # print(confusion_matrix(y_test,y_pred))

filename = ['smote_oversample.csv','random_oversample.csv']

for i in filename:
    print(i,RF(i))



from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df = pd.read_csv('random_oversample.csv')
Y = df['RainTomorrow']
df.drop(columns=['RainTomorrow'],inplace=True)

scalar = StandardScaler()
df = scalar.fit_transform(df)

pca = PCA(0.95)
x_pca = pca.fit_transform(df)
print(x_pca.shape)

x_train, x_test, y_train, y_test = train_test_split(x_pca,Y,test_size=0.25,random_state=42)
# rf = RandomForestClassifier(n_estimators=1000,max_features=15)
lr = LogisticRegression(max_iter=10000000)
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))

svm = SVC()
svm.fit(x_train,y_train)
print(svm.score(x_test,y_test))


rf = RandomForestClassifier(n_estimators=1000,max_features=15)
rf.fit(x_train,y_train)
print(rf.score(x_test,y_test))

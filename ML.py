from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import plotly.graph_objects as go



# ======================================================================================================================
# GINI importance
# ======================================================================================================================

df = pd.read_csv('clean_weatherAUS.csv')
y = df['RainTomorrow']
x = df.drop(columns=['RainTomorrow'])

print(df.drop(columns=['RainTomorrow']).columns)


dt = DecisionTreeClassifier()
dt.fit(x,y)

imps = dt.feature_importances_


cols=list(x)
fig = go.Figure(data=[
    go.Bar(name='Own asset', x=cols, y=imps),
])
# Change the bar mode
fig.update_layout(title="GINI Importance score of our features")
fig.show()
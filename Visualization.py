from dabl import plot
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('clean_weatherAUS.csv')

plot(df,'RainTomorrow')
plt.show()
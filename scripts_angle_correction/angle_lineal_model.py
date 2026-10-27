import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from scipy import stats

#Datos Lago	VV	Angulo	Orbita	Zona
dt = pd.read_excel("../data/angle_correction.xlsx")
X_train, X_test, y_train, y_test = train_test_split(dt["Angulo"].to_numpy(), dt["VV"].to_numpy(), test_size=0.2, random_state=42)
print( len(y_train), len(y_test) )
#Regresion lineal
#reg = linear_model.LinearRegression()
#reg.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Predict on the test set  
#y_pred = reg.predict(X_test.reshape(-1,1))  

# Evaluate the model  
#r2 = r2_score(y_test, y_pred)  
#print(f"R-squared: {r2}")

#rmse = root_mean_squared_error(y_test, y_pred)
#print(f"RMSE: {rmse}")

#print(f"Model coef: {reg.coef_}")
#print(f"Model intercept: {reg.intercept_}")


slope, intercept, r, p, se = stats.linregress(X_train, y_train)
y_pred = slope*X_test + intercept
print(f"p: {p}")
#print(slope, intercept, r, p, se)

r2 = r2_score(y_test, y_pred)  
print(f"R-squared: {r2}")

rmse = root_mean_squared_error(y_test, y_pred)
print(f"RMSE: {rmse}")

print(f"Model coef: {slope}")
print(f"Model intercept: {intercept}")
print(f"Median: {dt["Angulo"].median()}")

fig, ax = plt.subplots(figsize=(14,8))
symbols = {"Ascending": "o", "Descending": "^"}
colors = {"Castillo": "red", "Escondida": "blue", "Las Perdices": "green", "Lynch": "orange", "Parrillar": "purple", "TdF01": "brown", "Torres": "pink"}

for clase, grupo in dt.groupby('Orbita'):
    for subclase, subgrupo in grupo.groupby('Lago'):
        if clase == "Ascending":
            plt.scatter(subgrupo['Angulo'], subgrupo['VV'], marker=symbols[clase], color=colors[subclase], label=f"Lake {subclase}")
        else:
            plt.scatter(subgrupo['Angulo'], subgrupo['VV'], marker=symbols[clase], color=colors[subclase])

x_pred = np.linspace(29.5, 44.5, 11)
plt.plot( x_pred, slope*x_pred + intercept, linestyle="--", color="black", label=f"Lineal Model: slope -0.211 dB/°" )

# Añadir etiquetas y título
plt.title('Angle effect on Backscattering')
plt.ylabel('Backscattering (dB)')
plt.xlabel('Angle (°)')

# Mostrar cuadrícula y gráfico
plt.grid(True)
plt.legend(loc="upper left", ncols=2)

plt.tight_layout()
plt.savefig('AngleBack.png', dpi=120, bbox_inches='tight')
#plt.show()

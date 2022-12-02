# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import anderson, probplot, multivariate_normal, chi2
from pingouin import multivariate_normality
#%matplotlib inline

# %%
url = "https://raw.githubusercontent.com/crisb-7/Mercurio/main/mercurio.csv"

columnNames = ["ID", "Lago", "Alcalinidad", "pH", "Calcio", "Clorofila", "Mercurio", 
               "N_Peces", "MinMercurio", "MaxMercurio", "Estimacion", "Edad"]

df = pd.read_csv(url, names = columnNames, header = 0)

# %%
df.head()

# %%
df = df.drop(columns = "ID")
columnNames.remove("ID")

# %%
df.describe()

# %% [markdown]
# # Análisis de Normalidad

# %%
numericVars =list(df.describe().columns)
numericVars

# %% [markdown]
# Prueba de normalidad univariada de cada variable

# %%
# Anderson() SciPy
# If the returned statistic is larger than these critical values then for the corresponding significance level, 
# the null hypothesis that the data come from the chosen distribution can be rejected. 
# The returned statistic is referred to as ‘A2’ in the references.

# %%
aN = [] # Anderson Normal - Boolean
aS = [] # Anderson Statistic
aC = [] # Anderson Critical Values
aL = [] # Anderson Significance Levels

for var in numericVars:
  
  ds = df[var]
  nt = anderson(ds, dist="norm")

  aS.append(nt.statistic)
  aC.append(nt.critical_values)
  aL.append(nt.significance_level)
  
  if nt.statistic < nt.critical_values[2]:
    aN.append(True)
  else:
    aN.append(False)

univar_normal = pd.DataFrame({"Variable":numericVars, "Normal":aN, "Statistic":aS, "Crit":aC, "Significance":aL})
univar_normal.head(10)

# %%
print("Hay", univar_normal.Normal.sum(), "variable(s) Anderson normale(s)")

# %% [markdown]
# Variable normal de acuerdo con la prueba de Anderson-Darling

# %%
univar_normal.loc[aN]

# %%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 15
# plt.rcParams["text.usetex"] = True
plt.rcParams["axes.titlesize"] = 17

# plt.rcParams.update({"figure.dpi": 180, "font.family": "serif", "text.usetex": True, "font.size": 14, "axes.titlesize": 16,})

fig, axes = plt.subplots(1,2, figsize=(12,4))

probplot(df.pH, plot = axes[0])
axes[0].get_lines()[0].set_marker('o')
axes[0].get_lines()[0].set_linestyle('--')
axes[0].get_lines()[0].set_color('midnightblue')
axes[0].get_lines()[1].set_color('darkorange')
axes[0].set_title("Gráfico Q-Q - pH")
axes[0].set_xlabel("Cuantiles teóricos")
axes[0].set_ylabel("Valores observados")
axes[0].grid()

sns.histplot(data = df, x = "pH", kde = True, color="indigo", ax = axes[1])
axes[1].set_title("Distribución pH")
axes[1].set_xlabel("Mercurio [mg Hg/kg]")
axes[1].set_ylabel("Frecuencia")
plt.show()

# %%
df.MaxMercurio.skew()

# %%
df.MaxMercurio.kurtosis()

# %%
fig, axes = plt.subplots(1,2, figsize=(12,4))
probplot(df.MaxMercurio, plot = axes[0])
axes[0].get_lines()[0].set_marker('p')
axes[0].get_lines()[0].set_markeredgecolor("k")
# axes[0].get_lines()[0].set_linestyle('-')
axes[0].get_lines()[0].set_color('midnightblue')
axes[0].get_lines()[1].set_color('darkorange')
axes[0].set_title("Gráfico Q-Q: Mercurio Máximo")
axes[0].set_xlabel("Cuantiles teóricos")
axes[0].set_ylabel("Valores observados")
axes[0].grid()

sns.histplot(data = df, x = "MaxMercurio", kde = True, color="slateblue", ax = axes[1])
axes[1].set_title("Distribución Mercurio Máximo")
axes[1].set_xlabel("Mercurio [mg Hg/kg]")
axes[1].set_ylabel("Frecuencia")
plt.show()

# %%
multivariate_normality(df[["pH", "MaxMercurio"]], alpha=0.05)

# %% [markdown]
# Prueba de normalidad multivariada de todas las combinaciones

# %%
numericVars

# %%
from itertools import combinations

nn = len(numericVars)
c = 0

for i in range(2, nn):
  comb = combinations(numericVars, i)
  for j in comb:
    cols = list(j)
    # print(cols)
    data = df[cols]
    lol = multivariate_normality(data, alpha=0.05)
    if lol.normal:
      c += 1
      print(list(data.columns))

print("Hay", c, "normal(es) multivariada(s)")

# %%
X1 = (df.pH - df.pH.mean())/df.pH.std()
X2 = (df.MaxMercurio - df.MaxMercurio.mean())/df.MaxMercurio.std()

plt.scatter(X1, X2, color = "royalblue", edgecolor="k")

fig = plt.gcf()
fig.set_size_inches(7, 4)

plt.title("Normal Multivariada")
plt.ylabel(r"Max Mercurio [$\sigma$ de la media]")
plt.xlabel(r"pH [$\sigma$ de la media]")
plt.grid()
plt.axis("equal")
plt.show()

# %%
S = np.cov(X1, X2)
S

# %%
rho = -S[0,1]/(np.sqrt(S[0,0])*np.sqrt(S[1,1]))
rho

# %%
lol = np.ones(len(df))*(-0.05)

# %%
x = np.linspace(-3,3, 200)
y = np.linspace(-3,3, 200)


rv = np.random.multivariate_normal([X1.mean(), X2.mean()], S, size = 250)

X, Y = np.meshgrid(x,y)
f = (1/(2*np.pi*np.sqrt(1-rho**2)))*np.exp( -( 1/(2*(1-rho**2)) ) * (X**2 + Y**2) - 2*rho*X*Y )

# %%
# from mpl_toolkits.mplot3d import axes3d

fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(projection="3d")
ax.plot_surface(X, Y, f, cmap="coolwarm")

# ax.scatter(X1, X2, zs=-0.2, color="k")

ax.set_title("Normal Multivariada: pH y Mercurio máximo")
ax.set_xlabel(r"pH [$\sigma$ de la media]")
ax.set_ylabel(r"Mercurio máximo [$\sigma$ de la media]")
ax.set_zlabel(r"$f(x_1, x_2)$")
plt.show()

# %%
plt.contour(X, Y, f, 8, cmap='coolwarm', linestyles="solid")

fig = plt.gcf()
fig.set_size_inches(9, 5)

ax.set_figure
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel(r"$f(x_1, x_2)$", rotation=270)

plt.scatter(X1, X2, color = "k")

plt.title("Curvas de Nivel - Normal Multivariada")
ax.set_xlabel(r"pH [$\sigma$ de la media]")
ax.set_ylabel(r"Mercurio máximo [$\sigma$ de la media]")
plt.ylabel(r"Max Mercurio [$\sigma$ de la media]")
plt.xlabel(r"pH [$\sigma$ de la media]")
plt.grid()
plt.tight_layout()

plt.show()

# %%
Md = pd.DataFrame(np.array([X1, X2]).T)
lam, v = np.linalg.eig(S)

# %%
# Chi_2^2 (alpha 0.05) = 5.99 - 95% confidence

def get_confidence_interval(alpha):
  gl = 2
  significance = 1 - alpha
  ci = chi2.ppf(significance, gl)
  print(ci)
  theta = np.linspace(0, 2*np.pi, len(X1));
  ab = np.sqrt(ci*lam[None,:])
  return (ab * v) @ np.array([np.sin(theta), np.cos(theta)])

ellipsis68 = get_confidence_interval(0.32)
ellipsis95 = get_confidence_interval(0.05)

# %%
# plt.scatter(Md[0], Md[1], color = "blue")
plt.scatter(X1, X2, color = "k", edgecolor="k")

fig = plt.gcf()
fig.set_size_inches(8, 5)

plt.plot(ellipsis68[0,:], ellipsis68[1,:], color = "firebrick", label=r"$\sigma$ (68%)")
plt.plot(ellipsis95[0,:], ellipsis95[1,:], color = "navy", label=r"$2\sigma$ (95%)")

plt.title("Normal Multivariada: Intervalos de Confianza")
plt.xlabel(r"pH [$\sigma$ de la media]")
plt.ylabel(r"Max Mercurio [$\sigma$ de la media]")
plt.xlim([-3, 3])
plt.ylim([-3,3])
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Análisis de Componentes Principales

# %%
cols = df.corr(numeric_only=True).columns
dfs = df[cols]

# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# %%
# scaler = MinMaxScaler()
scaler = StandardScaler()
dfs = scaler.fit_transform(dfs)
dfs = pd.DataFrame(dfs, columns = cols)

# %%
dfs.head()

# %%
R = dfs.corr()
lam, v = np.linalg.eig(R)

# %%
# PCA on similar correlation variables and compare them???
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(R.round(decimals=2), cmap="RdBu", annot=True, ax = ax)
plt.title("Matriz de correlación")
plt.show()

# %% [markdown]
# Eigenvalores

# %%
pd.DataFrame(lam).head()

# %% [markdown]
# Eigenvectores

# %%
pd.DataFrame(v).head()

# %% [markdown]
# Varianza explicada

# %%
var_expl = (lam/lam.sum())*100
pd.DataFrame(var_expl).round(decimals=4).head()

# %%
np.cumsum(var_expl)

# %%
df.head()

# %%
coeffs = pd.DataFrame(v, columns = cols).loc[0:4, :]
coeffs = coeffs.round(decimals=2)#.abs()
# coeffs = (coeffs > 0.69/2).astype(int)

if coeffs.min().min() > 0:
  colormap = "Purples"
else:
  colormap = "coolwarm"

# fig, ax = plt.subplots(figsize=(10,6))
fig, ax = plt.subplots(figsize=(10,3))
sns.heatmap(coeffs.loc[0:3, :], cmap=colormap, annot=True, ax = ax)
plt.title("Coeficientes de Descomposición")
# plt.title("Eigenvectores de S")
plt.show()

# %%
df.head()

# %%
# n = 5 #len(df)
# pcs = 3
# pc_evol = np.zeros((pcs, n, len(dfs.columns)))
# for pc in range(pcs):
#   for datapoint in range(n):
#     PC = coeffs.loc[pc, :]
#     data = df[cols].loc[datapoint]
#     pc_evol[pc, datapoint,:] = np.cumsum(PC*data)

# plt.plot(pc_evol[0].T, color="r", marker="o", linestyle="--")
# plt.plot(pc_evol[1].T, color="b", marker="*", linestyle="-.")
# plt.plot(pc_evol[2].T, color="g", marker="p", linestyle="--")
# plt.show()

# %% [markdown]
# PCA1 - Variacion del nivel de alcalinidad, ph y calcio en funcion del nivel de mercurio y nivel minimo de mercurio en el lago regulado por el nivel maximo de mercurio

# %%
M = np.array(dfs[cols])

# %%
PCA = (v @ M.T)

pca_columns = ["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8", "PCA9", "PCA10"]

PCA = pd.DataFrame(PCA.T, columns = pca_columns)
PCA.head()

# %%
dummy = PCA.loc[:,PCA.columns[0:4]]
dummy["Mercurio"] = df.Estimacion

sns.pairplot(data=dummy, hue="Mercurio")
plt.show()

# %%
dummy = PCA.loc[:,PCA.columns[0:4]]
dummy["Mercurio"] = df.Estimacion
colormap = "mako"
fig, ax = plt.subplots(figsize=(8,3))
sns.heatmap(dummy.corr(), cmap=colormap, annot=True, ax = ax)
plt.title("Correlación PCs - Mercurio")
plt.show()

# %%
plt.scatter(PCA.PCA1, PCA.PCA2, c=df.Mercurio, cmap="inferno", edgecolor="k")

# plt.colorbar()
ax.set_figure
cbar = plt.colorbar()
cbar.ax.get_yaxis().labelpad = 20
cbar.ax.set_ylabel("Mercurio [mg Hg/kg]", rotation=270)

plt.title("Componentes Principales")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid()
plt.axis("equal")
plt.xlim([-3, 3])
plt.ylim([-3, 3])

plt.show()

# %% [markdown]
# 



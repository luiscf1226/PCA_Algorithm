import matplotlib.pyplot as plt
import numpy as np



correlation_matrix = np.array([
    [0.9999999999999998, 0.8540787779257426, 0.3845742420832207, 0.20719425251263762, -0.7871626873142726],
    [0.8540787779257426, 1, -0.02005218433210575, -0.02153942000025033, -0.687720557120766],
    [0.3845742420832207, -0.02005218433210575, 1, 0.8209161926045639, -0.36554341640973137],
    [0.20719425251263762, -0.02153942000025033, 0.8209161926045639, 1, -0.5080013180428953],
    [-0.7871626873142726, -0.687720557120766, -0.36554341640973137, -0.5080013180428953, 0.9999999999999998]
])


eigenvalues = np.array([2.893249673417947, 1.6286504249773157, 0.3465960485145294, 0.12261245959725231, 0.008891393492955226])


eigenvectors = np.array([
    [0.5266439738396538, -0.27049630257018364, 0.4382007055494512, -0.26121779314245686, 0.6238776173398424],
    [0.4249362197779643, -0.5080722146236653, 0.04049490866260933, 0.6736272448787415, -0.3253895071329122],
    [0.35914703736054543, 0.562081593130318, 0.5622758253310839, -0.0700864697353343, -0.48374732187703134],
    [0.3526974703476121, 0.5864898518056105, -0.394180315545198, 0.44664495139125737, 0.4204334835433635],
    [-0.5373018089535246, 0.09374599413511181, 0.5786260269926566, 0.5230561859490815, 0.30679407054660235]
])


variables = ["Matematicas","Ciencias","Español","Historia","EdFisica"]


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)


circle = plt.Circle((0, 0), 1, color='blue', fill=False)
ax.add_artist(circle)


for i, var in enumerate(variables):
    ax.arrow(0, 0, eigenvectors[i][0], eigenvectors[i][1], head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(eigenvectors[i][0]*1.15, eigenvectors[i][1]*1.15, var, color='black', ha='center', va='center')


plt.title("Gráfico de Círculo de Correlación")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
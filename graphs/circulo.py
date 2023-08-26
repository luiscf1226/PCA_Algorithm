import matplotlib.pyplot as plt
import numpy as np

# Matriz de correlación
correlation_matrix = [
    [0.9999999999999998, 0.8540787779257426, 0.3845742420832207, 0.20719425251263762, -0.7871626873142726],
    [0.8540787779257426, 1, -0.02005218433210575, -0.02153942000025033, -0.687720557120766],
    [0.3845742420832207, -0.02005218433210575, 1, 0.8209161926045639, -0.36554341640973137],
    [0.20719425251263762, -0.02153942000025033, 0.8209161926045639, 1, -0.5080013180428953],
    [-0.7871626873142726, -0.687720557120766, -0.36554341640973137, -0.5080013180428953, 0.9999999999999998]
]

# Eigenvalues
eigenvalues = [2.893249673417947, 1.6286504249773157, 0.3465960485145294, 0.12261245959725234, 0.008891393492955223]

# Eigenvectors
eigenvectors = [
    [0.5266439738876594, 0.42493621986813285, 0.35914703726079167, 0.3526974702435266, -0.5373018089701619],
    [-0.2704963023561824, -0.5080722144852626, 0.5620815933050926, 0.5864898518736607, 0.09374599402904804],
    [0.43820070572067726, 0.04049490884114658, 0.5622758250505991, -0.3941803159017376, 0.5786260268801622],
    [-0.2612177930062507, 0.6736272449011483, -0.07008646952758624, 0.44664495124355114, 0.523056186142212],
    [0.6238776172473827, -0.3253895068945355, -0.4837473219018625, 0.42043348370139366, 0.30679407073172926]
]

# Variables
variables = ["Ciencias","Matematicas","Historia","Español","EdFisica"]

# Crear el gráfico
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.axhline(0, color='grey', linewidth=0.5)
ax.axvline(0, color='grey', linewidth=0.5)

# Dibuja el círculo de correlación
circle = plt.Circle((0, 0), 1, color='blue', fill=False)
ax.add_artist(circle)

# Trazar las variables en el gráfico
for i, var in enumerate(variables):
    ax.arrow(0, 0, eigenvectors[i][0], eigenvectors[i][1], head_width=0.05, head_length=0.05, fc='red', ec='red')
    ax.text(eigenvectors[i][0]*1.15, eigenvectors[i][1]*1.15, var, color='black', ha='center', va='center')

# Mostrar el gráfico
plt.title("Gráfico de Círculo de Correlación")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.show()
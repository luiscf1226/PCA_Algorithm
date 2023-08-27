import numpy as np
import matplotlib.pyplot as plt

# Matriz de componentes principales
componentes_principales = np.array([
    [0.30648411837978234, 1.6815643912948854],
    [0.6312923526679473, -1.5546093574950623],
    [0.9510996380386848, -0.4892288307906545],
    [-3.0093133659463533, -0.2492968983127778],
    [-0.4637808775420726, 1.2953341665009073],
    [1.6209518021909708, -0.9692701435798634],
    [0.06411749226446109, 1.3872941364981297],
    [1.9086133922189699, -1.2103914039429928],
    [-2.8859233324717013, -1.1904843535198324],
    [0.8764587801993119, 1.2990882933472554]
])

# Nombres de los estudiantes
nombres_estudiantes = ["Lucia", "Pedro", "Ines", "Luis", "Andres", "Ana", "Carlos", "Jose", "Sonia", "Maria"]

# Coordenadas x e y
x = componentes_principales[:, 0]
y = componentes_principales[:, 1]

# Crear el gráfico
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(x, y, color='blue')

# Etiquetar cada punto con el nombre del estudiante
for i, nombre in enumerate(nombres_estudiantes):
    ax.text(x[i], y[i], nombre, fontsize=9)

# Configuración del gráfico
ax.set_title("Gráfico en el Plano Principal")
ax.set_xlabel("Componente Principal 1")
ax.set_ylabel("Componente Principal 2")
ax.grid(True)

# Mostrar el gráfico
plt.show()

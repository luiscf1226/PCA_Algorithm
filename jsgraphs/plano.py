import numpy as np
import matplotlib.pyplot as plt

# New principal component matrix
componentes_principales = np.array([
  [0.3230626266016885, 1.7725245031418886],
  [0.6654405682498195, -1.6387021469393983],
  [1.002547046079509, -0.5156924670549489],
  [-3.172094809812786, -0.2627820051364756],
  [-0.4888679696638482, 1.3654020988140012],
  [1.70863322427405, -1.021700440045488],
  [0.06758577087845127, 1.4623364186094379],
  [2.0118551642638427, -1.2758645650079778],
  [-3.0420302941886086, -1.2548806928818648],
  [0.9238686733178811, 1.3693592965008263]
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

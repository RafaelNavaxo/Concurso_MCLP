import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_manhattan_coverage(j, coords_km, coverage, R):
    #Grafica el candidato j, su radio Manhattan (rombo) y cuáles puntos cubre.
    xj, yj = coords_km[j]
    # Puntos cubiertos y no cubiertos
    covered = coverage[j]
    not_covered = np.setdiff1d(np.arange(len(coords_km)), covered)
    # --- preparar figura ---
    plt.figure(figsize=(9, 9))
    # Puntos no cubiertos (gris)
    plt.scatter(coords_km[not_covered, 0], coords_km[not_covered, 1],
                s=10, color="lightgray", alpha=0.5, label="No cubiertos")
    # Puntos cubiertos (verde)
    plt.scatter(coords_km[covered, 0], coords_km[covered, 1],
                s=15, color="green", alpha=0.8, label="Cubiertos")
    # Candidato j (rojo)
    plt.scatter([xj], [yj], s=120, color="red", label=f"Instancia j={j}")
    # --- Dibujar rombo Manhattan ---
    rombo_x = [xj, xj + R, xj, xj - R, xj]
    rombo_y = [yj + R, yj, yj - R, yj, yj]

    plt.plot(rombo_x, rombo_y, color="black", linewidth=2, label="Radio Manhattan")

    # --- Estética ---
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.title(f"Cobertura Manhattan del candidato j={j} (R={R} km)")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    plt.show()

def plot_puntos_lat_lon(ptos):
    plt.figure(figsize=(8, 8))
    plt.scatter(ptos["LONGITUD"], ptos["LATITUD"], s=5, color="blue", alpha=0.6)
    plt.title("Distribución de puntos en Aguascalientes")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.grid(True)
    plt.show()

def plot_puntos_km(ptos_km):
    plt.figure(figsize=(8, 8))
    plt.scatter(x_km, y_km, s=8, color="blue", alpha=0.6)
    plt.title("Puntos en coordenadas planas (km) – Aguascalientes")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True)
    plt.show()

df = pd.read_csv('Casos/instancia_2019_3ambulancias_0.5km.csv')

print(df.head())
print(df.columns)

if "id" not in df.columns:
    df["id"] = df.index

puntos = df[["id", "LATITUD", "LONGITUD"]].copy()

print("Numero de puntos de demanda", len(puntos))






coords = puntos[["LONGITUD", "LATITUD"]].to_numpy()
N = len(coords)
print("Numero de puntos de demanda", N)

#-----Conversion de lat,lon a km
R_tierra = 6371 #Radio de la tierra en KM
latitud_prom = np.radians(puntos["LATITUD"].mean())
latitud_rad = np.radians(puntos["LATITUD"])
longitud_rad = np.radians(puntos["LONGITUD"])
latitud_0 = latitud_rad.iloc[0]
longitud_0 = longitud_rad.iloc[0]
x_km = R_tierra*(longitud_rad - longitud_0) * np.cos(latitud_prom)
y_km = R_tierra*(latitud_rad-latitud_0)
coords_km = np.stack([x_km, y_km]).T
print("Coordenadas convertidas a km:", coords_km.shape)

#-----Construccion de matriz de cobertura Manhattan
Cobertura = []
R = 0.5 #radio en Km
N = len(coords_km)
for j in range(N):
    x_j, y_j = coords_km[j]
    M_dist = np.abs(coords_km[:,0] - x_j) + np.abs(coords_km[:,1] - y_j)

    idx_Cobertura = np.where(M_dist <= R)[0]
    Cobertura.append(idx_Cobertura)
"""
for j in range(5):
    print(f"Candidato {j} cubre {len(Cobertura[j])} puntos")
mejor_j = np.argmax([len(c) for c in Cobertura])
print("Candidato que más cubre:", mejor_j)
print("Cobertura:", len(Cobertura[mejor_j]))
plot_manhattan_coverage(len(Cobertura[mejor_j]), coords_km, Cobertura, R)
"""






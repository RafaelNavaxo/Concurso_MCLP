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

def conv_lat_lon2km(pts):
    # -----Conversion de lat,lon a km
    R_tierra = 6371  # Radio de la tierra en KM
    latitud_prom = np.radians(pts["LATITUD"].mean())
    latitud_rad = np.radians(pts["LATITUD"])
    longitud_rad = np.radians(pts["LONGITUD"])
    latitud_0 = latitud_rad.iloc[0]
    longitud_0 = longitud_rad.iloc[0]
    x_km = R_tierra * (longitud_rad - longitud_0) * np.cos(latitud_prom)
    y_km = R_tierra * (latitud_rad - latitud_0)
    coords_km = np.stack([x_km, y_km]).T

    return coords_km

def plot_puntos_km(ptos_km):
    plt.figure(figsize=(8, 8))
    plt.scatter(ptos_km[:,0], ptos_km[:,1], s=8, color="blue", alpha=0.6)
    plt.title("Puntos en coordenadas planas (km) – Aguascalientes")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True)
    plt.show()

def greedy_search_cover(cov,p,n):
    """
    Greedy search para encontrar la mejor forma de cubrir las
    N demandas con p instalaciones
    """
    noCubiertos = set(range(n)) #Set de ptos no cubiertos
    seleccionados = []
    for _ in range(p):
        mejor_j = None
        mejor_ganancia = -1
        for j in range(len(cov)):
            ganancia = len(noCubiertos.intersection(cov[j]))
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_j = j
        #Agregar el mejor candidato
        seleccionados.append(mejor_j)
        noCubiertos -= set(cov[mejor_j])
        print(f"Elegido j={mejor_j}, aporta {mejor_ganancia} nuevos puntos.")
    cobertura_total = n-len(noCubiertos)
    return seleccionados, cobertura_total

def plot_cobertura(cords,chosen):
    plt.figure(figsize=(8, 8))
    plt.scatter(cords[:, 0], cords[:, 1], s=5, color="lightgray")
    for j in chosen:
        xj, yj = cords[j]
        plt.scatter([xj], [yj], s=120, color="red")
    plt.title("Instalaciones seleccionadas por Greedy")
    plt.xlabel("X (km)")
    plt.ylabel("Y (km)")
    plt.grid(True)
    plt.axis("equal")
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

coords_km = conv_lat_lon2km(puntos)
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
"""
mejor_j = np.argmax([len(c) for c in Cobertura])
print("Candidato que más cubre:", mejor_j)
print("Cobertura:", len(Cobertura[mejor_j]))
#plot_manhattan_coverage(len(Cobertura[mejor_j]), coords_km, Cobertura, R)

p = 3  # por ejemplo
chosen, covered = greedy_search_cover(Cobertura, p, N)

print("Instalaciones elegidas:", chosen)
print("Cobertura total:", covered, "puntos")
print("Porcentaje:", covered / N * 100, "%")

print("ptos instalaciones en lat/lon", )
for b in chosen:
    Lat = df.iloc[b]['LATITUD']
    Lon = df.iloc[b]['LONGITUD']
    #print("id: "+str(b)+"- "+str(pto))
    print(str(Lat) + ", " + str(Lon))
plot_cobertura(coords_km, chosen)




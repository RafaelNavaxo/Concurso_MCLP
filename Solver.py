import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

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
    latitud_0 = np.radians(pts["LATITUD"].mean())
    longitud_0 = np.radians(pts["LONGITUD"].mean())
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

def haversine_vec(punto_ref, lista_puntos):
    R = 6371.0

    # Convertir
    lat1 = np.radians(punto_ref[0])
    lon1 = np.radians(punto_ref[1])
    lat2 = np.radians(lista_puntos[:, 0])
    lon2 = np.radians(lista_puntos[:, 1])
    # Diferencias
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    # Fórmula Haversine
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c
    return distance

def greedy_search_cover(cob,cords_km,p,n,dist_min):
    """
    Greedy search para encontrar la mejor forma de cubrir las
    N demandas con p instalaciones
    """
    noCubiertos = set(range(n)) #Set de ptos no cubiertos
    seleccionados = []
    for _ in range(p):
        mejor_j = None
        mejor_ganancia = -1
        for j in range(len(cob)):
            if j in seleccionados:
                continue
            valido = True
            xj, yj = cords_km[j]
            for c in seleccionados:
                xc, yc = cords_km[c]
                distancia = abs(xj-xc) + abs(yj-yc)
                if distancia < dist_min:
                    valido = False
                    break
            if not valido:
                continue

            ganancia = len(noCubiertos.intersection(cob[j]))
            if ganancia > mejor_ganancia:
                mejor_ganancia = ganancia
                mejor_j = j
        if mejor_j is None:
            print("No cubiertos encontrados con distancia minima")
            break
        #Agregar el mejor candidato
        seleccionados.append(mejor_j)
        noCubiertos -= set(cob[mejor_j])
        #print(f"Elegido j={mejor_j}, aporta {mejor_ganancia} nuevos puntos.")
    cobertura_total = n-len(noCubiertos)
    return seleccionados, cobertura_total

def calc_cobertura_total(selec,cobt):
    #Calcular cobertura de los seleccionados
    cubiertos = set()
    for j in selec:
        cubiertos |= set(cobt[j])
    return len(cubiertos),cubiertos

def local_optimization(seleccionados, cobertura,N_demanda):
    """
    Optimización local con un cambio a la vez, para mejorar lo que greedy_search_cover
    seleccionó
        seleccionados: viene de la solución del greedy
        cobertura: lista de puntos cubiertos por candidato j
        N_demanda: numéro total de puntos a cubrir
    """
    seleccionados = seleccionados.copy() #Evitar modificar el original
    p = len(seleccionados)
    mejor_cobertura, mejor_set = calc_cobertura_total(seleccionados,cobertura)
    mejora = True
    #print(f"Cobertura inicial (greedy): {mejor_cobertura}")
    while mejora:
        mejora = False
        for j_in in seleccionados.copy():
            for j_out in range(len(cobertura)):
                if j_out in seleccionados:
                    continue
                candidato_nuevo = seleccionados.copy()
                candidato_nuevo.remove(j_in)
                candidato_nuevo.append(j_out)

                cobt, cobt_set = calc_cobertura_total(candidato_nuevo,cobertura)

                #Evaluar mejora
                if cobt > mejor_cobertura:
                    #print("Se mejoro el set")
                    #print(f"mejora encontrada: +{cobt-mejor_cobertura} puntos")
                    #print(f"Se reemplazó {j_in} por {j_out}")
                    #Hacer el cambio
                    seleccionados = candidato_nuevo
                    mejor_cobertura = cobt
                    mejor_set = cobt_set
                    mejora = True
                    break
                if mejora:
                    break
    #print(f"Cobertura final tras búsqueda local: {mejor_cobertura}")
    return seleccionados, mejor_cobertura, mejor_set

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

def vecino_random(solucion, cant_candidatos):
    """
    Como si fuera optimizaciion local, solo que aqui seleccionamos una entrada
    y salida aleatoria
    Con la finalidad de que no se quede atascado en un maximo local
    """
    j_in = random.choice(solucion) #candidato random
    in_set = set(solucion)         #generamos el complemento de la solucion
    posible_out = [j for j in range(cant_candidatos) if j not in in_set]

    j_out = random.choice(posible_out)
    #Remplazamos
    vecino = solucion.copy()
    idx = vecino.index(j_in)
    vecino[idx] = j_out

    return vecino, j_in, j_out

def simulated_annealing(sol_inicial, cobertura, N_demanda, T0=1.0, a=0.995, iter=150, min_temp=1e-3):
    """
    Parametros:
            T0 - temperatura inicial
             a - alpha, constante de enfriamiento
          iter - iteraciones por nivel de temperatura
      min_temp - temperatura minima, bandera de finalizacion
    """
    num_candidatos = len(cobertura)
    #Guardar solucion actual
    actual = sol_inicial.copy()
    cobertura_actual,_ = calc_cobertura_total(actual,cobertura)

    #Guardar la mejor solucion
    mejor = actual.copy()
    mejor_cobertura = cobertura_actual

    T = T0 #Inicializamos la mejor temp
    #print(f"SA: cobertura inicial = {cobertura_actual}")

    while T > min_temp:
        for _ in range(iter):
            vecino, j_in, j_out = vecino_random(actual, num_candidatos)
            cob_vecino,_ = calc_cobertura_total(vecino,cobertura)

            delta = cob_vecino - cobertura_actual

            if delta >= 0:
                accept = True
            else:
                rho = math.exp(delta/T)
                accept = (random.random() < rho)
            if accept:
                actual = vecino
                cobertura_actual = cob_vecino

                if cobertura_actual>mejor_cobertura:
                    mejor = actual
                    mejor_cobertura = cobertura_actual
        T *= a
    print(f"SA: mejor cobertura encontrada = {mejor_cobertura}")
    return mejor, mejor_cobertura

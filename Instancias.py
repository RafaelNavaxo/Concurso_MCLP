import pandas as pd
import numpy as np
import glob
import os
import Solver
import csv
# Sin la extensión .py

lista_dfs = [pd.read_csv(f) for f in glob.glob('Casos/*.csv')]

lista_tuplas = [
    (os.path.basename(f), pd.read_csv(f))
    for f in glob.glob('Casos/*.csv')
]

for nombre, df in lista_tuplas:
    print(f"Analizando el archivo: {nombre}")
    if "id" not in df.columns:
        df["id"] = df.index

    puntos = df[["id", "LATITUD", "LONGITUD"]].copy()
    puntos_np = puntos[["LONGITUD", "LATITUD"]].to_numpy()
    cords_km = Solver.conv_lat_lon2km(puntos)

    # -----Construccion de matriz de cobertura Manhattan
    R = [0.5,1.0,1.5,2.0,2.5] # Vector de radios en Km
    Cobertura = [[] for _ in range(len(R))]
    N = len(cords_km)
    for i in range(len(R)):
        for j in range(N):
            x_j, y_j = cords_km[j]
            Euc_dist = np.sqrt(np.square(cords_km[:,0] - x_j) + np.square(cords_km[:,1] - y_j))
            #M_dist = np.abs(cords_km[:, 0] - x_j) + np.abs(cords_km[:, 1] - y_j)
            idx_Cobertura = np.where(Euc_dist <= R[i])[0]
            Cobertura[i].append(idx_Cobertura)

    P = [3,5,7,10]
    for p in P:
        for r in range(len(R)):
            chosen, covered = Solver.greedy_search_cover(Cobertura[r], cords_km, p, N, (r/2))
            seleccion_opt, cobertura_opt, set_cobertur_opt = Solver.local_optimization(chosen, Cobertura[r], N)
            mejor_sl, mejor_cob = Solver.simulated_annealing(seleccion_opt, Cobertura[r], N)

            print("Cobertura optimizada:",p, mejor_cob)
            print("Porcentaje cobertura:", mejor_cob / N * 100, "%")
            Solucion = [
                ["LATITUD", "LONGITUD"]
            ]
            for b in mejor_sl:
                Lat = df.iloc[b]['LATITUD']
                Lon = df.iloc[b]['LONGITUD']
                fila = [Lat, Lon]
                Solucion.append(fila)

            # 1. Cortamos por el punto
            partes = nombre.split('.')  # Genera: ['ambulancias-2019', 'csv']
            header = partes[0]
            n_amb = str(R[r]).replace(".", "_")
            name_sol =str(header)+"-euclidiana-"+n_amb+"-"+str(p)+".csv"
            with open('Soluciones/Euclidiana/'+name_sol, mode='w', newline='', encoding='utf-8') as archivo:
                writer = csv.writer(archivo)
                writer.writerows(Solucion)
            print("Archivo CSV creado:",name_sol," creado exitosamente")

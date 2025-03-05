"""
La idea del codigo es poder simular una sala de emergencia poder prececir en q
momento habrá mas pacientes y asignar mas personal y haci preveer congestión, mejorar
la asignación de médicos y reducir tiempos de espera
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from scipy.fftpack import fft

# EN ESTE CASO DEFINIREMOS LOS CASO DE ESTADOS DE EMERGENCIA
estados = ["Normal", "Saturado", "Colapsado"]

# Aqui definimos las probabilidades de transición es decir cadenas de Markov
transiciones = {
    "Normal": {"Normal": 0.6, "Saturado": 0.3, "Colapsado": 0.1},
    "Saturado": {"Normal": 0.2, "Saturado": 0.5, "Colapsado": 0.3},
    "Colapsado": {"Normal": 0.1, "Saturado": 0.4, "Colapsado": 0.5},
}


# Simulamos la cademas de Markov para la evolución del hospital

def simular_markov(estado_inicial, pasos):
    estados_simulados = [estado_inicial]
    estado_actual = estado_inicial

    for _ in range(pasos):
        prob = random.random()
        acumulado = 0
        for estado, p in transiciones[estado_actual].items():
            acumulado += p
            if prob <= acumulado:
                estado_actual = estado
                break
        estados_simulados.append(estado_actual)
    return estados_simulados


# Para la simulacion de llegada de los pacientes utilizaremos el Proceso de Poisson
def simulado_llegadas_poisson(tasa_llegada, horas):
    return np.random.poisson(tasa_llegada, horas)


# Aqui aplicaremos la Transformada de Fourier(FTT) para poder analizar patrones en las llegadas
def analizar_fft(llegadas):
    transformada = np.abs(fft(llegadas))
    plt.figure(figsize=(8, 4))
    plt.plot(transformada, 'b-o')
    plt.title("Analisis de Frecuencia en la llegada de pacientes")
    plt.xlabel("FRECUENCIA")
    plt.ylabel("MAGNITUD")
    plt.grid(True)
    plt.show()


# Ahora ejecutaremos las simulaciones
num_horas = 50
tasa_llegada_pacientes = 5  # Esto corresponde a la tasa de llegada de pacientes en una hora
estado_inicial = "Normal"

# Aqui ejecutaremos las simulaciones
simulacion_estados = simular_markov(estado_inicial, num_horas)
llegadas_pacientes = simulado_llegadas_poisson(tasa_llegada_pacientes, num_horas)

# ........Gráfico 1: En este gráfica observaremos la llegada de paciente en una hora
plt.figure(figsize=(8, 5))
plt.bar(range(num_horas), llegadas_pacientes, color='skyblue', edgecolor='black')
plt.axhline(y=np.mean(llegadas_pacientes), color='r', linestyle='dashed',
            label=f"Promedio: {np.mean(llegadas_pacientes):.2f}")
plt.xlabel("Tiempo(Horas)")
plt.ylabel("Cantidad de pacientes")
plt.title("Llegada de pacientes  por hora(Proceso de Poisson)")
plt.legend()
plt.grid(axis='y')
plt.show()

# ........Gráfico 2: Aqui observaremos la evolución del estado de emergencia

mapa_estados = {"Normal": 2, "Saturado": 1, "Colapsado": 0}
valores_numericos = [mapa_estados[estado] for estado in simulacion_estados]

plt.figure(figsize=(8, 5))
plt.plot(range(len(simulacion_estados)), valores_numericos, 'r--o', markersize=5)
plt.yticks([0, 1, 2], ["Colapsado", "Saturado", "Normal"])
plt.xlabel("Tiempo(horas)")
plt.ylabel("Estado del respectivo hospita+l")
plt.title("Estado de la sala de emergencia a lo largo del tiempo")
plt.grid(True)
plt.show()

# .......Gráfico 3: Aqui obervaremos la gráfica de la cadena de markov
G = nx.DiGraph()

# Ahora agregamos los respectivos nodos
G.add_nodes_from(estados)

# Ahora agregamos las transiciones con pesos
for estado_origen in transiciones:
    for estado_destino, prob in transiciones[estado_origen].items():
        if prob > 0:  # Si hay una transición gráfica entonces se añadira
            G.add_edge(estado_origen, estado_destino, weight=prob)

# Agregamos las posiciciones de los nodos
pos = {"Normal": (0, 1), "Saturado": (1, 0.5), "Colapsado": (2, 0)}

plt.figure(figsize=(6, 4))
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="black", node_size=2000, font_size=10)

# Agregaresmo ahora las etiquetas de pesos en las transiciones

edge_labels = {(u, v): f"{p:.1f}" for u, v, p in G.edges(data="weight")}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

plt.title("Diagrama de cadenas de Markov - Estados del Hosiptal")
plt.show()

# .........Gráfico 4: Aqui observaremos la transforma de Fourier(TF)
analizar_fft(llegadas_pacientes)
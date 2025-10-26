from typing import List
from data import RecsData, ItemId
from sim_item_multi import item_similarity_multi

"""
correlation_multi.py
--------------------
Suma la "correlación múltiple" interna de un individuo (lista de ítems),
usando una similitud de ítem-ítem que combina varias señales (colaborativa,
semántica, popularidad, tiempo, etc.) definida en `item_similarity_multi`.

La métrica resultante es una SUMA sobre todos los pares (a, b) con a < b.
Como todos los individuos tienen el mismo tamaño N, comparar sumas es válido.
Si N variara entre individuos, convendría NORMALIZAR por el número de pares C(N,2).
"""

def correlation_of_individual_multi(
    ind: List[ItemId],
    data: RecsData,
    *,
    cm=None,
    um=None,
    **kwargs
) -> float:
    """
    Calcula la correlación interna de un individuo con una similitud multi-señal.

    Parámetros
    ----------
    ind : List[ItemId]
        Individuo (Top-N) como lista de ítems sin repetidos.
    data : RecsData
        Acceso a ratings, metadatos y todo lo necesario para `item_similarity_multi`.

    Return
    ------
    float
        Suma de similitudes multi-señal para todos los pares del individuo.

    Notas
    -----
    - Complejidad O(N^2) sobre el tamaño del individuo.
    - Evitamos pares duplicados recorriendo solo a<b (no contamos (b,a)).
    - `item_similarity_multi(i, j, data)` debe ser simétrica o al menos consistente.
    """
    n = len(ind)
    if n < 2:
        return 0.0 # sin pares, sin correlación interna
    
    total = 0.0

    # Recorremos combinaciones C(n,2): índices a < b
    for a in range(n-1):
        for b in range(a+1, n):

            # Suma la similitud multi-señal de los dos ítems
            total += item_similarity_multi(ind[a], ind[b], data)
    return total

import math
import random
from typing import List
from data import RecsData, ItemId, UserId
from correlation import correlation_of_individual
from correlation_multi import correlation_of_individual_multi

"""
population.py
---------------
Genera la población inicial y aplica la selección (élite) según correlaciones.

- generate_initial_population: crea M individuos, cada uno es una lista Top-N
    de ítems que el usuario objetivo NO calificó.
- select_top_by_correlation: ordena individuos por una correlación "simple"
    (p.ej., basada en categorías/semántica provista a la función).
- select_top_by_correlation_multi: ordena individuos por una correlación
    "múltiple" (combinación de varias señales/criterios según 'data').

Notas:
- Dentro de CADA individuo no hay ítems repetidos (muestreo sin reemplazo).
- Entre individuos de la población SÍ puede haber repetidos (no se deduplica).
- 'topX' es una proporción en (0, 1]; se usa ceil para garantizar al menos 1.
"""

def generate_initial_population(
    data: RecsData,
    target_user: UserId,
    M: int,
    N: int,
    rng: random.Random = random
) -> List[List[ItemId]]:
    """
    Genera M individuos (listas de ítems). Cada individuo tiene N ítems NO
    calificados por 'target_user'.

    Parámetros
    ----------
    data : RecsData
        Acceso al conjunto de ratings y metadatos. Debe exponer
        'unrated_items_for(user)' para obtener candidatos.
    target_user : UserId
        Usuario objetivo para el cual recomendaremos.
    M : int
        Tamaño de población (cantidad de individuos a crear).
    N : int
        Longitud del individuo (Top-N recomendaciones).
    rng : random.Random
        Generador aleatorio (inyectable para reproducibilidad en tests).

    Retorna
    -------
    List[List[ItemId]]
        Lista con M individuos; cada individuo es una lista de N ítems.

    Lógica
    ------
    1) Obtiene los candidatos = ítems no calificados por el usuario.
    2) Verifica que haya al menos N candidatos (si no, aborta).
    3) Para cada individuo, toma N ítems distintos con rng.sample
        (muestreo sin reemplazo ⇒ no hay duplicados dentro del individuo).
    """
    candidates = data.unrated_items_for(target_user)
    if len(candidates) < N:
        raise ValueError(
            f"No hay suficientes ítems no calificados ({len(candidates)}) para formar individuos de tamaño {N}"
        )
    population: List[List[ItemId]] = []
    for _ in range(M):
        # Muestreo sin reemplazo: N ítems únicos dentro del individuo.
        individual = rng.sample(candidates, N)
        population.append(individual)
    return population

def select_top_by_correlation(
    population: List[List[ItemId]],
    data: RecsData,
    topX: float
) -> List[List[ItemId]]:
    """
    Selecciona la élite (proporción topX) ordenando por una métrica de
    correlación "simple" (definida en 'correlation_of_individual').

    Parámetros
    ----------
    population : List[List[ItemId]]
        Población actual (cada individuo es una lista de ítems).
    data : RecsData
        Debe proveer 'item_categories' (u otra estructura requerida por la
        función de correlación simple).
    topX : float
        Proporción de individuos a conservar, en (0, 1]. Ej.: 0.5 ⇒ 50% mejor.

    Retorna
    -------
    List[List[ItemId]]
        Subconjunto ordenado (desc) de los mejores individuos.

    Lógica
    ------
    1) Calcula un 'score' de correlación por individuo.
    2) Ordena descendentemente por ese 'score'.
    3) Toma ceil(M * topX) individuos (al menos 1).
    """

    # scored = [(score, individuo), ...]
    scored = [
        (correlation_of_individual(ind, data.item_categories), ind)
        for ind in population
    ]

    # Mejores primero (desc)
    scored.sort(key=lambda t: t[0], reverse=True)

    # Cantidad a conservar (élite). ceil garantiza no quedarnos en 0.
    k = max(1, math.ceil(topX * len(population)))

    # Devolvemos solo los individuos (ya ordenados por score)
    return [ind for _, ind in scored[:k]]

def select_top_by_correlation_multi(
    population: List[List[ItemId]],
    data: RecsData,
    topX: float
) -> List[List[ItemId]]:
    """
    Selección por correlación múltiple: combina varias señales/criterios
    (según la implementación de 'correlation_of_individual_multi') para
    puntuar cada individuo y conservar el topX.

    Parámetros
    ----------
    population : List[List[ItemId]]
        Población actual (cada individuo es una lista de ítems).
    data : RecsData
        Estructura completa para calcular el score multi-criterio
        (ratings, semántica, etc., según la implementación).
    topX : float
        Proporción de individuos a conservar, en (0, 1].

    Retorna
    -------
    List[List[ItemId]]
        Subconjunto ordenado (desc) de los mejores individuos.

    Lógica
    ------
    1) Puntúa cada individuo con 'correlation_of_individual_multi(...)'.
    2) Ordena descendentemente por ese puntaje.
    3) Conserva ceil(M * topX) individuos.
    """
    scored = [(correlation_of_individual_multi(ind, data), ind) for ind in population]
    scored.sort(key=lambda t: t[0], reverse=True)
    k = max(1, math.ceil(topX * len(population)))
    return [ind for _, ind in scored[:k]]

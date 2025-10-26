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
    candidates = data.unrated_items_for(target_user)
    if len(candidates) < N:
        raise ValueError(
            f"No hay suficientes ítems no calificados ({len(candidates)}) para formar individuos de tamaño {N}"
        )
    population: List[List[ItemId]] = []
    for _ in range(M):
        individual = rng.sample(candidates, N)
        population.append(individual)
    return population


def select_top_by_correlation(
    population: List[List[ItemId]],
    data: RecsData,
    topX: float
) -> List[List[ItemId]]:
    scored = [
        (correlation_of_individual(ind, data.item_categories), ind)
        for ind in population
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    k = max(1, math.ceil(topX * len(population)))
    return [ind for _, ind in scored[:k]]


def select_top_by_correlation_multi(
    population: List[List[ItemId]],
    data: RecsData,
    topX: float,
    *,
    cm=None,
    um=None
) -> List[List[ItemId]]:
    """
    Selección por correlación múltiple (versión híbrida):
    combina señales colaborativas y, si se proveen, contenido (cm) y promedios (um).
    """

    # Calcular el score híbrido para cada individuo
    scored = [
        (correlation_of_individual_multi(ind, data, cm=cm, um=um), ind)
        for ind in population
    ]

    scored.sort(key=lambda t: t[0], reverse=True)
    k = max(1, math.ceil(topX * len(population)))
    return [ind for _, ind in scored[:k]]

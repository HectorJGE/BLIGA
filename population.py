import math
import random
from typing import List
from data import RecsData, ItemId, UserId
from correlation import correlation_of_individual

def generate_initial_population(
    data: RecsData,
    target_user: UserId,
    M: int,
    N: int,
    rng: random.Random = random
) -> List[List[ItemId]]:
    """
    Genera M individuos (listas de ítems).
    Cada individuo tiene N ítems NO calificados por target_user.
    """
    candidates = data.unrated_items_for(target_user)
    if len(candidates) < N:
        raise ValueError(
            f"No hay suficientes ítems no calificados ({len(candidates)}) para formar individuos de tamaño {N}"
        )
    population: List[List[ItemId]] = []
    for _ in range(M):
        individual = rng.sample(candidates, N)  # muestra sin reemplazo
        population.append(individual)
    return population

def select_top_by_correlation(
    population: List[List[ItemId]],
    data: RecsData,
    topX: float
) -> List[List[ItemId]]:
    """
    Selecciona el topX (proporción) de los individuos con mayor correlación(z).
    topX debe estar en (0,1].
    """
    scored = [
        (correlation_of_individual(ind, data.item_categories), ind)
        for ind in population
    ]
    scored.sort(key=lambda t: t[0], reverse=True)
    k = max(1, math.ceil(topX * len(population)))
    return [ind for _, ind in scored[:k]]

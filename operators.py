
import random
from typing import List
from data import ItemId

def tournament_selection(population: List[List[ItemId]], k: int, rng: random.Random = random) -> List[ItemId]:
    """Selecciona el mejor individuo de un torneo de tamaño k."""
    candidates = rng.sample(population, k)
    # aquí simplemente elegimos el primero (suponiendo que ya están ordenados por fitness)
    return candidates[0]

def one_point_crossover(parent1: List[ItemId], parent2: List[ItemId], rng: random.Random = random) -> List[ItemId]:
    """Cruza dos padres para generar un hijo (sin duplicados)."""
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener el mismo tamaño")
    n = len(parent1)
    point = rng.randint(1, n - 1)
    child = parent1[:point] + [x for x in parent2[point:] if x not in parent1[:point]]
    # Si faltan genes, completamos con los que no están
    while len(child) < n:
        candidate = rng.choice(parent2)
        if candidate not in child:
            child.append(candidate)
    return child

def mutate(individual: List[ItemId], candidates: List[ItemId], mutP: float, rng: random.Random = random) -> List[ItemId]:
    """Muta un individuo reemplazando un ítem por otro válido con probabilidad mutP."""
    new_ind = individual[:]
    for i in range(len(new_ind)):
        if rng.random() < mutP:
            new_gene = rng.choice(candidates)
            while new_gene in new_ind:
                new_gene = rng.choice(candidates)
            new_ind[i] = new_gene
    return new_ind

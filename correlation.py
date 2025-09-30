from typing import List, Set, Dict
from data import ItemId, Category

def jaccard_items(cats_p: Set[Category], cats_q: Set[Category]) -> float:
    """
    Calcula la correlación (Jaccard binaria) entre dos ítems p y q.
    corr(p,q) = |categorías comunes| / |categorías unión|
    Rango: 0..1
    """
    if not cats_p and not cats_q:
        return 0.0  # sin categorías
    inter = len(cats_p & cats_q)
    union = len(cats_p | cats_q)
    return inter / union if union > 0 else 0.0

def correlation_of_individual(
    individual: List[ItemId],
    item_categories: Dict[ItemId, Set[Category]]
) -> float:
    """
    Calcula la correlación de un individuo z:
    correlation(z) = sum corr(p,q) para todos los pares (p,q) en el individuo.
    """
    n = len(individual)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        p = individual[i]
        cats_p = item_categories.get(p, set())
        for j in range(i + 1, n):
            q = individual[j]
            cats_q = item_categories.get(q, set())
            total += jaccard_items(cats_p, cats_q)
    return total

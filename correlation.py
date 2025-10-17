from typing import List, Set, Dict
from data import ItemId, Category

"""
correlation.py (métrica simple)
--------------------------------
Define dos utilidades:

1) jaccard_items: similitud binaria entre dos ítems a partir de sus categorías/atributos.
2) correlation_of_individual: “correlación interna” de un individuo, sumando las
    similitudes Jaccard de todos los pares de ítems del individuo.

Notas:
- Jaccard ∈ [0,1]. 1 = idénticas categorías, 0 = sin solapamiento.
- La correlación del individuo es una SUMA sobre pares (no está normalizada).
    Como todos los individuos tienen el mismo tamaño N, la comparación es válida.
    Si N variara, convendría normalizar por el número de pares (n*(n-1)/2).
"""

def jaccard_items(cats_p: Set[Category], cats_q: Set[Category]) -> float:
    """
    Calcula la similitud Jaccard entre dos ítems p y q a partir de sus categorías.

    Fórmula:
        J(p, q) = |cats_p ∩ cats_q| / |cats_p ∪ cats_q|   ∈ [0, 1]

    Casos borde:
    - Si ambos no tienen categorías (∅, ∅) ⇒ retornamos 0.0 (sin evidencia).
    """
    if not cats_p and not cats_q:
        return 0.0  # sin categorías
    
    # Intersección y unión de categorías
    inter = len(cats_p & cats_q)
    union = len(cats_p | cats_q)

    # Evita división por cero si, por alguna razón, la unión fuera vacía
    return inter / union if union > 0 else 0.0

def correlation_of_individual(
    individual: List[ItemId],
    item_categories: Dict[ItemId, Set[Category]]
) -> float:
    """
    Calcula la "correlación interna" de un individuo z (lista de ítems).

    Idea:
    - Recorremos todos los pares no ordenados (p, q) del individuo (i < j)
        y sumamos Jaccard(p, q).
    - A mayor suma, más "coherente/parecido" es el conjunto a nivel semántico.

    Importante:
    - Complejidad O(N^2) sobre el tamaño del individuo.
    - No normaliza por la cantidad de pares; como N es fijo para todos los
        individuos, se puede comparar la suma directamente.

    Params
    ------
    individual : list[ItemId]
        Lista de ítems (sin repetidos) que forma el individuo.
    item_categories : dict[ItemId, set[Category]]
        Mapa ítem → conjunto de categorías/atributos.

    Return
    ------
    float
        Suma de Jaccard en todos los pares del individuo.
    """
    n = len(individual)
    if n < 2:
        # Con menos de 2 ítems no hay pares que comparar
        return 0.0
    
    total = 0.0

    # Doble lazo sobre pares no repetidos (combinaciones C(n,2))
    for i in range(n - 1):
        p = individual[i]
        cats_p = item_categories.get(p, set()) # si falta, asumimos sin categorías
        for j in range(i + 1, n):
            q = individual[j]
            cats_q = item_categories.get(q, set())
            total += jaccard_items(cats_p, cats_q)
    return total

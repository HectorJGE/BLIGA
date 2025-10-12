from typing import List
from .data import RecsData, ItemId
from .sim_item_multi import item_similarity_multi

def correlation_of_individual_multi(ind: List[ItemId], data: RecsData) -> float:
    n = len(ind)
    if n < 2: return 0.0
    total = 0.0
    for a in range(n-1):
        for b in range(a+1, n):
            total += item_similarity_multi(ind[a], ind[b], data)
    return total

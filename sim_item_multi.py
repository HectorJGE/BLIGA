import math
from typing import Dict, Set, Any
from .data import ItemId, Category, RecsData

def jaccard(a: Set, b: Set) -> float:
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def sim_year(y1: int | None, y2: int | None, max_gap: int = 20) -> float:
    if not y1 or not y2: return 0.0
    gap = min(abs(y1 - y2), max_gap)
    return 1.0 - gap / max_gap  # lineal; 1 si igual, 0 si dif ≥ max_gap

def cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    if not v1 or not v2: return 0.0
    inter = set(v1) & set(v2)
    num = sum(v1[k]*v2[k] for k in inter)
    den1 = math.sqrt(sum(x*x for x in v1.values()))
    den2 = math.sqrt(sum(x*x for x in v2.values()))
    return (num / (den1*den2)) if den1 and den2 else 0.0

def item_similarity_multi(i: ItemId, j: ItemId, data: RecsData,
                          w_cat=0.5, w_dir=0.2, w_act=0.2, w_year=0.1) -> float:
    cats_i = data.item_categories.get(i, set())
    cats_j = data.item_categories.get(j, set())
    s_cat = jaccard(cats_i, cats_j)

    meta_i = (data.item_meta or {}).get(i, {})
    meta_j = (data.item_meta or {}).get(j, {})
    s_dir = 1.0 if meta_i.get("director") and meta_i.get("director") == meta_j.get("director") else 0.0
    s_act = jaccard(meta_i.get("actores", set()), meta_j.get("actores", set()))
    s_year = sim_year(meta_i.get("año"), meta_j.get("año"))

    # combinación ponderada
    score = w_cat*s_cat + w_dir*s_dir + w_act*s_act + w_year*s_year
    # clamp [0,1]
    return max(0.0, min(1.0, score))

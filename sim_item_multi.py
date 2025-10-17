import math
from typing import Dict, Set, Any, Optional
from data import ItemId, Category, RecsData

"""
sim_item_multi.py
-----------------
Similitud multi-señal entre dos ítems (películas). Combina:

- Categorías/Géneros (Jaccard)
- Director (igualdad exacta)
- Actores (Jaccard sobre el conjunto de actores)
- Año (función lineal: más cerca ⇒ mayor similitud)

Los pesos (w_cat, w_dir, w_act, w_year) ponderan cada señal.
El resultado final se limita a [0, 1] para estabilidad.
"""

def jaccard(a: Set, b: Set) -> float:
    """
    Similitud Jaccard entre dos conjuntos: |A∩B| / |A∪B|.
    Si ambos vacíos ⇒ 0.0 (no hay evidencia).
    """
    if not a and not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def sim_year(y1: Optional[int], y2: Optional[int], max_gap: int = 20) -> float:
    """
    Similitud por año de lanzamiento (0..1).
    - 1.0 si y1 == y2
    - Decrece linealmente hasta 0.0 cuando |y1 - y2| >= max_gap
    - Si falta algún año ⇒ 0.0
    """
    if not y1 or not y2: return 0.0
    gap = min(abs(y1 - y2), max_gap)
    return 1.0 - gap / max_gap  # lineal; 1 si igual, 0 si dif ≥ max_gap

def cosine_sparse(v1: Dict[str, float], v2: Dict[str, float]) -> float:
    """
    Coseno entre vectores dispersos representados como dict[key -> weight].
    Útil para tags/TF-IDF/embeddings discretos.
    Si alguno vacío ⇒ 0.0.
    """
    if not v1 or not v2: return 0.0
    inter = set(v1) & set(v2)
    num = sum(v1[k]*v2[k] for k in inter)
    den1 = math.sqrt(sum(x*x for x in v1.values()))
    den2 = math.sqrt(sum(x*x for x in v2.values()))
    return (num / (den1*den2)) if den1 and den2 else 0.0

def item_similarity_multi(i: ItemId, j: ItemId, data: RecsData,
    w_cat=0.5, w_dir=0.2, w_act=0.2, w_year=0.1) -> float:
    """
    Similitud multi-señal entre ítems i y j (0..1), combinando categorías, director,
    actores y año según pesos.

    Parámetros
    ----------
    i, j : ItemId
        Identificadores de los ítems a comparar.
    data : RecsData
        Provee 'item_categories' (mapa item->set de categorías) y 'item_meta' (director, actores, año, ...).
    w_* : float
        Pesos para cada señal (la suma típica es 1.0, pero no es requisito estricto).

    Return
    ------
    float en [0, 1]
    """

    # --- 1) Categorías/Géneros: Jaccard ---
    cats_i = data.item_categories.get(i, set())
    cats_j = data.item_categories.get(j, set())
    s_cat = jaccard(cats_i, cats_j)

    # --- 2) Metadatos básicos: director, actores, año ---
    meta_i = (data.item_meta or {}).get(i, {})
    meta_j = (data.item_meta or {}).get(j, {})

    # Director: 1.0 si es exactamente el mismo (y no nulo), 0 si distinto o faltante.
    s_dir = 1.0 if meta_i.get("director") and meta_i.get("director") == meta_j.get("director") else 0.0
    
    # Actores: Jaccard sobre el conjunto de actores (si falta alguno ⇒ set() vacío).
    s_act = jaccard(meta_i.get("actores", set()), meta_j.get("actores", set()))

    # Año: similitud lineal según diferencia acotada por max_gap.
    s_year = sim_year(meta_i.get("año"), meta_j.get("año"))

    # --- 3) Combinación ponderada (suma lineal) ---
    score = w_cat*s_cat + w_dir*s_dir + w_act*s_act + w_year*s_year
    
    # --- 4) Asegurar rango [0, 1] (numeritos fuera por redondeos/pesos raros) ---
    return max(0.0, min(1.0, score))

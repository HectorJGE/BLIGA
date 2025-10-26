# bliga/score_aggregator.py
from typing import Dict, List, Tuple
from data import RecsData, UserId, ItemId
from prediction import predict_rating
from sim_item_multi import item_similarity_multi

"""
score_aggregator.py
-------------------
Define cómo puntuar un ítem y un individuo combinando 3 señales:

1) CF (Collaborative Filtering): predicción de rating del usuario para el ítem.
2) Contenido (Content-based): similitud del ítem con los ítems que el usuario "amó".
3) Popularidad: cuántos usuarios calificaron ese ítem.

Luego, agrega con pesos Ω = (w_cf, w_cont, w_pop) a nivel ítem e
individual (suma de los ítems del Top-N).
"""

def score_cf(
    target_user: UserId,
    item: ItemId,
    data: RecsData,
    *,
    cm=None,
    um=None
) -> float:
    """
    Puntaje CF (0..5) para 'target_user' en 'item'.
    Intenta pasar 'user_means' si la firma de predict_rating lo soporta;
    de lo contrario, cae al llamado sin ese argumento.
    """
    try:
        score = predict_rating(target_user, item, data, user_means=um)  # puede no existir el kwarg
    except TypeError:
        score = predict_rating(target_user, item, data)
    return max(0.0, min(5.0, score))


def score_content(
    target_user: UserId,
    item: ItemId,
    data: RecsData,
    like_thr: float = 4.0,
    *,
    cm=None
) -> float:
    """
    Puntaje basado en contenido (0..1).
    Si 'cm' está disponible, calcula similitud TF-IDF promedio entre
    el ítem y los ítems que el usuario calificó con rating >= like_thr.
    Caso contrario, usa 'item_similarity_multi' (fallback colaborativo/semántico).
    """
    user_r = data.ratings.get(target_user, {})
    liked = [i for i, r in user_r.items() if r >= like_thr]
    if not liked:
        return 0.0

    if cm is not None:
        import numpy as np
        vec_item = cm.item_vector(item)
        if vec_item is None:
            return 0.0
        vecs_liked = [cm.item_vector(li) for li in liked if cm.item_vector(li) is not None]
        if not vecs_liked:
            return 0.0
        sims = [float(vec_item @ v) for v in vecs_liked]
        return float(np.mean(sims)) if sims else 0.0

    sims = [item_similarity_multi(item, li, data) for li in liked if li != item]
    return sum(sims) / len(sims) if sims else 0.0


def score_popularity(item: ItemId, data: RecsData) -> float:
    """
    Popularidad (0..1) del ítem basada en conteo de usuarios que lo calificaron.
    Usa índices precalculados si están en 'data'.
    """
    if hasattr(data, "item_pop") and hasattr(data, "max_item_pop"):
        m = data.max_item_pop or 1
        return data.item_pop.get(item, 0) / m
    cnt = sum(1 for u, ur in data.ratings.items() if item in ur)
    max_cnt = max((len(ur) for ur in data.ratings.values()), default=1)
    return cnt / max_cnt if max_cnt else 0.0


def aggregate_item_score(
    target_user: UserId,
    item: ItemId,
    data: RecsData,
    omega: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    *,
    cm=None,
    um=None
) -> float:
    """
    Score final de un ítem combinando señales con pesos Ω.
    Si se proveen 'cm' y/o 'um', se usan (modelo cacheado).
    """
    w_cf, w_cont, w_pop = omega
    s_cf = score_cf(target_user, item, data, cm=cm, um=um) / 5.0
    s_cont = score_content(target_user, item, data, cm=cm)
    s_pop = score_popularity(item, data)
    return w_cf * s_cf + w_cont * s_cont + w_pop * s_pop


def aggregate_individual(
    ind: List[ItemId],
    target_user: UserId,
    data: RecsData,
    omega: Tuple[float, float, float] = (0.5, 0.3, 0.2),
    *,
    cm=None,
    um=None
) -> float:
    """
    Score final de un individuo (suma de los ítems).
    Admite cm/um para cálculo híbrido.
    """
    return sum(aggregate_item_score(target_user, i, data, omega, cm=cm, um=um) for i in ind)

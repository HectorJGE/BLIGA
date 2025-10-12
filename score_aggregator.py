
# bliga/score_aggregator.py
from typing import Dict, List, Tuple
from data import RecsData, UserId, ItemId
from prediction import predict_rating
from sim_item_multi import item_similarity_multi

def score_cf(target_user: UserId, item: ItemId, data: RecsData) -> float:
    # normaliza a [0,5], asumiendo que predict_rating ya lo devuelve en ese rango o similar
    score = predict_rating(target_user, item, data)
    return max(0.0, min(5.0, score))

def score_content(target_user: UserId, item: ItemId, data: RecsData, like_thr: float = 4.0) -> float:
    # promedio de similitud con ítems del usuario con rating >= like_thr
    user_r = data.ratings.get(target_user, {})
    liked = [i for i, r in user_r.items() if r >= like_thr]
    if not liked: return 0.0
    sims = [item_similarity_multi(item, li, data) for li in liked if li != item]
    return sum(sims)/len(sims) if sims else 0.0

def score_popularity(item: ItemId, data: RecsData) -> float:
    # cuenta de usuarios que calificaron el ítem, normalizado por el máximo
    cnt = sum(1 for u, ur in data.ratings.items() if item in ur)
    max_cnt = max((len(ur) for ur in data.ratings.values()), default=1)
    return cnt / max_cnt if max_cnt else 0.0

def aggregate_item_score(target_user: UserId, item: ItemId, data: RecsData,
                         omega: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
    w_cf, w_cont, w_pop = omega
    s_cf = score_cf(target_user, item, data) / 5.0     # escala [0,1]
    s_cont = score_content(target_user, item, data)    # ya [0,1]
    s_pop = score_popularity(item, data)               # [0,1]
    return w_cf*s_cf + w_cont*s_cont + w_pop*s_pop

def aggregate_individual(ind: List[ItemId], target_user: UserId, data: RecsData,
                         omega: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
    return sum(aggregate_item_score(target_user, i, data, omega) for i in ind)

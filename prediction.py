from typing import Dict, List
from data import RecsData, UserId, ItemId
from similarity import user_similarity

"""
prediction.py
-------------
Predice calificaciones usando la fórmula clásica de Resnick (CF basado en vecinos):

    r̂(u, i) = r̄_u +  ( Σ_v sim(u, v) * (r(v, i) - r̄_v) ) / ( Σ_v |sim(u, v)| )

Notas:
- Se centra por medias (mean-centering).
- Si no hay vecinos o el denominador queda en 0, retorna el promedio del usuario (backoff).
- No se acota la salida; si necesitás [0,5], clampleá en el consumidor.
"""

# Cache manual de similitudes usuario-usuario (clave simétrica)
_SIM_CACHE: Dict[tuple, float] = {}

def _user_sim_cached(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    """
    Devuelve sim(u1,u2) cacheada. La clave es simétrica para no duplicar (u1,u2)/(u2,u1).
    """
    key = (u1, u2) if u1 <= u2 else (u2, u1)
    val = _SIM_CACHE.get(key)
    if val is not None:
        return val
    val = user_similarity(u1, u2, ratings)
    if len(_SIM_CACHE) < 200_000:  # límite blando para evitar crecer sin fin
        _SIM_CACHE[key] = val
    return val


def predict_rating(
    target_user: UserId,
    item: ItemId,
    data: RecsData,
    k: int = 50,
    min_sim: float = 0.05
) -> float:
    """
    Predice la calificación que el usuario activo (target_user) daría a un ítem (item),
    usando Resnick (vecinos ponderados por similitud y centrado por la media).
    """

    # Media del usuario objetivo (backoff seguro)
    if hasattr(data, "user_mean"):
        ru_mean = data.user_mean.get(target_user, 0.0)
    else:
        ru = data.ratings.get(target_user, {})
        ru_mean = (sum(ru.values()) / len(ru)) if ru else 0.0

    # Vecinos = usuarios que calificaron 'item'
    if hasattr(data, "users_by_item"):
        neighbors = [u for u in data.users_by_item.get(item, set()) if u != target_user]
    else:
        neighbors = [u for u, ratings in data.ratings.items() if item in ratings and u != target_user]

    if not neighbors:
        return ru_mean

    # Similitudes con umbral y top-k por |sim|
    sims = []
    for u in neighbors:
        sim = _user_sim_cached(target_user, u, data.ratings)
        if abs(sim) >= min_sim:
            sims.append((u, sim))

    if not sims:
        return ru_mean

    sims.sort(key=lambda t: abs(t[1]), reverse=True)
    sims = sims[:k]

    # Fórmula de Resnick con centrado por media
    num = 0.0
    den = 0.0
    for u, sim in sims:
        if hasattr(data, "user_mean"):
            rv_mean = data.user_mean.get(u, 0.0)
        else:
            rv = data.ratings[u]
            rv_mean = (sum(rv.values()) / len(rv)) if rv else 0.0
        rv_i = data.ratings[u][item]
        num += sim * (rv_i - rv_mean)
        den += abs(sim)

    if den == 0.0:
        return ru_mean

    return ru_mean + (num / den)


def predict_individual(
    individual: List[ItemId],
    target_user: UserId,
    data: RecsData
) -> float:
    """
    Fitness/score del individuo como suma de predicciones CF para sus ítems.

    Como todos los individuos tienen la misma longitud N, sumar (en vez de promediar)
    es consistente para comparar entre individuos.

    Parámetros
    ----------
    individual : List[ItemId]
        Ítems del Top-N.
    target_user : UserId
        Usuario objetivo.
    data : RecsData
        Estructura de ratings.

    Return
    ------
    float
        Suma de predicciones CF sobre todos los ítems del individuo.
    """
    return sum(predict_rating(target_user, i, data) for i in individual)

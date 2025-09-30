from typing import Dict, List
from data import RecsData, UserId, ItemId
from similarity import user_similarity

def predict_rating(
    target_user: UserId,
    item: ItemId,
    data: RecsData
) -> float:
    """
    Predice la calificación que el usuario activo (target_user) daría a un ítem (item).
    Usa la fórmula de Resnick (ajuste ponderado con similitud).
    """
    # Usuarios que calificaron el ítem
    neighbors = [u for u, ratings in data.ratings.items() if item in ratings and u != target_user]
    if not neighbors:
        return 0.0

    # Promedio del usuario activo
    ru_mean = sum(data.ratings[target_user].values()) / len(data.ratings[target_user]) if data.ratings[target_user] else 0.0

    num = 0.0
    den = 0.0
    for u in neighbors:
        sim = user_similarity(target_user, u, data.ratings)
        if sim == 0:
            continue
        ru_i = data.ratings[u][item]
        ru_avg = sum(data.ratings[u].values()) / len(data.ratings[u])
        num += sim * (ru_i - ru_avg)
        den += abs(sim)
    if den == 0:
        return ru_mean
    return ru_mean + (num / den)

def predict_individual(
    individual: List[ItemId],
    target_user: UserId,
    data: RecsData
) -> float:
    """
    Fitness final: suma de predicciones para todos los ítems del individuo.
    """
    return sum(predict_rating(target_user, i, data) for i in individual)

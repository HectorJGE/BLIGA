import math
from typing import Dict, List, Set
from data import RecsData, UserId, ItemId


def pearson_similarity(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    """
    Calcula la similitud de Pearson entre dos usuarios sobre ítems comunes.
    Devuelve valor entre -1 y 1.
    """
    r1 = ratings.get(u1, {})
    r2 = ratings.get(u2, {})
    common_items = set(r1.keys()) & set(r2.keys())
    if not common_items:
        return 0.0

    # medias
    mean1 = sum(r1[i] for i in common_items) / len(common_items)
    mean2 = sum(r2[i] for i in common_items) / len(common_items)

    # numerador y denominadores
    num = sum((r1[i] - mean1) * (r2[i] - mean2) for i in common_items)
    den1 = math.sqrt(sum((r1[i] - mean1) ** 2 for i in common_items))
    den2 = math.sqrt(sum((r2[i] - mean2) ** 2 for i in common_items))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)

def jaccard_users(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    items1 = set(ratings.get(u1, {}).keys())
    items2 = set(ratings.get(u2, {}).keys())
    if not items1 and not items2:
        return 0.0
    inter = len(items1 & items2)
    union = len(items1 | items2)
    return inter / union if union > 0 else 0.0

def user_similarity(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    return pearson_similarity(u1, u2, ratings) * jaccard_users(u1, u2, ratings)

def similarity_of_individual(
    individual: List[ItemId],
    data: RecsData,
    target_user: UserId
) -> float:
    """
    Similarity(z) = sum Sim(targetUser, u) para todos los usuarios u que calificaron ≥1 ítem en z
    """
    selected_users: Set[UserId] = {
        u for u, ur in data.ratings.items() if any(i in ur for i in individual)
    }
    total = 0.0
    for u in selected_users:
        if u == target_user:
            continue
        total += user_similarity(target_user, u, data.ratings)
    return total

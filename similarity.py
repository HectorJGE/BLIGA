import math
from typing import Dict, List, Set
from data import RecsData, UserId, ItemId


def pearson_similarity(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    r1 = ratings.get(u1, {})
    r2 = ratings.get(u2, {})
    common_items = set(r1.keys()) & set(r2.keys())
    if not common_items:
        return 0.0
    mean1 = sum(r1[i] for i in common_items) / len(common_items)
    mean2 = sum(r2[i] for i in common_items) / len(common_items)
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
    target_user: UserId,
    *,
    cm=None,
    um=None
) -> float:
    """
    Similitud colaborativa + (opcionalmente) similitud de contenido.

    Si se pasa 'cm' (ContentModel) y 'um' (user_means):
        -> combina la similitud colaborativa (entre usuarios)
           con la similitud semántica entre los ítems del individuo.

    Retorna una métrica híbrida normalizada en [0,1] aprox.
    """

    # --- 1) Parte colaborativa (como antes) ---
    selected_users: Set[UserId] = set()
    ubi = getattr(data, "users_by_item", None)
    if ubi is None:
        selected_users = {u for u, ur in data.ratings.items() if any(i in ur for i in individual)}
    else:
        for i in individual:
            selected_users |= ubi.get(i, set())

    total_collab = 0.0
    for u in selected_users:
        if u == target_user:
            continue
        total_collab += user_similarity(target_user, u, data.ratings)

    # --- 2) Parte de contenido (si hay modelo cm) ---
    content_score = 0.0
    if cm is not None:
        import numpy as np

        # Vectores TF-IDF de los ítems del individuo
        vecs = [cm.item_vector(i) for i in individual if cm.item_vector(i) is not None]
        if vecs:
            # Promedio de vectores del individuo = “perfil de recomendación”
            profile = np.mean(vecs, axis=0)
            # Promedio de similitudes coseno internas (coherencia semántica)
            sims = [profile @ v for v in vecs]
            content_score = float(np.mean(sims))

    # --- 3) Ponderar ambas partes ---
    alpha = 0.7  # peso colaborativo
    beta = 0.3   # peso de contenido

    hybrid_score = alpha * total_collab + beta * content_score
    return hybrid_score

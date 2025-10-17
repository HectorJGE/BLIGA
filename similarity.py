import math
from typing import Dict, List, Set
from data import RecsData, UserId, ItemId


def pearson_similarity(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    """
    Similitud de Pearson entre dos usuarios sobre los ÍTEMS EN COMÚN.
    Devuelve un valor en [-1, 1].

    Notas:
    - Se calculan las medias usando solo los ítems co-calificados (intersección).
    - Si no hay intersección, no hay evidencia ⇒ 0.0.
    - Si alguno de los usuarios tiene varianza 0 en esos ítems (den=0), ⇒ 0.0.
    """
    r1 = ratings.get(u1, {})
    r2 = ratings.get(u2, {})
    common_items = set(r1.keys()) & set(r2.keys())
    if not common_items:
        return 0.0

    # Medias sobre los ítems en común
    mean1 = sum(r1[i] for i in common_items) / len(common_items)
    mean2 = sum(r2[i] for i in common_items) / len(common_items)

    # Numerador y denominadores del coeficiente de Pearson
    num = sum((r1[i] - mean1) * (r2[i] - mean2) for i in common_items)
    den1 = math.sqrt(sum((r1[i] - mean1) ** 2 for i in common_items))
    den2 = math.sqrt(sum((r2[i] - mean2) ** 2 for i in common_items))
    if den1 == 0 or den2 == 0:
        return 0.0
    return num / (den1 * den2)

def jaccard_users(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    """
    Jaccard entre conjuntos de ítems calificados por dos usuarios:
        |I(u1) ∩ I(u2)| / |I(u1) ∪ I(u2)|
    Devuelve [0,1]. Si ambos conjuntos son vacíos ⇒ 0.0.
    """
    items1 = set(ratings.get(u1, {}).keys())
    items2 = set(ratings.get(u2, {}).keys())
    if not items1 and not items2:
        return 0.0
    inter = len(items1 & items2)
    union = len(items1 | items2)
    return inter / union if union > 0 else 0.0

def user_similarity(u1: UserId, u2: UserId, ratings: Dict[UserId, Dict[ItemId, float]]) -> float:
    """
    Similitud compuesta usuario-usuario:
    Pearson(u1,u2) * Jaccard(u1,u2)

    Intuición:
    - Pearson captura la correlación de preferencias en la intersección.
    - Jaccard actúa como "confianza" (shrinkage): si comparten pocos ítems,
        reduce el impacto de un Pearson alto/ruidoso.
    """
    return pearson_similarity(u1, u2, ratings) * jaccard_users(u1, u2, ratings)

def similarity_of_individual(
    individual: List[ItemId],
    data: RecsData,
    target_user: UserId
) -> float:
    """
    Suma de similitudes del usuario objetivo contra TODOS los usuarios
    que calificaron al menos un ítem del individuo.

    Definición:
        Similarity(z) = Σ_{u ∈ U(z)} Sim(target_user, u)
        donde U(z) = { usuarios que calificaron ≥ 1 ítem en z }

    Detalles:
    - Se usa un set para no contar dos veces al mismo usuario si calificó
        varios ítems del individuo.
    - Se excluye al propio target_user si aparece en ratings.
    - Devuelve un valor ≥ 0 (porque user_similarity usa Jaccard ≥ 0; Pearson
        puede ser negativo pero Jaccard amortigua; si necesitás >=0 estricto,
        podrías max(sim,0) en user_similarity).
    """

    # Conjunto de usuarios que calificaron al menos un ítem del individuo
    selected_users: Set[UserId] = {
        u for u, ur in data.ratings.items() if any(i in ur for i in individual)
    }
    total = 0.0
    for u in selected_users:
        if u == target_user:
            continue # no nos comparamos con nosotros mismos
        total += user_similarity(target_user, u, data.ratings)
    return total

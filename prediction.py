from typing import Dict, List
from data import RecsData, UserId, ItemId
from similarity import user_similarity

"""
prediction.py
-------------
Predice calificaciones usando la fórmula clásica de Resnick (CF basado en vecinos):

    r̂(u, i) = r̄_u +  ( Σ_v sim(u, v) * (r(v, i) - r̄_v) ) / ( Σ_v |sim(u, v)| )

donde:
- r̂(u, i)   : predicción del usuario u para el ítem i
- r̄_u       : promedio de ratings del usuario u
- r(v, i)    : rating del vecino v sobre el ítem i
- r̄_v       : promedio de ratings del vecino v
- sim(u, v)  : similitud entre usuarios u y v (cualquier métrica simétrica)

Notas:
- Se centra por medias (mean-centering) para corregir sesgos de usuarios "altos" o "bajos".
- Si no hay vecinos o el denominador queda en 0, devolvemos el promedio de u (backoff seguro).
- No se acota la salida; si necesitás [0, 5], podés hacer clamp en el consumidor.
"""

def predict_rating(
    target_user: UserId,
    item: ItemId,
    data: RecsData
) -> float:
    """
    Predice la calificación que el usuario activo (target_user) daría a un ítem (item),
    usando Resnick (vecinos ponderados por similitud y centrado por la media).

    Parámetros
    ----------
    target_user : UserId
        Usuario para el cual predecimos.
    item : ItemId
        Ítem objetivo.
    data : RecsData
        Debe proveer 'ratings' como dict[user -> dict[item -> rating]].

    Return
    ------
    float
        Predicción (no forzada a rango). Si no hay vecinos válidos, retorna el promedio del usuario.
    """

    # 1) Vecinos: usuarios (distintos a target_user) que calificaron el ítem.
    neighbors = [u for u, ratings in data.ratings.items() if item in ratings and u != target_user]
    if not neighbors:
        # Sin evidencia: devolvemos 0.0 (podrías usar promedio global si preferís).
        return 0.0

    # 2) Promedio del usuario activo (r̄_u). Si no tiene ratings, 0.0.
    ru_mean = sum(data.ratings[target_user].values()) / len(data.ratings[target_user]) if data.ratings[target_user] else 0.0

    # 3) Acumular numerador y denominador de Resnick.
    num = 0.0
    den = 0.0
    for u in neighbors:
        # Similitud entre target_user y el vecino u (definida en similarity.user_similarity).
        sim = user_similarity(target_user, u, data.ratings)
        if sim == 0:
            # Vecino no informativo (o sin solapamiento): lo saltamos.
            continue

        # r(v, i): rating del vecino u para el ítem
        ru_i = data.ratings[u][item]

        # r̄_v: promedio del vecino u
        ru_avg = sum(data.ratings[u].values()) / len(data.ratings[u])

        # Numerador: sim(u,v) * (r(v,i) - r̄_v)  (mean-centering)
        num += sim * (ru_i - ru_avg)

        # Denominador: suma de magnitudes de similitud (|sim|)
        den += abs(sim)
    
    # 4) Si no hay peso total (den == 0), “caemos” al promedio del usuario activo.
    if den == 0:
        return ru_mean
    
    # 5) Predicción final: r̂(u, i)
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

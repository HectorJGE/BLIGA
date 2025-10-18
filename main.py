# main.py
from pathlib import Path
import random, numpy as np
from data_loader_tmdb import load_tmdb_dataset
from population import select_top_by_correlation_multi
from operators import one_point_crossover, mutate
from similarity import similarity_of_individual
from score_aggregator import aggregate_individual
from adaptive import schedule_rates, indiv_rates, temperature, accept_with_sa

"""
Pipeline BLIGA (versión simplificada/extendida):

1) Carga de datos (TMDB/MovieLens).
2) Construcción del conjunto de candidatos: ítems NO calificados por el usuario objetivo,
    pero SÍ calificados por otros (indicio de que hay “evidencia social” disponible).
3) Inicialización de población: M individuos (cada individuo = lista de N ítems candidatos).
4) Evolución por maxGen generaciones:
    - Selección (select_top_by_correlation_multi): conserva la élite/top-X según correlaciones
        múltiples (colaborativas/semánticas, según implemente ese módulo).
    - Ajuste adaptativo de tasas (schedule_rates): define p_crossover y p_mutation por generación
        (normalmente decreciendo o variando según “enfriamiento”/progreso).
    - Cruce (one_point_crossover) y Mutación (mutate) sólo dentro del espacio de candidatos válidos.
    - Reordenamiento por similitud al usuario (similarity_of_individual) para mantener a
        los más parecidos/pertinentes arriba (facilita convergencia).
5) Puntuación final multi-criterio (aggregate_individual) con pesos Ω (OMEGA):
    combina criterios (p.ej., relevancia, diversidad, novedad… según tu implementacion).
6) Devuelve: datos, mejor individuo, y su score.

Este archivo orquesta el flujo; la “inteligencia” de cada etapa vive en los módulos importados.
"""

# Rutas base del proyecto/dataset/pesos
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "the-movies-dataset"
WEIGHTS_PATH = ROOT / "best_weights.npy"

def neighbor_rated_unrated_items(data, user):
    """
    Devuelve ítems candidatos para el 'user': ítems que el usuario NO calificó,
    pero que SÍ fueron calificados por al menos otro usuario.

    Idea: nos enfocamos en ítems con evidencia de otros (vecinos), lo cual
    permite estimar relevancia y similitud de manera más robusta.

    Params
    ------
    data : objeto dataset con:
        - ratings: dict[user_id -> set/list de item_ids calificados]
        - unrated_items_for(u): ítems no calificados por u
    user : str
        Id del usuario objetivo.

    Return
    ------
    list[str]
        Ítems no calificados por 'user' pero calificados por alguien más.
    """
    
    # Ítems que 'user' NO calificó
    unrated = data.unrated_items_for(user)
    # Filtrar de una: ítems no calificados por 'user' que sí tienen calificaciones de otros usuarios.
    return [i for i in unrated if any(u != user and i in ur for u, ur in data.ratings.items())]

def run_bliga(
    target_user: str,
    use_small: bool = False,
    M: int = 20,
    N: int = 10,
    topX: float = 0.5,
    maxGen: int = 6,
    crossoverP: float = 0.9,
    mutP: float = 0.3,
    seed: int = 42,
    return_population: bool = False,
):
    """
    Ejecuta el flujo principal tipo BLIGA: genera Top-N recomendaciones con un GA.

    Params
    ------
    target_user : str
        Usuario para el que recomendaremos.
    use_small : bool
        Si True, usa subset pequeño (más rápido).
    M : int
        Tamaño de población (número de individuos por generación).
    N : int
        Longitud de cada individuo (Top-N a recomendar).
    topX : float
        Proporción de élite/sobrevivientes en cada generación (0 < topX ≤ 1).
        p.ej., 0.5 => se quedan los mejores 50% según correlación múltiple.
    maxGen : int
        Número de generaciones (iteraciones evolutivas).
    crossoverP : float
        Probabilidad máxima de cruce (usada por schedule_rates).
    mutP : float
        Probabilidad máxima de mutación (usada por schedule_rates).
    seed : int
        Semilla para reproducibilidad de la inicialización y operadores estocásticos.

    Return
    ------
    (data, best_ind, best_score)
        data: dataset cargado (para inspecciones posteriores),
        best_ind: lista[str] con los N items del mejor individuo,
        best_score: float con el score multi-criterio final.
    """
    rng = random.Random(seed)

    # 1) Cargar dataset (TMDB/MovieLens) con metadatos y estructura de ratings
    data = load_tmdb_dataset(path=str(DATASET_DIR), use_small=use_small)

    # Si el loader aún no construyó índices/cachés, hacelo acá:
    if not hasattr(data, "users_by_item"):
        from collections import defaultdict
        users_by_item = defaultdict(set)
        for u, ur in data.ratings.items():
            for i in ur.keys():
                users_by_item[i].add(u)
        data.users_by_item = dict(users_by_item)
        data.item_pop = {i: len(us) for i, us in data.users_by_item.items()}
        data.max_item_pop = max(data.item_pop.values(), default=1)
        data.user_mean = {u: (sum(r.values())/len(r) if r else 0.0)
                        for u, r in data.ratings.items()}

    # Validaciones iniciales de usuario/candidatos
    if target_user not in data.ratings:
        raise ValueError(f"target_user {target_user!r} no existe. Ejemplos: {list(data.ratings.keys())[:10]}")

    # 2) Construir candidatos: ítems que el usuario no calificó, pero otros sí
    candidates = neighbor_rated_unrated_items(data, target_user)
    if len(candidates) < N:
        # No se puede armar un individuo de longitud N si no hay suficientes ítems candidatos
        raise ValueError(f"No hay suficientes candidatos ({len(candidates)}) para N={N}")

    # 3) Inicialización de la población: M individuos. Cada individuo = N ítems únicos tomados al azar de 'candidates' (sin repetir).
    population = [rng.sample(candidates, N) for _ in range(M)]

    # 4) Evolución: selección → cruce/mutación → ranking por similitud, por maxGen generaciones
    for gen in range(maxGen):
        # 4.1) Selección top-X según correlaciones múltiples (colaborativa/semántica)
        #      Este paso concentra la “estrategia BLIGA”: filtrar por items co-correlacionados
        #      con el perfil del usuario y entre sí, según tu implementación de 'population.py'.
        best = select_top_by_correlation_multi(population, data, topX)

        # 4.2) Tasas globales por generación (topes suaves)
        pc_global, pm_global = schedule_rates(gen, maxGen, pc_max=crossoverP, pm_max=mutP)

        # 4.3) Fitness para AGA/MGA: usamos similarity_of_individual como fitness
        parent_scores = [(ind, similarity_of_individual(ind, data, target_user)) for ind in best]
        parent_scores.sort(key=lambda t: t[1], reverse=True)
        f_best = parent_scores[0][1] if parent_scores else 0.0
        parents = [ind for ind, _ in parent_scores]

        # Temperatura para SA (cooling)
        T = temperature(gen, T0=1.0, shrink=0.10)

        # 4.4) Elitismo + relleno con AGA (tasas por individuo) + aceptación SA
        new_pop = parents[:]
        while len(new_pop) < M:
            p1 = rng.choice(parents)
            p2 = rng.choice(parents)

            # Fitness del padre principal (referencia)
            f_p1 = similarity_of_individual(p1, data, target_user)

            # Tasas por individuo (acotadas por los topes globales)
            pc_i, pm_i = indiv_rates(
                f_p1, f_best,
                pc_max=pc_global, pc_min=0.5*pc_global,
                pm_max=pm_global, pm_min=0.5*pm_global
            )

            # Cruce probabilístico
            child = one_point_crossover(p1, p2, rng) if rng.random() < pc_i else p1[:]

            # Mutación con tasa individual
            child = mutate(child, candidates, pm_i, rng)

            # Aceptación con SA si es peor; si es mejor, siempre entra
            f_child = similarity_of_individual(child, data, target_user)
            delta = f_p1 - f_child  # >0 si el hijo es peor
            if accept_with_sa(delta, T, rng) and (child not in new_pop):
                new_pop.append(child)

        # 4.5) Re-rank por similitud al usuario objetivo
        #      'similarity_of_individual' devuelve una medida de qué tan alineado está el
        #      conjunto propuesto con el perfil del usuario (colaborativo/semántico).
        sims = [(ind, similarity_of_individual(ind, data, target_user)) for ind in new_pop]
        sims.sort(key=lambda t: t[1], reverse=True)

        # La población para la próxima generación queda ordenada (mejores primero)
        population = [ind for ind, _ in sims]

    # 5) Cargar pesos Ω (si existe un archivo sintonizado por experimentos previos); si falla, usar default.
    try:
        OMEGA = np.load(WEIGHTS_PATH)
    except Exception:
        # Por defecto: (ej.) relevancia 0.5, diversidad 0.3, novedad 0.2 (ajusta a tu definición real)
        OMEGA = (0.5, 0.3, 0.2)

    # 6) Puntuar cada individuo con el agregador multi-criterio y escoger el mejor
    scored = [(ind, aggregate_individual(ind, target_user, data, OMEGA)) for ind in population]
    scored.sort(key=lambda t: t[1], reverse=True)

    best_ind, best_score = scored[0]
    
    if return_population:
        return data, best_ind, best_score, population
    return data, best_ind, best_score

if __name__ == "__main__":
    # Ejecución de ejemplo: ajustá 'target_user' a un ID válido del dataset cargado
    data, best_ind, score = run_bliga(target_user="1", use_small=True)  # <- userId real (por ejemplo "1")
    
    # Helper para imprimir títulos legibles si existen metadatos
    def title(i): return (data.item_meta or {}).get(i, {}).get("title", i)
    
    print("Mejor individuo:", best_ind, "score:", round(score, 4))
    print("Top-N (títulos):", [title(i) for i in best_ind])

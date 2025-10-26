from pathlib import Path
import random, numpy as np
from data_loader_tmdb import load_tmdb_dataset
from population import select_top_by_correlation_multi
from operators import one_point_crossover, mutate
from similarity import similarity_of_individual
from score_aggregator import aggregate_individual
from adaptive import schedule_rates, indiv_rates, temperature, accept_with_sa
from bliga_cache_single import load_content_model_only, load_user_means_only

# Rutas base del proyecto/dataset/pesos
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "the-movies-dataset"
WEIGHTS_PATH = ROOT / "best_weights.npy"

def neighbor_rated_unrated_items(data, user):
    unrated = data.unrated_items_for(user)
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
    rng = random.Random(seed)

    # === 0) Cargar modelos cacheados ===
    print("Cargando modelos desde cache...")
    cm = load_content_model_only(version="v1")
    um = load_user_means_only(version="v1")
    print(f"ContentModel: {cm.item_vecs.shape}, user_means: {um.shape}")

    # === 1) Cargar dataset ===
    data = load_tmdb_dataset(path=str(DATASET_DIR), use_small=use_small)

    # === 2) Añadir cm y um al dataset ===
    data.content_model = cm
    data.user_means = um

    # === 3) Crear índices si faltan ===
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

    if target_user not in data.ratings:
        raise ValueError(f"target_user {target_user!r} no existe.")

    # === 4) Construcción de candidatos ===
    candidates = neighbor_rated_unrated_items(data, target_user)
    if len(candidates) < N:
        raise ValueError(f"No hay suficientes candidatos ({len(candidates)}) para N={N}")

    # === 5) Inicializar población ===
    population = [rng.sample(candidates, N) for _ in range(M)]

    # === 6) Evolución por generaciones ===
    for gen in range(maxGen):
        best = select_top_by_correlation_multi(population, data, topX, cm=cm, um=um)
        pc_global, pm_global = schedule_rates(gen, maxGen, pc_max=crossoverP, pm_max=mutP)

        parent_scores = [(ind, similarity_of_individual(ind, data, target_user, cm=cm, um=um)) for ind in best]
        parent_scores.sort(key=lambda t: t[1], reverse=True)
        f_best = parent_scores[0][1] if parent_scores else 0.0
        parents = [ind for ind, _ in parent_scores]

        T = temperature(gen, T0=1.0, shrink=0.10)
        new_pop = parents[:]

        while len(new_pop) < M:
            p1, p2 = rng.choice(parents), rng.choice(parents)
            f_p1 = similarity_of_individual(p1, data, target_user, cm=cm, um=um)

            pc_i, pm_i = indiv_rates(
                f_p1, f_best,
                pc_max=pc_global, pc_min=0.5*pc_global,
                pm_max=pm_global, pm_min=0.5*pm_global
            )

            child = one_point_crossover(p1, p2, rng) if rng.random() < pc_i else p1[:]
            child = mutate(child, candidates, pm_i, rng)

            f_child = similarity_of_individual(child, data, target_user, cm=cm, um=um)
            delta = f_p1 - f_child
            if accept_with_sa(delta, T, rng) and (child not in new_pop):
                new_pop.append(child)

        sims = [(ind, similarity_of_individual(ind, data, target_user, cm=cm, um=um)) for ind in new_pop]
        sims.sort(key=lambda t: t[1], reverse=True)
        population = [ind for ind, _ in sims]

    # === 7) Evaluar población final ===
    try:
        OMEGA = np.load(WEIGHTS_PATH)
    except Exception:
        OMEGA = (0.5, 0.3, 0.2)

    scored = [(ind, aggregate_individual(ind, target_user, data, OMEGA, cm=cm, um=um)) for ind in population]
    scored.sort(key=lambda t: t[1], reverse=True)
    best_ind, best_score = scored[0]

    if return_population:
        return data, best_ind, best_score, population
    return data, best_ind, best_score


if __name__ == "__main__":
    data, best_ind, score = run_bliga(target_user="1", use_small=True)

    def title(i): return (data.item_meta or {}).get(i, {}).get("title", i)
    print("Mejor individuo:", best_ind, "score:", round(score, 4))
    print("Top-N (títulos):", [title(i) for i in best_ind])

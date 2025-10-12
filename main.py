import random
from dataset_example import build_example_dataset
from population import generate_initial_population, select_top_by_correlation_multi
from similarity import similarity_of_individual
from operators import one_point_crossover, mutate
from prediction import predict_individual
from score_aggregator import aggregate_individual
from adaptive import schedule_rates  # tasas adaptativas
import numpy as np


def run_bliga(
    target_user: str,
    M: int = 6,
    N: int = 3,
    topX: float = 0.5,
    maxGen: int = 5,
    crossoverP: float = 0.9,
    mutP: float = 0.3,
    seed: int = 42
):
    rng = random.Random(seed)
    data = build_example_dataset()

    # 1️⃣ Población inicial
    population = generate_initial_population(data, target_user, M, N, rng)
    print("Población inicial:", population)

    for gen in range(maxGen):
        print(f"\n🧬 Generación {gen+1}/{maxGen}")

        # 2️⃣ Filtro 1: correlación semántica multi-feature
        best_corr = select_top_by_correlation_multi(population, data, topX)
        print("Top por correlación (multi-feature):", best_corr)

        # 3️⃣ Tasas adaptativas según generación (AGA)
        pc, pm = schedule_rates(gen, maxGen, pc_max=crossoverP, pm_max=mutP)
        print(f"Tasas adaptativas → pc={pc:.2f}, pm={pm:.2f}")

        # 4️⃣ Crossover + Mutación
        children = []
        for i in range(0, len(best_corr) - 1, 2):
            if rng.random() < pc:
                child = one_point_crossover(best_corr[i], best_corr[i+1], rng)
                child = mutate(child, list(data.item_categories.keys()), pm, rng)
                # evitar duplicados exactos
                if child not in best_corr:
                    children.append(child)
        population = best_corr + children
        print("Nueva población:", population)

        # 5️⃣ Filtro 2: similitud usuario-usuario (Pearson + Jaccard)
        sims = [(ind, similarity_of_individual(ind, data, target_user)) for ind in population]
        sims.sort(key=lambda t: t[1], reverse=True)
        population = [ind for ind, _ in sims]
        print("Ordenados por similitud:", sims)

    # 6️⃣ Filtro 3: predicción final
    USE_AGGREGATED = True  # usa la versión mejorada
    try:
        OMEGA = np.load("best_weights.npy")
    except FileNotFoundError:
        OMEGA = (0.5, 0.3, 0.2)

    if USE_AGGREGATED:
        scored = [(ind, aggregate_individual(ind, target_user, data, OMEGA)) for ind in population]
        print("\n🔮 Usando score agregado (ω):", OMEGA)
    else:
        scored = [(ind, predict_individual(ind, target_user, data)) for ind in population]
        print("\n🔮 Usando predicción Resnick clásica")

    # seleccionar mejor lista
    scored.sort(key=lambda t: t[1], reverse=True)
    best = scored[0]
    print("\n🏆 Mejor individuo final:", best)
    return best


if __name__ == "__main__":
    best_ind = run_bliga(target_user="u1")
    print("\nRecomendación final para u1:", best_ind)

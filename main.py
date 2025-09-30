import random
from dataset_example import build_example_dataset
from population import generate_initial_population, select_top_by_correlation
from similarity import similarity_of_individual
from operators import one_point_crossover, mutate
from prediction import predict_individual

def run_bliga(
    target_user: str,
    M: int = 6,
    N: int = 3,
    topX: float = 0.5,
    maxGen: int = 5,
    crossoverP: float = 0.7,
    mutP: float = 0.2,
    seed: int = 42
):
    rng = random.Random(seed)
    data = build_example_dataset()

    # 1. Población inicial
    population = generate_initial_population(data, target_user, M, N, rng)
    print("Población inicial:", population)

    for gen in range(maxGen):
        print(f"\nGeneración {gen+1}/{maxGen}")

        # 2. Filtrado por correlación
        best_corr = select_top_by_correlation(population, data, topX)
        print("Top por correlación:", best_corr)

        # 3. Crossover y mutación
        children = []
        for i in range(0, len(best_corr) - 1, 2):
            if rng.random() < crossoverP:
                child = one_point_crossover(best_corr[i], best_corr[i+1], rng)
                child = mutate(child, list(data.item_categories.keys()), mutP, rng)
                children.append(child)

        population = best_corr + children
        print("Nueva población:", population)

        # 4. Filtrado por similitud (opcional: se puede ordenar aquí también)
        sims = [(ind, similarity_of_individual(ind, data, target_user)) for ind in population]
        sims.sort(key=lambda t: t[1], reverse=True)
        population = [ind for ind, _ in sims]
        print("Ordenados por similitud:", sims)

    # 5. Predicción final
    scored = [(ind, predict_individual(ind, target_user, data)) for ind in population]
    scored.sort(key=lambda t: t[1], reverse=True)
    best = scored[0]
    print("\nMejor individuo final:", best)

    return best

if __name__ == "__main__":
    best_ind = run_bliga(target_user="u1")
    print("\nRecomendación final para u1:", best_ind)

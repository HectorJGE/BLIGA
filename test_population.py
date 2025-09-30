import random
from dataset_example import build_example_dataset
from population import generate_initial_population, select_top_by_correlation
from correlation import correlation_of_individual

if __name__ == "__main__":
    data = build_example_dataset()
    rng = random.Random(42)

    # Generar población inicial (M individuos de N ítems cada uno)
    pop = generate_initial_population(data, target_user="u1", M=5, N=3, rng=rng)
    print("Población inicial:", pop)

    # Calcular correlación de cada individuo
    for ind in pop:
        print(ind, "-> correlación =", correlation_of_individual(ind, data.item_categories))

    # Seleccionar top 40% por correlación
    best = select_top_by_correlation(pop, data, topX=0.4)
    print("Top 40% por correlación:", best)

import random
from dataset_example import build_example_dataset
from operators import one_point_crossover, mutate

if __name__ == "__main__":
    data = build_example_dataset()
    rng = random.Random(42)

    # Padres ficticios
    p1 = ["I1", "I2", "I3"]
    p2 = ["I4", "I5", "I6"]

    child = one_point_crossover(p1, p2, rng=rng)
    print("Padre1:", p1)
    print("Padre2:", p2)
    print("Hijo:", child)

    mutated = mutate(child, candidates=list(data.item_categories.keys()), mutP=0.5, rng=rng)
    print("Mutado:", mutated)

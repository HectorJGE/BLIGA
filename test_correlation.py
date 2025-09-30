from dataset_example import build_example_dataset
from correlation import jaccard_items, correlation_of_individual

if __name__ == "__main__":
    data = build_example_dataset()

    # Ejemplo: correlación entre I1 (acción, scifi) y I4 (aventura, scifi)
    corr_I1_I4 = jaccard_items(data.item_categories["I1"], data.item_categories["I4"])
    print("Corr(I1, I4):", corr_I1_I4)  # debería dar 1/3 ≈ 0.33

    # Ejemplo: correlación de un individuo [I1, I2, I4]
    individual = ["I1", "I2", "I4"]
    corr_ind = correlation_of_individual(individual, data.item_categories)
    print("Correlation([I1, I2, I4]):", corr_ind)
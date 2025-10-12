from dataset_example import build_example_dataset
from ga_weights import train_weights

# 1️⃣ Cargar dataset de ejemplo
data = build_example_dataset()

# 2️⃣ Crear una lista de (usuario, ítem, rating real)
samples = [(u, i, r) for u, ratings in data.ratings.items() for i, r in ratings.items()]

# 3️⃣ Entrenar GA de pesos
best_w = train_weights(data, samples, pop_size=12, gens=15)

print("\nPesos aprendidos ω:", best_w)

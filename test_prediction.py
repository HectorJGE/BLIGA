from dataset_example import build_example_dataset
from prediction import predict_rating, predict_individual

if __name__ == "__main__":
    data = build_example_dataset()
    target = "u1"

    # Predecir un ítem que u1 no calificó
    print("Predicción(u1, I3):", predict_rating(target, "I3", data))
    print("Predicción(u1, I5):", predict_rating(target, "I5", data))

    # Predecir para un individuo
    ind = ["I3", "I4", "I5"]
    print("Predicción(individuo, u1):", predict_individual(ind, target, data))


from dataset_example import build_example_dataset
from similarity import pearson_similarity, jaccard_users, user_similarity, similarity_of_individual

if __name__ == "__main__":
    data = build_example_dataset()

    # Ejemplo simple: similitud entre u1 y u2
    print("Pearson(u1,u2):", pearson_similarity("u1", "u2", data.ratings))
    print("Jaccard(u1,u2):", jaccard_users("u1", "u2", data.ratings))
    print("Sim(u1,u2):", user_similarity("u1", "u2", data.ratings))

    # Similarity de un individuo
    individual = ["I3", "I4", "I5"]  # ítems que u2 y u3 calificaron
    sim_ind = similarity_of_individual(individual, data, target_user="u1")
    print("Similarity([I3,I4,I5], u1):", sim_ind)

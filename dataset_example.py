from data import RecsData

def build_example_dataset() -> RecsData:
    """
    Devuelve un dataset pequeño de ejemplo con:
    - 3 usuarios
    - 6 ítems
    - Categorías de películas
    """
    ratings = {
        "u1": {"I1": 5, "I2": 4},       # Usuario 1 calificó 2 ítems
        "u2": {"I2": 3, "I3": 5},       # Usuario 2 calificó 2 ítems
        "u3": {"I1": 2, "I4": 4, "I5": 3},  # Usuario 3 calificó 3 ítems
    }

    item_categories = {
        "I1": {"acción", "ciencia_ficción"},
        "I2": {"acción"},
        "I3": {"romance"},
        "I4": {"aventura", "ciencia_ficción"},
        "I5": {"drama"},
        "I6": {"acción", "aventura"},
    }

    return RecsData(ratings=ratings, item_categories=item_categories)

if __name__ == "__main__":
    data = build_example_dataset()
    print("Todos los ítems:", data.items())
    print("Ítems no calificados por u1:", data.unrated_items_for("u1"))

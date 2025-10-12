from data import RecsData

def build_example_dataset() -> RecsData:
    ratings = {
        "u1": {"I1": 5, "I2": 4},
        "u2": {"I2": 3, "I3": 5},
        "u3": {"I1": 2, "I4": 4, "I5": 3},
    }

    item_categories = {
        "I1": {"acción", "ciencia_ficción"},
        "I2": {"acción"},
        "I3": {"romance"},
        "I4": {"aventura", "ciencia_ficción"},
        "I5": {"drama"},
        "I6": {"acción", "aventura"},
    }

    # === NUEVO: metadata sencilla por ítem ===
    item_meta = {
        "I1": {"director": "D1", "año": 2010, "actores": {"A1", "A2"}},
        "I2": {"director": "D1", "año": 2012, "actores": {"A2"}},
        "I3": {"director": "D2", "año": 2008, "actores": {"A3"}},
        "I4": {"director": "D3", "año": 2011, "actores": {"A2", "A4"}},
        "I5": {"director": "D4", "año": 2005, "actores": {"A5"}},
        "I6": {"director": "D1", "año": 2016, "actores": {"A6"}},
    }

    return RecsData(ratings=ratings, item_categories=item_categories, item_meta=item_meta)

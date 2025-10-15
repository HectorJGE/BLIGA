import pandas as pd, ast, re
from data import RecsData

def parse_list(x):
    try:
        data = ast.literal_eval(x)
        if isinstance(data, list):
            return {d["name"] for d in data if isinstance(d, dict) and "name" in d}
        return set()
    except:
        return set()

def extract_director(crew_str):
    try:
        crew = ast.literal_eval(crew_str)
        for c in crew:
            if isinstance(c, dict) and c.get("job") == "Director":
                return c.get("name")
    except:
        pass
    return None

def extract_year(date_str):
    if not isinstance(date_str, str):
        return None
    m = re.match(r"(\d{4})", date_str)
    return int(m.group(1)) if m else None

def load_tmdb_dataset(path="the-movies-dataset"):
    print("📂 Cargando dataset desde:", path)
    movies = pd.read_csv(f"{path}/movies_metadata.csv", low_memory=False)
    credits = pd.read_csv(f"{path}/credits.csv")
    ratings = pd.read_csv(f"{path}/ratings_small.csv")

    # merge por id
    credits["id"] = credits["id"].astype(str)
    movies["id"] = movies["id"].astype(str)
    movies = movies.merge(credits, on="id", how="left")

    item_categories = {}
    item_meta = {}

    for _, row in movies.iterrows():
        mid = str(row["id"])
        genres = parse_list(row.get("genres"))
        director = extract_director(row.get("crew"))
        cast = parse_list(row.get("cast"))
        year = extract_year(row.get("release_date"))
        pop_val = row.get("popularity")
        try:
            popularity = float(pop_val)
        except (ValueError, TypeError):
            popularity = 0.0

        item_categories[mid] = genres
        item_meta[mid] = {
            "director": director,
            "actores": cast,
            "año": year,
            "popularity": popularity,
        }

    ratings_dict = {}
    for u, i, r in ratings[["userId", "movieId", "rating"]].values:
        ratings_dict.setdefault(str(u), {})[str(i)] = float(r)

    print(f"✅ {len(ratings_dict)} usuarios, {len(item_categories)} ítems cargados.")
    return RecsData(ratings=ratings_dict,
                    item_categories=item_categories,
                    item_meta=item_meta)

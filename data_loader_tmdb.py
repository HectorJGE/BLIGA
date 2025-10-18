# data_loader_tmdb.py
import pandas as pd, re
from typing import Dict, Set, Any
from data import RecsData, ItemId, Category
from collections import defaultdict

# Helpers rápidos (sin ast.literal_eval ni iterrows sobre 45k filas)
def _year(s: str):
    if not isinstance(s, str): return None
    m = re.match(r"(\d{4})", s)
    return int(m.group(1)) if m else None

# extrae todos los valores de "name" tanto con ' ' como con " "
_NAME_RE = re.compile(r'''["']name["']\s*:\s*["']([^"']+)["']''')
def _names(s: str) -> Set[str]:
    if not isinstance(s, str): return set()
    return {m.group(1).strip() for m in _NAME_RE.finditer(s)}

# busca Director dentro de crew
_DIRECTOR_RE = re.compile(
    r''' \{ [^{}]* ["']job["']\s*:\s*["']Director["'] [^{}]* ["']name["']\s*:\s*["']([^"']+)["'] ''',
    re.X
)
def _director(s: str):
    if not isinstance(s, str): return None
    m = _DIRECTOR_RE.search(s)
    return m.group(1).strip() if m else None

def _build_indexes(data):
    users_by_item = defaultdict(set)
    for u, ur in data.ratings.items():
        for i in ur.keys():
            users_by_item[i].add(u)
    data.users_by_item = dict(users_by_item)
    data.item_pop = {i: len(us) for i, us in data.users_by_item.items()}
    data.max_item_pop = max(data.item_pop.values(), default=1)
    data.user_mean = {u: (sum(r.values())/len(r) if r else 0.0)
                        for u, r in data.ratings.items()}

def load_tmdb_dataset(path: str = "the-movies-dataset",
                    use_small: bool = True) -> RecsData:
    """
    Carga The Movies Dataset mapeando MovieLens→TMDB vía links(_small).csv.
    - Devuelve RecsData con claves de ítem = movieId (MovieLens)  ← importante
    - Filtra movies/credits SOLO a los títulos que aparecen en ratings
    - Usa extracción por regex (rápido) para géneros/actores/director
    """
    print("📂 Cargando dataset desde:", path)

    # 1) ratings + links (MovieLens)
    ratings_path = f"{path}/{'ratings_small.csv' if use_small else 'ratings.csv'}"
    links_path   = f"{path}/{'links_small.csv'   if use_small else 'links.csv'}"

    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    links   = pd.read_csv(links_path,   usecols=["movieId", "tmdbId"])

    ratings["movieId"] = ratings["movieId"].astype(str)
    links["movieId"]   = links["movieId"].astype(str)
    # quedarnos sólo con pelis que realmente aparecen en ratings y tienen tmdb válido
    links = links[links["movieId"].isin(ratings["movieId"].unique()) & links["tmdbId"].notna()]
    links["tmdbId"] = links["tmdbId"].astype("Int64").astype(str)
    tmdb_needed = set(links["tmdbId"].unique())

    # 2) movies_metadata y credits filtrados a esos tmdbId
    movies = pd.read_csv(
        f"{path}/movies_metadata.csv",
        usecols=["id", "genres", "release_date", "title", "original_title", "popularity"],
        dtype=str, low_memory=False
    )
    movies = movies[movies["id"].isin(tmdb_needed)].copy()
    movies["genres_set"] = movies["genres"].map(_names)
    movies["year"]       = movies["release_date"].map(_year)
    movies["popularity"] = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0.0)

    credits = pd.read_csv(f"{path}/credits.csv", usecols=["id", "cast", "crew"], dtype=str)
    credits = credits[credits["id"].isin(tmdb_needed)].copy()
    credits["actors_set"] = credits["cast"].map(_names)
    credits["director"]   = credits["crew"].map(_director)

    movies["genres_set"] = movies["genres_set"].astype(object)
    credits["actors_set"] = credits["actors_set"].astype(object)

    # 3) movieId -> tmdbId -> metadatos
    meta = links.merge(
        movies[["id", "genres_set", "year", "popularity", "title", "original_title"]],
        left_on="tmdbId", right_on="id", how="left"
    ).merge(
        credits[["id", "actors_set", "director"]],
        on="id", how="left"
    )

    # 4) construir RecsData (clave = movieId)  — versión robusta contra NaN
    item_categories: Dict[ItemId, Set[Category]] = {}
    item_meta: Dict[ItemId, Dict[str, Any]] = {}

    def _to_set(x):
        if isinstance(x, (set, frozenset)):
            return set(x)
        if isinstance(x, (list, tuple)):
            return set(x)
        # cualquier otro (NaN, None, float, str, etc.) -> set vacío
        return set()

    for row in meta.itertuples(index=False):
        mv = str(row.movieId)

        # nunca iterar sobre NaN
        gset   = _to_set(getattr(row, "genres_set", None))
        actors = _to_set(getattr(row, "actors_set", None))

        # año seguro (si viene NaN/None queda None)
        y = getattr(row, "year", None)
        year = int(y) if isinstance(y, (int,)) or (isinstance(y, str) and y.isdigit()) else None

        # popularidad segura
        pop_raw = getattr(row, "popularity", 0.0)
        try:
            pop = float(pop_raw) if pop_raw not in (None, "") else 0.0
        except Exception:
            pop = 0.0

        title = (getattr(row, "title", None) or getattr(row, "original_title", None) or mv)
        director = getattr(row, "director", None) if isinstance(getattr(row, "director", None), str) else None

        item_categories[mv] = gset
        item_meta[mv] = {
            "director": director,
            "actores": actors,
            "año": year,
            "popularity": pop,
            "title": title,
        }

    ratings_dict: Dict[str, Dict[str, float]] = {}
    for u, i, r in ratings.itertuples(index=False):
        ratings_dict.setdefault(str(u), {})[str(i)] = float(r)

    print(f"✅ users={len(ratings_dict)}  items={len(item_categories)}  (subset según {'ratings_small' if use_small else 'ratings'})")
    data = RecsData(
        ratings=ratings_dict,
        item_categories=item_categories,
        item_meta=item_meta
    )

    # 👉 construir índices/cachés aquí
    _build_indexes(data)

    return data

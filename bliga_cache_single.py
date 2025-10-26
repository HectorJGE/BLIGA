# bliga_cache_single.py
# -----------------------------------------------------------
# Cache persistente + modelo de contenido + user_means
# En un único archivo, listo para importar.
#
# Uso típico:
#   from bliga_cache_single import (
#       build_or_load_content_model,
#       build_or_load_user_means,
#       ContentModel
#   )
#   content_model = build_or_load_content_model(items, version="v1", rebuild=False)
#   user_means = build_or_load_user_means(ratings_df, version="v1", rebuild=False)
#
# Donde:
#   - items: List[dict] con metadatos (genres, keywords, cast_top3, director, companies, movieId)
#   - ratings_df: DataFrame con columnas ['userId','rating'] (userId enteros 0..U-1 o mapeados)
# -----------------------------------------------------------

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import os, json, pickle, hashlib, time
import numpy as np

# ================ Config de cache ================

CACHE_DIR = ".bliga_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_npz(path: str, **arrays) -> None:
    np.savez_compressed(path, **arrays)


def _load_npz(path: str):
    return np.load(path)


# ================ Modelo de Contenido ================

class ContentModel:
    """
    Contiene:
      - item_vecs: np.ndarray [N_items, V] TF-IDF L2-normalizado
      - vocab: dict token -> idx
      - idf: np.ndarray [V]
      - item_index: dict itemId -> fila
    """
    def __init__(self, item_vecs, vocab, idf, item_index):
        self.item_vecs = item_vecs
        self.vocab = vocab
        self.idf = idf
        self.item_index = item_index

    def item_vector(self, item_id: str) -> Optional[np.ndarray]:
        i = self.item_index.get(str(item_id))
        return self.item_vecs[i] if i is not None else None


def _tokenize(movie: Dict[str, Any]) -> List[str]:
    """
    Ajustá esta función según tus metadatos.
    """
    toks = []
    toks += [g.lower() for g in movie.get("genres", []) if g]
    toks += [k.lower() for k in movie.get("keywords", []) if k]
    toks += [c.lower() for c in movie.get("cast_top3", []) if c]
    if movie.get("director"):
        toks.append(str(movie["director"]).lower())
    toks += [co.lower() for co in movie.get("companies", []) if co]
    # Si querés sumar overview/plot tokenizado, agregalo acá.
    return [t for t in toks if t]


def _build_content_from_raw(items: List[Dict[str, Any]], id_key: str = "movieId"
                            ) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Construye TF-IDF simple con L2-normalización por ítem.
    """
    from collections import defaultdict
    from math import log

    docs: List[List[str]] = []
    for it in items:
        docs.append(_tokenize(it))

    # vocab & df
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
    N = len(docs)
    V = len(vocab)

    X = np.zeros((N, V), dtype=np.float32)
    for i, d in enumerate(docs):
        for t in d:
            j = vocab.get(t)
            if j is not None:
                X[i, j] += 1.0

    idf = np.zeros(V, dtype=np.float32)
    for t, j in vocab.items():
        idf[j] = log((N + 1) / (df[t] + 1)) + 1.0

    # TF-IDF
    X *= idf
    # L2-normalización por fila
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms

    # index de ítems
    item_index: Dict[str, int] = {}
    for i, it in enumerate(items):
        item_index[str(it[id_key])] = i

    return X, idf, vocab, item_index


# ================ Persistencia de ContentModel ================

def save_content_model(item_vecs, idf, vocab, item_index, version: str = "v1") -> None:
    _save_npz(os.path.join(CACHE_DIR, f"content_model_{version}.npz"),
              item_vecs=item_vecs, idf=(idf if idf is not None else np.array([])))
    with open(os.path.join(CACHE_DIR, f"content_index_{version}.pkl"), "wb") as f:
        pickle.dump({"vocab": vocab, "item_index": item_index}, f)


def try_load_content_model(version: str = "v1"):
    npz = os.path.join(CACHE_DIR, f"content_model_{version}.npz")
    pkl = os.path.join(CACHE_DIR, f"content_index_{version}.pkl")
    if os.path.exists(npz) and os.path.exists(pkl):
        data = _load_npz(npz)
        with open(pkl, "rb") as f:
            idx = pickle.load(f)
        item_vecs = data["item_vecs"]
        idf = data["idf"] if "idf" in data else None
        vocab = idx["vocab"]
        item_index = idx["item_index"]
        return item_vecs, idf, vocab, item_index
    return None


def build_or_load_content_model(items: List[Dict[str, Any]],
                                *,
                                version: str = "v1",
                                rebuild: bool = False,
                                id_key: str = "movieId") -> ContentModel:
    """
    Carga del cache si existe; si no, construye y persiste.
    """
    loaded = try_load_content_model(version)
    if loaded and not rebuild:
        item_vecs, idf, vocab, item_index = loaded
    else:
        item_vecs, idf, vocab, item_index = _build_content_from_raw(items, id_key=id_key)
        save_content_model(item_vecs, idf, vocab, item_index, version=version)
    # sanity checks mínimos
    # (opcional) assert np.allclose(np.linalg.norm(item_vecs, axis=1), 1, atol=1e-3)
    return ContentModel(item_vecs, vocab, idf, item_index)


# ================ User Means (Resnick) ================

def compute_user_means(ratings_df) -> np.ndarray:
    """
    ratings_df: pandas.DataFrame con columnas ['userId','rating'].
    - Si tus userId no son 0..U-1, mapealos antes de llamar aquí,
      o ajustá este método a tu mapping.
    """
    import pandas as pd  # import local para no requerir pandas si no se usa
    if not {"userId", "rating"}.issubset(ratings_df.columns):
        raise ValueError("ratings_df debe tener columnas ['userId','rating']")
    means = ratings_df.groupby('userId')['rating'].mean()
    U = int(ratings_df['userId'].max()) + 1
    vec = np.zeros(U, dtype=np.float32)
    vec[means.index.to_numpy()] = means.to_numpy(dtype=np.float32)
    return vec


def save_user_means(user_means: np.ndarray, version: str = "v1") -> None:
    np.save(os.path.join(CACHE_DIR, f"user_means_{version}.npy"), user_means)


def try_load_user_means(version: str = "v1"):
    path = os.path.join(CACHE_DIR, f"user_means_{version}.npy")
    return np.load(path) if os.path.exists(path) else None


def build_or_load_user_means(ratings_df,
                             *,
                             version: str = "v1",
                             rebuild: bool = False) -> np.ndarray:
    loaded = try_load_user_means(version)
    if loaded is None or rebuild:
        user_means = compute_user_means(ratings_df)
        save_user_means(user_means, version=version)
        return user_means
    return loaded


# ================ (Opcional) Meta/validación simple de cache ================

def write_meta(dataset_dir: str, *, version: str = "v1",
               ratings_csv: Optional[str] = None) -> None:
    """
    Guarda metadata útil para invalidar cache (por ejemplo, checksum de ratings.csv).
    """
    meta = {
        "version": version,
        "dataset_dir": dataset_dir,
        "ts": int(time.time()),
    }
    if ratings_csv and os.path.exists(ratings_csv):
        meta["ratings_sha1"] = _sha1(ratings_csv)
    with open(os.path.join(CACHE_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# ============================================================
# Load-only helpers: cargan desde el cache, sin reconstruir nada
# ============================================================

def load_content_model_only(*, version: str = "v1") -> ContentModel:
    """
    Carga el ContentModel desde el cache (.npz + .pkl) sin reconstruir.
    Lanza error si los archivos no existen.
    """
    loaded = try_load_content_model(version)
    if not loaded:
        raise FileNotFoundError(f"No existe cache de content_model para version={version} en .bliga_cache/")
    item_vecs, idf, vocab, item_index = loaded
    return ContentModel(item_vecs, vocab, idf, item_index)


def load_user_means_only(*, version: str = "v1"):
    """
    Carga el vector de medias de usuario desde el cache (.npy).
    Lanza error si el archivo no existe.
    """
    loaded = try_load_user_means(version)
    if loaded is None:
        raise FileNotFoundError(f"No existe cache de user_means para version={version} en .bliga_cache/")
    return loaded

from pathlib import Path
import random, numpy as np, time
from data_loader_tmdb import load_tmdb_dataset
from ga_weights import train_weights, evaluate_weights

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data = load_tmdb_dataset(path=str(root / "the-movies-dataset"), use_small=True)

    all_samples = [(u, i, r) for u, ur in data.ratings.items() for i, r in ur.items()]
    print(f"✅ users={len(data.ratings)}  items={len(data.item_categories)}  samples={len(all_samples)}")

    rng = random.Random(42)

    # ⏱️ BAJÁ el batch por ahora (probá 600; si sigue lento, 300)
    MAX_TRAIN = 600
    train_samples = rng.sample(all_samples, min(MAX_TRAIN, len(all_samples)))
    print(f"🧪 Entrenando ω con {len(train_samples)} muestras...")

    # Parámetros más livianos para verificar que corre
    POP, GENS = 8, 6

    t0 = time.time()
    best_w = train_weights(data, train_samples, pop_size=POP, gens=GENS, seed=42)
    print(f"\n[train_weights] Terminado en {time.time()-t0:.1f}s  ω={best_w}")

    # (opcional) mini holdout
    MAX_TEST = 500
    test_samples = rng.sample(all_samples, min(MAX_TEST, len(all_samples)))
    fit = evaluate_weights(best_w, data, test_samples)
    print(f"[train_weights] Fitness holdout: {fit:.4f}")

    np.save(root / "best_weights.npy", best_w)
    print("[train_weights] Guardado:", root / "best_weights.npy")
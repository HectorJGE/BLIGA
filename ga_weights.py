import random, math
from typing import List, Tuple, Optional
import numpy as np
from score_aggregator import score_cf, score_content, score_popularity

def rmse(preds, reals):
    """RMSE clásico sobre (predicción, real). Menor es mejor."""
    return math.sqrt(sum((p - r)**2 for p, r in zip(preds, reals)) / len(preds))

def evaluate_weights(omega, data, samples):
    """
    Fitness de un vector de pesos Ω = (w_cf, w_cont, w_pop).
    fitness = 1 / (1 + RMSE)  (mayor = mejor).
    """
    preds, reals = [], []
    for u, i, r_real in samples:
        s_cf = score_cf(u, i, data) / 5.0
        s_cont = score_content(u, i, data)
        s_pop = score_popularity(i, data)
        r_pred = omega[0]*s_cf + omega[1]*s_cont + omega[2]*s_pop
        preds.append(r_pred*5)   # volver a escala 0..5
        reals.append(r_real)
    return 1 / (1 + rmse(preds, reals))

def fitness_to_rmse(f: float) -> float:
    """Convierte fitness=1/(1+RMSE) a RMSE."""
    if f <= 0:
        return float("inf")
    return (1.0 / f) - 1.0

def train_weights(
    data,
    samples,
    pop_size: int = 10,
    gens: int = 20,
    seed: int = 42,
    *,
    rmse_tau: Optional[float] = None,   # ← umbral de parada por objetivo (RMSE ≤ τ)
    patience: Optional[int] = None,     # ← #gens sin mejora para cortar
    min_delta: float = 1e-4,            # ← mejora mínima en fitness para resetear paciencia
    verbose: bool = True
):
    """
    Entrena Ω con un GA simple con early stopping por objetivo (RMSE ≤ τ) y por no-mejora.

    - Población inicial aleatoria (simplex).
    - Selección por elitismo (top 50%).
    - Crossover 1 punto + mutación suave.
    - Fitness = 1/(1+RMSE).

    Paradas:
    - Si rmse_tau está definido y el mejor RMSE de la generación ≤ τ → stop.
    - Si patience está definido y no hay mejora > min_delta durante 'patience' generaciones → stop.
    """
    rng = random.Random(seed)

    # Población inicial: pesos aleatorios normalizados (Ω en el simplex)
    population = [normalize([rng.random(), rng.random(), rng.random()]) for _ in range(pop_size)]

    best_fitness = -float("inf")
    no_improve = 0

    for g in range(gens):
        # Evaluar población
        scored = [(evaluate_weights(w, data, samples), w) for w in population]
        scored.sort(reverse=True)
        f_best, w_best = scored[0]
        rmse_best = fitness_to_rmse(f_best)

        if verbose:
            print(f"Gen {g+1}/{gens}: best_fitness={f_best:.6f}  best_RMSE={rmse_best:.6f}  w={w_best}")

        # ---- Parada por objetivo (RMSE ≤ τ) ----
        if rmse_tau is not None and rmse_best <= rmse_tau:
            if verbose:
                print(f"[early-stop] RMSE objetivo alcanzado: {rmse_best:.6f} ≤ {rmse_tau:.6f}")
            population = [w for _, w in scored]  # mantener orden para selección final
            break

        # ---- Parada por no-mejora (paciencia) ----
        if f_best > best_fitness + min_delta:
            best_fitness = f_best
            no_improve = 0
        else:
            no_improve += 1
            if patience is not None and no_improve >= patience:
                if verbose:
                    print(f"[early-stop] Sin mejora > {min_delta} en {patience} generaciones.")
                population = [w for _, w in scored]
                break

        # Elitismo: padres = top 50%
        parents = [w for _, w in scored[:max(2, pop_size // 2)]]

        # Reproducción: generar hijos hasta recuperar pop_size
        children = []
        while len(parents) + len(children) < pop_size:
            a, b = rng.sample(parents, 2)
            child = crossover(a, b, rng)
            child = mutate(child, rng)
            children.append(child)

        population = parents + children

    # Selección final del mejor
    best = max(population, key=lambda w: evaluate_weights(w, data, samples))
    np.save("best_weights.npy", best)
    return best

def normalize(v):
    """Proyecta al simplex; si sum=0 (caso extremo), reparte uniforme."""
    s = sum(v)
    if s <= 0:
        n = len(v)
        return [1.0/n]*n
    return [x/s for x in v]

def crossover(a, b, rng):
    """
    Crossover de un punto sobre Ω usando 'rng' (reproducible).
    """
    cut = rng.randint(1, len(a)-1)
    return normalize(a[:cut] + b[cut:])

def mutate(w, rng, rate=0.2):
    """
    Mutación simple: perturba un gen en [-rate, +rate], clamp≥0 y renormaliza.
    """
    i = rng.randrange(len(w))
    w = list(w)
    w[i] += rng.uniform(-rate, rate)
    return normalize([max(0.0, x) for x in w])
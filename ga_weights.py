import random, math
from typing import List, Tuple
import numpy as np
from score_aggregator import score_cf, score_content, score_popularity

def rmse(preds, reals):
    return math.sqrt(sum((p - r)**2 for p, r in zip(preds, reals)) / len(preds))

def evaluate_weights(omega, data, samples):
    preds, reals = [], []
    for u, i, r_real in samples:
        s_cf = score_cf(u, i, data) / 5.0
        s_cont = score_content(u, i, data)
        s_pop = score_popularity(i, data)
        r_pred = omega[0]*s_cf + omega[1]*s_cont + omega[2]*s_pop
        preds.append(r_pred*5)
        reals.append(r_real)
    return 1 / (1 + rmse(preds, reals))  # fitness

def train_weights(data, samples, pop_size=10, gens=20, seed=42):
    rng = random.Random(seed)
    population = [normalize([rng.random(), rng.random(), rng.random()]) for _ in range(pop_size)]

    for g in range(gens):
        scored = [(evaluate_weights(w, data, samples), w) for w in population]
        scored.sort(reverse=True)
        best = scored[0]
        print(f"Gen {g+1}: best={best}")
        parents = [w for _, w in scored[:pop_size//2]]
        children = []
        for i in range(0, len(parents)-1, 2):
            a, b = parents[i], parents[i+1]
            child = crossover(a, b)
            child = mutate(child, rng)
            children.append(child)
        population = parents + children

    best = max(population, key=lambda w: evaluate_weights(w, data, samples))
    np.save("best_weights.npy", best)
    return best


def normalize(v):
    s = sum(v)
    return [x/s for x in v]

def crossover(a, b):
    cut = random.randint(1, len(a)-1)
    return normalize(a[:cut] + b[cut:])

def mutate(w, rng, rate=0.2):
    i = rng.randrange(len(w))
    w[i] += rng.uniform(-0.2, 0.2)
    return normalize([max(0, x) for x in w])

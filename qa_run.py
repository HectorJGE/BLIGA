from pathlib import Path
import numpy as np, json, itertools, math, argparse
import pandas as pd

from main import run_bliga, WEIGHTS_PATH
from score_aggregator import score_cf, score_content, score_popularity, aggregate_individual
from similarity import similarity_of_individual, user_similarity
from sim_item_multi import item_similarity_multi
from correlation_multi import correlation_of_individual_multi

# Cache manager
from bliga_cache_single import (
    build_or_load_content_model,
    build_or_load_user_means,
)

def title_of(i, data):
    return (getattr(data, "item_meta", None) or {}).get(i, {}).get("title", i)

def _ensure_cache_injected(data, *, cache_version="v1", rebuild=False):
    has_content = getattr(data, "content_model", None) is not None
    has_means = getattr(data, "user_means", None) is not None

    item_meta = getattr(data, "item_meta", None) or {}
    items_list = []
    for iid, meta in item_meta.items():
        row = dict(meta)
        row["movieId"] = iid
        items_list.append(row)

    if not has_content:
        content_model = build_or_load_content_model(
            items_list, version=cache_version, rebuild=rebuild
        )
        setattr(data, "content_model", content_model)

    if not has_means:
        rows = []
        for u, ur in data.ratings.items():
            for iid, r in ur.items():
                rows.append((int(u) if str(u).isdigit() else 0, float(r)))
        ratings_df = (
            pd.DataFrame(rows, columns=["userId", "rating"])
            if rows
            else pd.DataFrame({"userId": [], "rating": []})
        )

        user_means_vec = build_or_load_user_means(
            ratings_df, version=cache_version, rebuild=rebuild
        )
        means_map = ratings_df.groupby("userId")["rating"].mean().to_dict()
        setattr(data, "user_means", {str(int(k)): float(v) for k, v in means_map.items()})


def qa_report(data, target_user, best_ind, score, omega):
    cm = getattr(data, "content_model", None)
    um = getattr(data, "user_means", None)

    rows = []
    for i in best_ind:
        s_cf = score_cf(target_user, i, data, um=um) / 5.0
        s_cont = score_content(target_user, i, data, cm=cm)
        s_pop = score_popularity(i, data)
        total = omega[0]*s_cf + omega[1]*s_cont + omega[2]*s_pop
        rows.append((title_of(i, data), s_cf, s_cont, s_pop, total))
    print("\n[QA] Desglose por ítem:")
    for t, cf, cont, pop, tot in rows:
        print(f"  {t:<35} total={tot:.3f}  [cf={cf:.3f}, cont={cont:.3f}, pop={pop:.3f}]")
    print("  Σ score items =", round(sum(tot for *_, tot in rows), 4))

    pairs = list(itertools.combinations(best_ind, 2))
    sims = [item_similarity_multi(a, b, data) for a, b in pairs] if pairs else []
    coherence = correlation_of_individual_multi(best_ind, data, cm=cm, um=um)
    diversity = 1.0 - (sum(sims)/len(sims) if sims else 0.0)
    print(f"\n[QA] Coherencia multi = {coherence:.3f} | Diversidad = {diversity:.3f}")

    sim_total = similarity_of_individual(best_ind, data, target_user, cm=cm, um=um)
    contributors = []
    for u, ur in data.ratings.items():
        if u == target_user:
            continue
        if any(i in ur for i in best_ind):
            contributors.append((u, user_similarity(target_user, u, data.ratings)))
    contributors.sort(key=lambda t: t[1], reverse=True)
    print(f"\n[QA] Similaridad con el usuario = {sim_total:.3f}")
    print("  Top vecinos que influyen:", contributors[:5])

    pops = [score_popularity(i, data) for i in best_ind]
    pop_mean = sum(pops)/len(pops) if pops else 0.0
    novelty = sum(-math.log(max(1e-9, p)) for p in pops)/len(pops) if pops else 0.0
    print(f"\n[QA] Popularidad media = {pop_mean:.3f} | Novedad media (-log pop) = {novelty:.3f}")

    base = aggregate_individual(best_ind, target_user, data, omega, cm=cm, um=um)
    print("\n[QA] Impacto de quitar cada ítem (Δscore):")
    for i in best_ind:
        lo = [x for x in best_ind if x != i]
        delta = base - aggregate_individual(lo, target_user, data, omega, cm=cm, um=um)
        print(f"  - {title_of(i, data)}: Δ={delta:.3f}")

    report = {
        "user": target_user,
        "omega": tuple(float(x) for x in omega),
        "best_ind": best_ind,
        "titles": [title_of(i, data) for i in best_ind],
        "score": float(score),
        "coherence": float(coherence),
        "diversity": float(diversity),
        "pop_mean": float(pop_mean),
        "novelty": float(novelty)
    }
    with open("run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[QA] Guardado run_report.json")


def ablation_on_population(data, target_user, population, omega_list):
    cm = getattr(data, "content_model", None)
    um = getattr(data, "user_means", None)
    print("\n[QA] Ablation por Ω (re-rank de la población final):")
    for W in omega_list:
        scored = [
            (ind, aggregate_individual(ind, target_user, data, W, cm=cm, um=um))
            for ind in population
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        top_ind, top_score = scored[0]
        print(f"  Ω={W}  score={top_score:.4f}  títulos={[title_of(i, data) for i in top_ind[:5]]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="1")
    parser.add_argument("--small", dest="small", action="store_true", help="usar subset pequeño")
    parser.add_argument("--full", dest="small", action="store_false", help="usar dataset completo")
    parser.set_defaults(small=True)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--gens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="ignora caches y reconstruye")
    parser.add_argument("--cache_version", default="v1")
    args = parser.parse_args()

    print(">>> Iniciando QA-RUN <<<")

    data, best_ind, score, population = run_bliga(
        target_user=args.user,
        use_small=args.small,
        M=args.pop,
        N=args.n,
        maxGen=args.gens,
        seed=args.seed,
        return_population=True
    )

    _ensure_cache_injected(data, cache_version=args.cache_version, rebuild=args.rebuild)

    try:
        OMEGA = np.load(WEIGHTS_PATH)
    except Exception:
        OMEGA = (0.5, 0.3, 0.2)

    print("\n== QA principal ==")
    print("Mejor individuo:", best_ind, "score:", round(score, 4))
    print("Top-N (títulos):", [title_of(i, data) for i in best_ind])
    qa_report(data, args.user, best_ind, score, OMEGA)

    if args.ablation:
        ablation_on_population(
            data, args.user, population,
            omega_list=[(1,0,0), (0,1,0), (0,0,1), tuple(OMEGA)]
        )


if __name__ == "__main__":
    main()

# qa_run.py
from pathlib import Path
import numpy as np, json, itertools, math, argparse

from main import run_bliga, WEIGHTS_PATH
from score_aggregator import score_cf, score_content, score_popularity, aggregate_individual
from similarity import similarity_of_individual, user_similarity
from sim_item_multi import item_similarity_multi
from correlation_multi import correlation_of_individual_multi

def title_of(i, data):
    return (data.item_meta or {}).get(i, {}).get("title", i)

def qa_report(data, target_user, best_ind, score, omega):
    # 1) Desglose por ítem
    rows = []
    for i in best_ind:
        s_cf = score_cf(target_user, i, data) / 5.0
        s_cont = score_content(target_user, i, data)
        s_pop = score_popularity(i, data)
        rows.append((title_of(i, data), s_cf, s_cont, s_pop, omega[0]*s_cf + omega[1]*s_cont + omega[2]*s_pop))
    print("\n[QA] Desglose por ítem:")
    for t, cf, cont, pop, tot in rows:
        print(f"  {t:<35} total={tot:.3f}  [cf={cf:.3f}, cont={cont:.3f}, pop={pop:.3f}]")
    print("  Σ score items =", round(sum(tot for *_, tot in rows), 4))

    # 2) Coherencia y diversidad
    pairs = list(itertools.combinations(best_ind, 2))
    sims = [item_similarity_multi(a, b, data) for a, b in pairs] if pairs else []
    coherence = correlation_of_individual_multi(best_ind, data)
    diversity = 1.0 - (sum(sims)/len(sims) if sims else 0.0)
    print(f"\n[QA] Coherencia multi = {coherence:.3f} | Diversidad = {diversity:.3f} | sim_prom = {(1-diversity):.3f}")

    # 3) Alineación colaborativa
    sim_total = similarity_of_individual(best_ind, data, target_user)
    contributors = []
    for u, ur in data.ratings.items():
        if u == target_user: 
            continue
        if any(i in ur for i in best_ind):
            contributors.append((u, user_similarity(target_user, u, data.ratings)))
    contributors.sort(key=lambda t: t[1], reverse=True)
    print(f"\n[QA] Similaridad con el usuario = {sim_total:.3f}")
    print("  Top vecinos que influyen:", contributors[:5])

    # 4) Novedad (anti-popularidad)
    pops = [score_popularity(i, data) for i in best_ind]
    pop_mean = sum(pops)/len(pops) if pops else 0.0
    novelty = sum(-math.log(max(1e-9, p)) for p in pops)/len(pops) if pops else 0.0
    print(f"\n[QA] Popularidad media = {pop_mean:.3f} | Novedad media (-log pop) = {novelty:.3f}")

    # 5) Sensibilidad por ítem
    base = aggregate_individual(best_ind, target_user, data, omega)
    print("\n[QA] Impacto de quitar cada ítem (Δscore):")
    for i in best_ind:
        lo = [x for x in best_ind if x != i]
        delta = base - aggregate_individual(lo, target_user, data, omega)
        print(f"  - {title_of(i, data)}: Δ={delta:.3f}")

    # 6) Guardar reporte
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
    print("\n[QA] Ablation por Ω (re-rank de la población final):")
    for W in omega_list:
        scored = [(ind, aggregate_individual(ind, target_user, data, W)) for ind in population]
        scored.sort(key=lambda t: t[1], reverse=True)
        top_ind, top_score = scored[0]
        print(f"  Ω={W}  score={top_score:.4f}  títulos={[title_of(i, data) for i in top_ind[:5]]}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="1")
    parser.add_argument("--small", dest="small", action="store_true", help="usar subset pequeño")
    parser.add_argument("--full",  dest="small", action="store_false", help="usar dataset completo")
    parser.set_defaults(small=True)
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--gens", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    # Corre BLIGA y pide que nos devuelva la población final
    data, best_ind, score, population = run_bliga(
        target_user=args.user,
        use_small=args.small,
        M=args.pop,
        N=args.n,
        maxGen=args.gens,
        seed=args.seed,
        return_population=True
    )

    # Carga Ω
    try:
        OMEGA = np.load(WEIGHTS_PATH)
    except Exception:
        OMEGA = (0.5, 0.3, 0.2)

    # QA principal
    print("\n== QA principal ==")
    print("Mejor individuo:", best_ind, "score:", round(score, 4))
    print("Top-N (títulos):", [title_of(i, data) for i in best_ind])
    qa_report(data, args.user, best_ind, score, OMEGA)

    # Ablation opcional
    if args.ablation:
        ablation_on_population(
            data, args.user, population,
            omega_list=[(1,0,0), (0,1,0), (0,0,1), tuple(OMEGA)]
        )

if __name__ == "__main__":
    main()

def schedule_rates(gen: int, max_gen: int,
                    pc_max=0.9, pc_min=0.5,
                    pm_max=0.3, pm_min=0.05) -> tuple[float, float]:
    """
    Calcula las tasas de cruce (pc) y mutación (pm) para la generación actual,
    usando un "enfriamiento" lineal (annealing lineal) desde un valor máximo
    al iniciar hasta un mínimo al final.

    Parámetros
    ----------
    gen : int
        Índice de la generación actual (típicamente 0 .. max_gen-1).
    max_gen : int
        Número total de generaciones del GA.
    pc_max, pc_min : float
        Cota superior e inferior para la probabilidad de cruce.
    pm_max, pm_min : float
        Cota superior e inferior para la probabilidad de mutación.

    Retorna
    -------
    (pc, pm) : tuple[float, float]
        Probabilidades para la generación actual.

    Intuición
    ---------
    - Al principio (gen=0):  pc≈pc_max  y  pm≈pm_max  → más exploración.
    - Al final   (gen=max_gen-1): pc≈pc_min y pm≈pm_min → más explotación.
    """

    # Progreso normalizado en [0,1]. El max() evita división por cero
    # cuando max_gen == 1 (un solo paso ⇒ t=0).
    t = gen / max(1, max_gen-1)

    # Interpolación lineal: valor = max - (max - min) * t
    pc = pc_max - (pc_max - pc_min) * t
    pm = pm_max - (pm_max - pm_min) * t
    return pc, pm


# ==== AGA: tasas adaptadas al fitness del individuo ====

def indiv_rates(f_i: float, f_best: float,
                pc_max=0.9, pc_min=0.5,
                pm_max=0.3, pm_min=0.05) -> tuple[float, float]:
    """
    Ajusta pc/pm para un individuo según su fitness relativo.
    Idea (Hassan): peores ⇒ más mutación (exploración), mejores ⇒ menos.
    """
    if f_best <= 0:
        rel = 1.0  # evita div/0; considera todos "lejos"
    else:
        # Mayor rel ⇒ más lejos del mejor
        rel = max(0.0, min(1.0, 1.0 - (f_i / f_best)))

    # Interpola entre min..max proporcional a 'rel'
    pc_i = pc_min + (pc_max - pc_min) * rel
    pm_i = pm_min + (pm_max - pm_min) * rel
    return pc_i, pm_i


# ==== MGA/SA: temperatura (cooling) y aceptación probabilística ====

def temperature(gen: int, T0: float = 1.0, shrink: float = 0.10) -> float:
    """
    Enfriamiento geométrico: T_g = T0 * (1 - shrink)^gen
    shrink típico: 0.05..0.15
    """
    shrink = max(0.0, min(0.5, shrink))
    return T0 * ((1.0 - shrink) ** max(0, gen))


def accept_with_sa(delta: float, T: float, rng) -> bool:
    """
    Acepta solución peor con prob ~ exp(-delta/T).
    delta = (score_parent - score_child). Si child >= parent → aceptar siempre.
    """
    if delta <= 0:
        return True
    if T <= 1e-12:
        return False
    import math
    p = math.exp(-delta / T)
    return rng.random() < p
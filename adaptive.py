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

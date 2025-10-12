def schedule_rates(gen: int, max_gen: int,
                   pc_max=0.9, pc_min=0.5,
                   pm_max=0.3, pm_min=0.05) -> tuple[float, float]:
    """Devuelve (pc, pm) para esta generación."""
    t = gen / max(1, max_gen-1)
    pc = pc_max - (pc_max - pc_min) * t
    pm = pm_max - (pm_max - pm_min) * t
    return pc, pm

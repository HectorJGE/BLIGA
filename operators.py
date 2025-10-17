
import random
from typing import List
from data import ItemId

def tournament_selection(population: List[List[ItemId]], k: int, rng: random.Random = random) -> List[ItemId]:
    """
    Selecciona un individuo vía torneo de tamaño k.

    IMPORTANTE: Esta versión asume que 'population' YA está ordenada por fitness
    (mejor → peor) antes de llamar a la función. Por eso retorna simplemente
    el "mejor" entre los k candidatos tomando 'candidates[0]'.

    Parámetros
    ----------
    population : List[List[ItemId]]
        Población completa (se asume ya ordenada por fitness desc).
    k : int
        Tamaño del torneo (k <= len(population)).
    rng : random.Random
        Generador aleatorio (inyectable para reproducibilidad).

    Return
    ------
    List[ItemId]
        El individuo ganador del torneo.

    Nota:
    - En un torneo "clásico" se evaluaría el fitness de los k candidatos y
        se elegiría el mejor. Aquí se evita reevaluar asumiendo orden global previo.
    """

    # Muestra sin reemplazo de k individuos para el torneo
    candidates = rng.sample(population, k)
    
    # Elegimos el primero suponiendo que 'population' ya estaba ordenada por fitness,
    # y por ende 'candidates[0]' es el mejor entre los elegidos.
    return candidates[0]

def one_point_crossover(parent1: List[ItemId], parent2: List[ItemId], rng: random.Random = random) -> List[ItemId]:
    """
    Cruce de un punto entre dos padres, manteniendo longitud y SIN duplicados.

    Estrategia:
    - Cortar ambos padres en un punto aleatorio (1..n-1).
    - Tomar el prefijo de parent1.
    - Completar con el sufijo de parent2, excluyendo ítems ya presentes en el prefijo.
    - Si aún faltan genes para llegar a n, completar con elementos de parent2
        que aún no estén en el hijo.

    Restricciones:
    - Ambos padres deben tener la MISMA longitud (n).
    - El hijo no tendrá ítems repetidos (conjunto único).

    Nota:
    - El "relleno" final toma genes solo de parent2. Esto favorece herencia de parent2
        cuando hay conflictos; es una decisión de diseño válida.
    """
    if len(parent1) != len(parent2):
        raise ValueError("Los padres deben tener el mismo tamaño")
    
    n = len(parent1)

    # Punto de corte en [1, n-1] para asegurar prefijo y sufijo no vacíos
    point = rng.randint(1, n - 1)

    # Prefijo del padre 1 + sufijo filtrado del padre 2 (evita duplicados del prefijo)
    child = parent1[:point] + [x for x in parent2[point:] if x not in parent1[:point]]
    
    # Si faltan genes (por exclusión de duplicados), completar con elementos de parent2
    # que aún no estén en 'child'. Mantiene longitud fija y unicidad.
    while len(child) < n:
        candidate = rng.choice(parent2)
        if candidate not in child:
            child.append(candidate)
    return child

def mutate(individual: List[ItemId], candidates: List[ItemId], mutP: float, rng: random.Random = random) -> List[ItemId]:
    """
    Mutación por reemplazo: recorre cada posición y, con probabilidad mutP,
    sustituye el ítem por otro válido tomado de 'candidates' (sin duplicar dentro del individuo).

    Parámetros
    ----------
    individual : List[ItemId]
        Individuo a mutar (lista de ítems únicos).
    candidates : List[ItemId]
        Universo de ítems válidos para reemplazo. Debe contener suficientes
        opciones para evitar estancarse buscando un ítem no repetido.
    mutP : float
        Probabilidad de mutación por gen (0..1).
    rng : random.Random
        Generador aleatorio.

    Return
    ------
    List[ItemId]
        Nuevo individuo mutado (misma longitud, sin duplicados).

    Notas:
    - Aplica mutación *por gen*, es decir, puede mutar 0..N posiciones según mutP.
    - Garantiza unicidad: si el candidato ya está en el individuo, sigue buscando otro.
    - Asegurate de que 'candidates' incluya ítems que NO estén todos ya en 'individual';
        de lo contrario, podría tardar en encontrar reemplazos válidos.
    """
    new_ind = individual[:]
    for i in range(len(new_ind)):
        if rng.random() < mutP:
            # Elegir un nuevo gen que no esté en el individuo (mantener unicidad)
            new_gene = rng.choice(candidates)
            while new_gene in new_ind:
                new_gene = rng.choice(candidates)
            new_ind[i] = new_gene
    return new_ind

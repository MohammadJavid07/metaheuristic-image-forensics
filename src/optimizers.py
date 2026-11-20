import random
import numpy as np
from deap import base, creator, tools, algorithms


def evaluate_fitness(
    individual,
    forensic_scores,
    semantic_scores,
    segments,
    region_ids=None,
    alpha=1.0,
    beta=1.0,
    lambda_reg=1.5,
):
    """
    Maximizes mean forensic and semantic signal strength of selected candidate regions
    while penalizing over-selection via parsimony ratio (k / N).
    """
    n_genes = len(individual)
    if n_genes == 0:
        return (-10.0,)

    selected_rids = []
    for idx, val in enumerate(individual):
        if int(val) == 1:
            rid = region_ids[idx] if (region_ids is not None and idx < len(region_ids)) else idx
            selected_rids.append(rid)

    if len(selected_rids) == 0:
        return (-10.0,)

    f_vals = [float(forensic_scores.get(rid, 0.0)) for rid in selected_rids]
    s_vals = [float(semantic_scores.get(rid, 0.0)) for rid in selected_rids]

    f_mean = sum(f_vals) / len(f_vals)
    s_mean = sum(s_vals) / len(s_vals)
    parsimony_ratio = len(selected_rids) / float(n_genes)

    fitness = (alpha * f_mean) + (beta * s_mean) - (lambda_reg * parsimony_ratio)
    return (float(fitness),)


def run_binary_genetic_algorithm(
    forensic_scores,
    semantic_scores,
    segments,
    n_regions=21,
    n_gens=30,
    pop_size=80,
    lambda_reg=1.5,
    random_seed=42,
    region_ids=None,
):
    """
    Sets up and runs the DEAP Binary Genetic Algorithm with reproducibility seed.
    """
    if n_regions <= 0:
        return []

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_regions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

    toolbox.register(
        "evaluate",
        evaluate_fitness,
        forensic_scores=forensic_scores,
        semantic_scores=semantic_scores,
        segments=segments,
        region_ids=region_ids,
        lambda_reg=lambda_reg,
    )

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)

    print(f"[Info] Running BGA optimization (lambda_reg={lambda_reg}, seed={random_seed}) for {n_gens} generations...")
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    algorithms.eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=n_gens,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    best_individual = list(hof[0])
    best_score = evaluate_fitness(
        best_individual,
        forensic_scores,
        semantic_scores,
        segments,
        region_ids=region_ids,
        lambda_reg=lambda_reg,
    )[0]
    print(f"[Success] BGA optimization complete! Best fitness score: {best_score:.4f}")

    return best_individual


def run_binary_pso(
    forensic_scores,
    semantic_scores,
    segments,
    n_regions=21,
    n_gens=30,
    pop_size=80,
    lambda_reg=1.5,
    random_seed=42,
    region_ids=None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max: float = 4.0,
):
    """
    Runs Binary Particle Swarm Optimization (BPSO) using a sigmoid transfer function
    under the same evaluation budget (pop_size * n_gens) and objective as BGA.
    """
    if n_regions <= 0:
        return []

    rng = np.random.default_rng(random_seed)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    positions = rng.integers(0, 2, size=(pop_size, n_regions)).astype(np.int32)
    velocities = rng.uniform(-1.0, 1.0, size=(pop_size, n_regions)).astype(np.float64)

    pbest_pos = positions.copy()
    pbest_fit = np.full(pop_size, -np.inf, dtype=np.float64)

    for i in range(pop_size):
        fit = evaluate_fitness(
            pbest_pos[i].tolist(),
            forensic_scores,
            semantic_scores,
            segments,
            region_ids=region_ids,
            lambda_reg=lambda_reg,
        )[0]
        pbest_fit[i] = fit

    gbest_idx = int(np.argmax(pbest_fit))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    print(f"[Info] Running BPSO optimization (lambda_reg={lambda_reg}, seed={random_seed}) for {n_gens} iterations...")

    for _ in range(n_gens):
        r1 = rng.random((pop_size, n_regions))
        r2 = rng.random((pop_size, n_regions))

        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (gbest_pos - positions)
        )
        velocities = np.clip(velocities, -v_max, v_max)

        # Sigmoid transfer function: V(v_ij) = 1 / (1 + exp(-v_ij))
        probs = 1.0 / (1.0 + np.exp(-velocities))
        positions = (rng.random((pop_size, n_regions)) < probs).astype(np.int32)

        for i in range(pop_size):
            fit = evaluate_fitness(
                positions[i].tolist(),
                forensic_scores,
                semantic_scores,
                segments,
                region_ids=region_ids,
                lambda_reg=lambda_reg,
            )[0]
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = positions[i].copy()
                if fit > gbest_fit:
                    gbest_fit = fit
                    gbest_pos = positions[i].copy()

    print(f"[Success] BPSO optimization complete! Best fitness score: {gbest_fit:.4f}")
    return gbest_pos.tolist()
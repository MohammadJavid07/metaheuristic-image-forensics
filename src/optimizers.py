import random
import numpy as np
from deap import base, creator, tools, algorithms

def evaluate_fitness(individual, forensic_scores, semantic_scores, segments, alpha=1.0, beta=1.0, lambda_reg=1.5):
    """
    Clean, balanced objective function:
    Maximizes forensic & semantic feature strength while strictly penalizing over-selection.
    """
    selected_indices = [i for i, val in enumerate(individual) if val == 1]
    N = len(individual)
    
    if len(selected_indices) == 0:
        return (-10.0,) # Heavy penalty for empty selection

    # 1. Mean Forensic Score of selected regions
    f_vals = [forensic_scores.get(rid, 0.0) for rid in selected_indices]
    f_mean = sum(f_vals) / len(f_vals)
    
    # 2. Mean Semantic Score of selected regions
    s_vals = [semantic_scores.get(rid, 0.0) for rid in selected_indices]
    s_mean = sum(s_vals) / len(s_vals)
    
    # 3. Parsimony ratio: fraction of the image selected (0.0 to 1.0)
    parsimony_ratio = len(selected_indices) / float(N)
    
    # Joint objective: reward signal quality, heavily penalize selecting too much of the image
    fitness = (alpha * f_mean) + (beta * s_mean) - (lambda_reg * parsimony_ratio)
    
    return (fitness,)

def run_binary_genetic_algorithm(forensic_scores, semantic_scores, segments, n_regions=21, n_gens=30, pop_size=80, lambda_reg=1.5, random_seed=42):
    """
    Sets up and runs the DEAP Binary Genetic Algorithm with reproducibility seed.
    """
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

    toolbox.register("evaluate", evaluate_fitness, 
                     forensic_scores=forensic_scores, 
                     semantic_scores=semantic_scores, 
                     segments=segments, 
                     lambda_reg=lambda_reg)
                     
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    
    print(f"[Info] Running BGA optimization (lambda_reg={lambda_reg}, seed={random_seed}) for {n_gens} generations...")
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gens, stats=stats, halloffame=hof, verbose=False)
    
    best_individual = hof[0]
    print(f"[Success] BGA optimization complete! Best fitness score: {best_individual.fitness.values[0]:.4f}")
    
    return best_individual

if __name__ == "__main__":
    from load_data import load_tgif2_sample
    from region_proposals import extract_superpixel_candidates
    from feature_extractor import compute_dual_signal_features
    
    img_path = "data/tgif2_sample/images/sample_01.jpg"
    mask_path = "data/tgif2_sample/masks/sample_01_mask.png"
    image, _ = load_tgif2_sample(img_path, mask_path)
    
    if image is not None:
        segments, region_ids = extract_superpixel_candidates(image, n_segments=50)
        f_scores, s_scores = compute_dual_signal_features(image, segments, region_ids)
        
        print("\nTesting Optimizer Execution...")
        best_mask = run_binary_genetic_algorithm(f_scores, s_scores, segments, n_regions=len(region_ids), lambda_reg=1.5, random_seed=42)
        print(f"Best solution vector (sample): {best_mask[:10]}...")
import random
import numpy as np
from deap import base, creator, tools, algorithms

def evaluate_fitness(individual, forensic_scores, semantic_scores, segments, alpha=1.0, beta=1.0, gamma=0.5, lambda_reg=0.2):
    """
    Joint objective function balancing forensic scores, semantic scores,
    spatial coherence, and parsimony penalty.
    """
    selected_indices = [i for i, val in enumerate(individual) if val == 1]
    N = len(individual)
    
    if len(selected_indices) == 0:
        return (-10.0,), # Heavy penalty for empty selection

    # 1. Forensic alignment score
    f_sum = sum(forensic_scores.get(rid, 0.0) for rid in selected_indices)
    
    # 2. Semantic alignment score
    s_sum = sum(semantic_scores.get(rid, 0.0) for rid in selected_indices)
    
    # 3. Spatial coherence (simplified adjacency check)
    coherence = 0.0
    for i in range(len(selected_indices)):
        for j in range(i + 1, len(selected_indices)):
            # If two selected regions are adjacent or close, reward coherence
            coherence += 1.0 # placeholder for boundary adjacency
    
    # 4. Parsimony penalty (penalize selecting too many regions)
    parsimony = len(selected_indices) / float(N)
    
    # Joint objective calculation
    fitness = (alpha * f_sum) + (beta * s_sum) + (gamma * coherence / (len(selected_indices) + 1)) - (lambda_reg * parsimony)
    
    return (fitness,)

def run_binary_genetic_algorithm(forensic_scores, semantic_scores, segments, n_regions=21, n_gens=20, pop_size=50):
    """
    Sets up and runs the DEAP Binary Genetic Algorithm.
    """
    # Clear DEAP classes if already declared
    if "FitnessMax" in creator.__dict__:
        del creator.FitnessMax
    if "Individual" in creator.__dict__:
        del creator.Individual

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    
    # Attribute generator: binary 0 or 1
    toolbox.register("attr_bool", random.randint, 0, 1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_regions)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop_size)

    # Register operators
    toolbox.register("evaluate", evaluate_fitness, forensic_scores=forensic_scores, semantic_scores=semantic_scores, segments=segments)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Run simple genetic algorithm
    print(f"[Info] Running BGA optimization for {n_gens} generations...")
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
        segments, region_ids = extract_superpixel_candidates(image, n_segments=21)
        f_scores, s_scores = compute_dual_signal_features(image, segments, region_ids)
        
        print("\nTesting Optimizer Execution...")
        best_mask = run_binary_genetic_algorithm(f_scores, s_scores, segments, n_regions=len(region_ids))
        print(f"Best solution vector (sample): {best_mask[:10]}...")
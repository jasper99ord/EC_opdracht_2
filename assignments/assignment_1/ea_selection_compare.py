# Standard library
import json
import random
from pathlib import Path

# Third-party libraries
import networkx as nx
import numpy as np
import torch

# Local scripts
from tree_edit_distance import mean_plus_std_tree_edit_distance

# Local libraries (ARIEL)
from ariel import console
from ariel.body_phenotypes.robogen_lite.decoders._blueprint import (load_graph_from_json)
from ariel.ec import Individual, Population
from ariel.ec.genotypes.tree.operators import (
    mutate_hoist,
    mutate_replace_node,
    mutate_shrink,
    mutate_subtree_replacement,
    random_tree,
)
from ariel.ec.genotypes.tree.tree_genome import TreeGenome

# --- DATA SETUP --- #
SCRIPT_NAME = Path(__file__).stem
HERE = Path(__file__).parent
REPO_ROOT = HERE.parent.parent
DATA = REPO_ROOT / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# --- EXPERIMENT CONSTANTS --- #
TARGET_DIR: Path = HERE / "target_bodies" 
NUM_OF_MODULES: int = 20  
MU: int = 50  
NUM_GENS: int = 100
NUM_SEEDS: int = 5
TOURNAMENT_K: int = 3  
MUTATION_TYPES = ["point", "subtree", "shrink", "hoist"]
MUTATION_PROBS = [0.4, 0.4, 0.1, 0.1]


# ── Individual factory ────────────────────────────────────────────────────────
def make_individual() -> Individual:
    while True:
        genome = random_tree(NUM_OF_MODULES)
        if len(genome.nodes) > 0:
            break
    ind = Individual()
    ind.genotype = genome.to_dict()
    return ind


def tournament_select(parents: list, k: int = TOURNAMENT_K) -> Individual:
    """Pick k random parents, return the one with the lowest fitness."""
    contestants = random.sample(parents, min(k, len(parents)))
    return min(contestants, key=lambda ind: ind.fitness)


def make_offspring(
    parents: list,
    mu: int,
    rng: np.random.Generator,
) -> list:
    """Create mu offspring: tournament parent selection + one mutation each.

    Parent selection is fitness-based in BOTH variants; the variants differ
    only in `survivor_selection`.
    """
    offspring = []
    for _ in range(mu):
        parent = tournament_select(parents)
        genome = TreeGenome.from_dict(parent.genotype)
        mutation_type = rng.choice(MUTATION_TYPES, p=MUTATION_PROBS)

        if mutation_type == "point":
            # Point mutation: change node type/rotation
            mutate_replace_node(genome)
        elif mutation_type == "subtree":
            # Subtree mutation: replace subtree with new random tree
            mutate_subtree_replacement(genome, max_modules=NUM_OF_MODULES)
        elif mutation_type == "shrink":
            # Shrink mutation: replace node+subtree with single leaf
            mutate_shrink(genome)
        elif mutation_type == "hoist":
            # Hoist mutation: promote child to replace parent
            mutate_hoist(genome)

        ind = Individual()
        ind.genotype = genome.to_dict()
        offspring.append(ind)

    return offspring


#  1. THE TARGET BODIES
# ============================================================================ #
def load_targets(target_dir: Path = TARGET_DIR) -> list[nx.DiGraph]:
    """Load every target body graph from a directory.

    Returns
    -------
    list of nx.DiGraph
        One graph per JSON file, sorted by filename.

    Raises
    ------
    FileNotFoundError
        If the directory holds no target JSON files.
    """
    paths = sorted(target_dir.glob("*.json"))
    if not paths:
        msg = f"no target bodies found in {target_dir}"
        raise FileNotFoundError(msg)
    return [load_graph_from_json(p) for p in paths]


#  2. FITNESS
# ============================================================================ #
def fitness_function(
    body: nx.DiGraph,
    targets: list[nx.DiGraph],
) -> float:
    """Score one body against the whole target set. LOWER IS BETTER.

    Some things worth thinking about:
      * The std term charges for unevenness - body that is mediocre against every target
        and one that is excellent on most but bad on one can still land close
        in fitness, but the latter is penalized a bit more.
      * Nothing here rewards small bodies. Does your EA bloat? Should a size
        penalty be part of fitness, or is that the encoding's job?
    """
    return mean_plus_std_tree_edit_distance(body, targets)


def evaluate(population: Population, targets: list[nx.DiGraph]) -> Population:
    """Evaluate every not-yet-evaluated individual (A1 tree edit fitness)."""
    for ind in population:
        if not (ind.alive and ind.requires_eval):
            continue
        body = TreeGenome.from_dict(ind.genotype).to_networkx()
        ind.fitness = fitness_function(body, targets)
    return population


#  3. SELECTION (the aspect under investigation)
# ============================================================================ #
def survivor_selection(
    parents: list,
    offspring: list,
    mode: str,
    mu: int,
) -> list:
    """Return the next population of size mu.

    mode == "fitness": (mu + lambda) - rank parents + offspring, keep best mu.
    mode == "age":     (mu, lambda) with lambda == mu - all parents die,
                       offspring replace them regardless of fitness.
    """
    if mode == "fitness":
        po = parents + offspring
        po.sort(key=lambda ind: ind.fitness)
        return po[:mu]

    if mode == "age":
        if len(offspring) != mu:
            msg = f"offspring length {len(offspring)} must equal mu={mu}"
            raise ValueError(msg)
        return offspring

    msg = f"unknown mode: {mode}"
    raise ValueError(msg)


#  4. LOGGING
# ============================================================================ #
def new_log() -> dict[str, list[float]]:
    return {"best": [], "mean": [], "std": [], "best_so_far": []}


def log_generation(log: dict[str, list[float]], population: list) -> None:
    """Append best / mean / std of the current population and best-so-far."""
    fits = np.array([ind.fitness for ind in population], dtype=float)
    best = float(fits.min())
    log["best"].append(best)
    log["mean"].append(float(fits.mean()))
    log["std"].append(float(fits.std()))
    prev = log["best_so_far"][-1] if log["best_so_far"] else float("inf")
    log["best_so_far"].append(min(prev, best))


def save_log(log: dict, method: str, seed: int, mu: int, num_gens: int) -> None:
    log["meta"] = {
        "method": method,
        "seed": seed,
        "mu": mu,
        "num_gens": num_gens,
        "evaluations": mu * (num_gens + 1),
        "tournament_k": TOURNAMENT_K,
        "mutation_types": MUTATION_TYPES,
        "mutation_probs": MUTATION_PROBS,
    }
    out = DATA / f"curve_{method}_seed{seed}.json"
    out.write_text(json.dumps(log))
    console.log(f"saved {out}")


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


#  5. THE EA AND THE RANDOM BASELINE
# ============================================================================ #
def run_ea(
    mode: str,
    seed: int,
    mu: int = MU,
    num_gens: int = NUM_GENS,
) -> None:
    """Run one EA with the given survivor-selection mode.

    Budget: mu (initial population) + num_gens * mu offspring evaluations.
    Generation 0 (the initial population) is logged so the curve has
    num_gens + 1 points and starts at the same value as the random baseline.
    """
    rng = seed_everything(seed)
    targets = load_targets()
    log = new_log()

    parents = list(
        evaluate(Population([make_individual() for _ in range(mu)]), targets)
    )
    log_generation(log, parents)

    for _ in range(num_gens):
        kids = make_offspring(parents, mu, rng)
        kids = list(evaluate(Population(kids), targets))
        parents = survivor_selection(parents, kids, mode=mode, mu=mu)
        log_generation(log, parents)

    save_log(log, mode, seed, mu, num_gens)


def run_random(seed: int, mu: int = MU, num_gens: int = NUM_GENS) -> None:
    """Random-search baseline: same evaluation budget as the EA, no evolution.

    num_gens + 1 batches of mu random bodies, so batch i lines up with EA
    generation i and the total evaluation count is identical.
    """
    seed_everything(seed)
    targets = load_targets()
    log = new_log()

    for _ in range(num_gens + 1):
        batch = list(
            evaluate(Population([make_individual() for _ in range(mu)]), targets)
        )
        log_generation(log, batch)

    save_log(log, "random", seed, mu, num_gens)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    for seed in range(NUM_SEEDS):
        for mode in ("fitness", "age"):
            console.log(f"=== seed {seed} mode {mode} ===")
            run_ea(mode, seed)

    for seed in range(NUM_SEEDS):
        console.log(f"=== seed {seed} mode random ===")
        run_random(seed)


if __name__ == "__main__":
    main()

"""Laboratory work 1: real-coded genetic algorithm for Alpinen No. 1."""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path


DIMENSION = 10
LOWER = -10.0
UPPER = 10.0
OPTIMUM = 0.0


def alpinen1(x: list[float]) -> float:
    return sum(abs(value * math.sin(value) + 0.1 * value) for value in x)


@dataclass(frozen=True)
class GAConfig:
    population_size: int
    generations: int
    crossover_probability: float = 0.9
    mutation_probability: float = 0.1
    mutation_sigma: float = 0.5
    tournament_size: int = 3

    @property
    def evaluations(self) -> int:
        return self.population_size * (self.generations + 1)


@dataclass
class RunResult:
    seed: int
    best_value: float
    best_vector: list[float]
    history: list[float]


def clip(value: float) -> float:
    return max(LOWER, min(UPPER, value))


def tournament(population: list[list[float]], scores: list[float],
               rng: random.Random, size: int) -> list[float]:
    candidates = [rng.randrange(len(population)) for _ in range(size)]
    winner = min(candidates, key=lambda index: scores[index])
    return population[winner].copy()


def arithmetic_crossover(first: list[float], second: list[float],
                         rng: random.Random) -> tuple[list[float], list[float]]:
    alpha = rng.random()
    child_a = [clip(alpha * a + (1 - alpha) * b) for a, b in zip(first, second)]
    child_b = [clip(alpha * b + (1 - alpha) * a) for a, b in zip(first, second)]
    return child_a, child_b


def mutate(individual: list[float], config: GAConfig, rng: random.Random) -> None:
    for index in range(DIMENSION):
        if rng.random() < config.mutation_probability:
            individual[index] = clip(
                individual[index] + rng.gauss(0.0, config.mutation_sigma)
            )


def run_ga(seed: int, config: GAConfig) -> RunResult:
    rng = random.Random(seed)
    population = [
        [rng.uniform(LOWER, UPPER) for _ in range(DIMENSION)]
        for _ in range(config.population_size)
    ]
    scores = [alpinen1(individual) for individual in population]
    best_index = min(range(len(population)), key=scores.__getitem__)
    best_vector = population[best_index].copy()
    best_value = scores[best_index]
    history = [best_value]

    for _ in range(config.generations):
        next_population = [best_vector.copy()]
        while len(next_population) < config.population_size:
            first = tournament(population, scores, rng, config.tournament_size)
            second = tournament(population, scores, rng, config.tournament_size)
            if rng.random() < config.crossover_probability:
                children = arithmetic_crossover(first, second, rng)
            else:
                children = (first, second)
            for child in children:
                mutate(child, config, rng)
                next_population.append(child)
                if len(next_population) == config.population_size:
                    break
        population = next_population
        scores = [alpinen1(individual) for individual in population]
        current_index = min(range(len(population)), key=scores.__getitem__)
        if scores[current_index] < best_value:
            best_value = scores[current_index]
            best_vector = population[current_index].copy()
        history.append(best_value)
    return RunResult(seed, best_value, best_vector, history)


def random_search(seed: int, evaluations: int) -> RunResult:
    rng = random.Random(seed)
    best_value = float("inf")
    best_vector: list[float] = []
    history: list[float] = []
    for _ in range(evaluations):
        candidate = [rng.uniform(LOWER, UPPER) for _ in range(DIMENSION)]
        value = alpinen1(candidate)
        if value < best_value:
            best_value, best_vector = value, candidate
        history.append(best_value)
    return RunResult(seed, best_value, best_vector, history)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, series: dict[str, list[float]], xlabel: str,
              ylabel: str) -> None:
    width, height, left, top = 1000, 560, 75, 35
    right, bottom = 25, 65
    all_values = [value for values in series.values() for value in values]
    low, high = min(all_values), max(all_values)
    if math.isclose(low, high):
        high = low + 1.0
    plot_width, plot_height = width - left - right, height - top - bottom

    def point(index: int, value: float, count: int) -> tuple[float, float]:
        x = left + index * plot_width / max(1, count - 1)
        y = top + (high - value) * plot_height / (high - low)
        return x, y

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="black"/>',
        f'<text x="{width/2}" y="{height-18}" text-anchor="middle">{xlabel}</text>',
        f'<text transform="translate(18,{height/2}) rotate(-90)" text-anchor="middle">{ylabel}</text>',
    ]
    for series_index, (name, values) in enumerate(series.items()):
        coordinates = " ".join(
            f"{x:.1f},{y:.1f}" for index, value in enumerate(values)
            for x, y in [point(index, value, len(values))]
        )
        y_legend = top + 20 * series_index
        lines.extend([
            f'<polyline points="{coordinates}" fill="none" stroke="{colors[series_index % len(colors)]}" stroke-width="2"/>',
            f'<text x="{width-right-180}" y="{y_legend}" fill="{colors[series_index % len(colors)]}">{name}</text>',
        ])
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiments(output_dir: Path, runs: int = 20) -> None:
    seeds = list(range(20260101, 20260101 + runs))
    configs = {
        "population_30": GAConfig(30, 199, mutation_sigma=0.5),
        "population_60": GAConfig(60, 99, mutation_sigma=0.5),
    }
    summaries: list[dict[str, object]] = []
    histories: dict[str, list[list[float]]] = {}
    for name, config in configs.items():
        results = [run_ga(seed, config) for seed in seeds]
        histories[name] = [result.history for result in results]
        values = [result.best_value for result in results]
        for result in results:
            summaries.append({
                "method": "genetic_algorithm", "configuration": name,
                "seed": result.seed, "evaluations": config.evaluations,
                "best_value": f"{result.best_value:.12g}",
                "best_vector": ";".join(f"{v:.8f}" for v in result.best_vector),
            })
        print(name, "mean=", statistics.mean(values), "best=", min(values))

    random_results = [random_search(seed, configs["population_30"].evaluations) for seed in seeds]
    histories["random_search"] = [result.history for result in random_results]
    for result in random_results:
        summaries.append({
            "method": "random_search", "configuration": "same_budget",
            "seed": result.seed, "evaluations": configs["population_30"].evaluations,
            "best_value": f"{result.best_value:.12g}",
            "best_vector": ";".join(f"{v:.8f}" for v in result.best_vector),
        })
    write_csv(output_dir / "results.csv", summaries)

    trajectory: dict[str, list[float]] = {}
    for name, runs_history in histories.items():
        length = min(len(history) for history in runs_history)
        trajectory[name] = [
            statistics.mean(history[index] for history in runs_history[:runs])
            for index in range(length)
        ]
    write_csv(output_dir / "trajectory.csv", [
        {"step": index, **{name: f"{values[index]:.12g}" for name, values in trajectory.items()}}
        for index in range(max(map(len, trajectory.values())))
        if all(index < len(values) for values in trajectory.values())
    ])
    write_svg(output_dir / "convergence.svg", trajectory, "Generation", "Best objective value")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-experiments", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.run_experiments:
        run_experiments(args.output_dir)
    else:
        result = run_ga(20260101, GAConfig(30, 199))
        print(f"seed={result.seed} best={result.best_value:.12g}")
        print("vector=", " ".join(f"{value:.8f}" for value in result.best_vector))


if __name__ == "__main__":
    main()

"""Punto de entrada por consola para ejecutar y verificar el MVP (Fase 1 SVRAI).

Ejemplos de uso desde la carpeta `proyecto/`:
    python run_mvp.py                      # Simula 1 partida con traza detallada + resumen
    python run_mvp.py --games 20           # Simula un torneo rápido de 20 partidas
    python run_mvp.py --games 50 --seed 42 # Torneo reproducible de 50 partidas
"""

from __future__ import annotations

import argparse

from svrai_catan.agents import ReactiveHeuristicAgent
from svrai_catan.environment import CatanSimulationEnvironment
from svrai_catan.metrics import TournamentSummary


def build_mvp_agents() -> list[ReactiveHeuristicAgent]:
    """Instancia los 4 agentes reactivos/heurísticos del MVP."""
    return [
        ReactiveHeuristicAgent(name="Agente_1"),
        ReactiveHeuristicAgent(name="Agente_2"),
        ReactiveHeuristicAgent(name="Agente_3"),
        ReactiveHeuristicAgent(name="Agente_4"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulador Multi-Agente sobre Catán - Fase 1 (MVP SVRAI)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Número de partidas a simular (por defecto: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla base para reproducibilidad experimental (por defecto: 42).",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=250,
        help="Límite máximo de rondas por partida (por defecto: 250).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Desactiva la traza paso a paso incluso cuando se ejecuta 1 sola partida.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Fuerza mostrar la traza paso a paso por consola.",
    )
    args = parser.parse_args()

    show_trace = args.verbose or (args.games == 1 and not args.quiet)
    summary = TournamentSummary()

    for i in range(args.games):
        game_seed = (args.seed + i) if args.seed is not None else None
        agents = build_mvp_agents()
        # Rotar el orden de asiento en torneos de varias partidas para evitar sesgo de salida
        if args.games > 1:
            shift = i % 4
            agents = agents[shift:] + agents[:shift]

        env = CatanSimulationEnvironment(
            agents=agents,
            max_points=10,
            max_rounds=args.max_rounds,
            seed=game_seed,
            verbose=show_trace,
        )
        metrics = env.run_game(game_id=i + 1)
        summary.games.append(metrics)

    print("\n" + summary.format_report())


if __name__ == "__main__":
    main()

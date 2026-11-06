"""Recolección de métricas objetivas para la evaluación experimental (Sección 4 Propuesta).

Mide en cada partida y agrega en torneos:
- Tasa de victorias (Success rate) por agente y por tipo de arquitectura.
- Eficiencia de negociación: ratio de propuestas aceptadas vs. rechazadas.
- Sobrecarga de comunicación: número de mensajes intercambiados por turno comercial.
- Eficacia de recursos: cantidad de cartas perdidas por descarte ante tirada de 7 (ladrón).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentGameStats:
    """Estadísticas individuales de un agente en una partida."""

    agent_name: str
    agent_type: str
    final_victory_points: int = 0
    won: bool = False
    turns_taken: int = 0
    trades_initiated: int = 0
    trades_accepted_as_initiator: int = 0
    trades_accepted_as_partner: int = 0
    bank_trades: int = 0
    messages_sent: int = 0
    cards_discarded_to_robber: int = 0
    settlements_built: int = 0
    cities_built: int = 0
    roads_built: int = 0
    dev_cards_bought: int = 0


@dataclass
class GameMetrics:
    """Resumen cuantitativo de una partida completa de 4 agentes."""

    game_id: int
    seed: int | None
    winner: str | None
    winner_type: str | None
    total_turns: int = 0
    total_rounds: int = 0
    trade_turns: int = 0  # Turnos en los que hubo al menos una negociación
    proposals_initiated: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    total_messages: int = 0
    total_cards_discarded: int = 0
    elapsed_seconds: float = 0.0
    per_agent: dict[str, AgentGameStats] = field(default_factory=dict)

    @property
    def negotiation_efficiency(self) -> float:
        """Ratio de rondas comerciales aceptadas sobre el total iniciadas."""
        if self.proposals_initiated == 0:
            return 0.0
        return self.proposals_accepted / self.proposals_initiated

    @property
    def messages_per_trade_turn(self) -> float:
        """Sobrecarga media de mensajes por turno con actividad comercial."""
        if self.trade_turns == 0:
            return 0.0
        return self.total_messages / self.trade_turns

    @property
    def messages_per_turn(self) -> float:
        """Sobrecarga media de mensajes sobre el total de turnos jugados."""
        if self.total_turns == 0:
            return 0.0
        return self.total_messages / self.total_turns


@dataclass
class TournamentSummary:
    """Agregador estadístico sobre N partidas simuladas."""

    games: list[GameMetrics] = field(default_factory=list)

    @property
    def num_games(self) -> int:
        return len(self.games)

    def win_counts_by_agent(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if self.games:
            for name in self.games[0].per_agent:
                counts[name] = 0
        for g in self.games:
            if g.winner is not None:
                counts[g.winner] = counts.get(g.winner, 0) + 1
        return counts

    def win_counts_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for g in self.games:
            for stats in g.per_agent.values():
                counts.setdefault(stats.agent_type, 0)
            if g.winner_type is not None:
                counts[g.winner_type] = counts.get(g.winner_type, 0) + 1
        return counts

    def format_report(self) -> str:
        """Genera una tabla resumen legible por consola con las 4 métricas clave."""
        if not self.games:
            return "Sin partidas registradas."

        n = len(self.games)
        completed = sum(1 for g in self.games if g.winner is not None)
        avg_rounds = sum(g.total_rounds for g in self.games) / n
        avg_turns = sum(g.total_turns for g in self.games) / n
        total_init = sum(g.proposals_initiated for g in self.games)
        total_acc = sum(g.proposals_accepted for g in self.games)
        total_rej = sum(g.proposals_rejected for g in self.games)
        total_msgs = sum(g.total_messages for g in self.games)
        total_trade_turns = sum(g.trade_turns for g in self.games)
        total_discards = sum(g.total_cards_discarded for g in self.games)
        total_time = sum(g.elapsed_seconds for g in self.games)

        eff = (100.0 * total_acc / total_init) if total_init > 0 else 0.0
        msgs_per_trade_turn = (
            total_msgs / total_trade_turns if total_trade_turns > 0 else 0.0
        )

        lines = [
            "=" * 74,
            f" RESUMEN EXPERIMENTAL DEL SIMULADOR CATÁN ({n} partidas)",
            "=" * 74,
            f" Partidas completadas (10 PV): {completed}/{n} ({100.0 * completed / n:.1f}%)",
            f" Duración media:               {avg_rounds:.1f} rondas ({avg_turns:.1f} turnos)",
            f" Tiempo total de cómputo:      {total_time:.2f} s ({total_time / n * 1000:.1f} ms/partida)",
            "-" * 74,
            " MÉTRICAS OBJETIVAS GLOBALES:",
            f"  • Propuestas 1x1 iniciadas:  {total_init} ({total_init / n:.1f}/partida)",
            f"  • Aceptadas vs. Rechazadas:  {total_acc} vs. {total_rej} (Eficiencia: {eff:.1f}%)",
            f"  • Mensajes ACL totales:      {total_msgs} ({msgs_per_trade_turn:.2f} msgs/turno comercial)",
            f"  • Cartas perdidas por '7':   {total_discards} ({total_discards / n:.1f}/partida)",
            "-" * 74,
            " DESGLOSE POR AGENTE:",
            f"  {'Agente':<14} {'Tipo':<14} {'Victorias':>11} {'PV med':>8} {'Trueques':>10} {'Descartes':>10}",
        ]

        agent_names = list(self.games[0].per_agent.keys())
        for name in agent_names:
            a_type = self.games[0].per_agent[name].agent_type
            wins = sum(1 for g in self.games if g.winner == name)
            avg_vp = sum(g.per_agent[name].final_victory_points for g in self.games) / n
            trades = sum(
                g.per_agent[name].trades_accepted_as_initiator
                + g.per_agent[name].trades_accepted_as_partner
                for g in self.games
            )
            discards = sum(g.per_agent[name].cards_discarded_to_robber for g in self.games)
            win_str = f"{wins} ({100.0 * wins / n:4.1f}%)"
            lines.append(
                f"  {name:<14} {a_type:<14} {win_str:>11} {avg_vp:>8.2f} {trades:>10} {discards:>10}"
            )

        lines.append("=" * 74)
        return "\n".join(lines)

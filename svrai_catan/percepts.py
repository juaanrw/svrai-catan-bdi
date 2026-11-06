"""Representación de percepciones con observabilidad parcial (información imperfecta).

Alineado con el Tema 1 (Sección 5.1: Observabilidad parcial) y Tema 2 (Creencias):
el entorno NUNCA entrega a un agente las referencias internas a los objetos `player`
de sus rivales. Cada agente percibe únicamente:
  1. Su propio estado privado completo (mano exacta y cartas de desarrollo ocultas).
  2. El estado público del tablero (hexágonos, vértices, aristas, puertos, ladrón).
  3. El resumen público de los rivales (puntos visibles, tamaño total de mano sin
     desglosar tipos de recurso, caballeros jugados, longitud de camino).
  4. Eventos públicos ocurridos (tiradas, producciones públicas, robos, construcciones),
     dejando preparado el gancho exacto para el rastreo de creencias BDI en la Fase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ontology import GamePhase


@dataclass(frozen=True)
class HexInfo:
    """Información pública e inmutable de un hexágono del tablero."""

    index: int
    resource_type: str
    dice_num: int | None
    has_robber: bool
    vertices: tuple[Any, ...]


@dataclass(frozen=True)
class VertexInfo:
    """Información pública e inmutable de una intersección (vértice) del grafo."""

    coord: Any
    vertex_index: int
    adjacent_hexes: tuple[int, ...]
    neighbors: tuple[Any, ...]
    owner: str | None
    building: str | None  # None | "SETTLEMENT" | "CITY"
    port: str | None
    is_colonised: bool


@dataclass(frozen=True)
class BoardView:
    """Instantánea pública de solo lectura del grafo del tablero."""

    hexes: dict[int, HexInfo]
    vertices: dict[Any, VertexInfo]
    roads: tuple[tuple[Any, Any, str], ...]  # (v1, v2, nombre_propietario)
    dev_cards_remaining: int


@dataclass(frozen=True)
class SelfState:
    """Estado interno privado del propio agente."""

    name: str
    resources: dict[str, int]
    victory_points: int
    visible_victory_points: int
    dev_cards: dict[str, int]
    new_dev_cards: tuple[str, ...]
    dev_card_played_this_turn: bool
    roads_left: int
    settlements_left: int
    cities_left: int
    ports: tuple[str, ...]
    knights_played: int
    max_road_length: int
    longest_road_flag: bool
    largest_army_flag: bool
    settlements: tuple[Any, ...]
    cities: tuple[Any, ...]
    roads: tuple[tuple[Any, Any], ...]

    @property
    def total_resources(self) -> int:
        return sum(self.resources.values())


@dataclass(frozen=True)
class PublicPlayerInfo:
    """Información públicamente observable de un jugador rival (sin desglose de mano)."""

    name: str
    visible_victory_points: int
    total_resource_cards: int
    total_hidden_dev_cards: int
    knights_played: int
    max_road_length: int
    longest_road_flag: bool
    largest_army_flag: bool
    roads_left: int
    settlements_left: int
    cities_left: int
    ports: tuple[str, ...]
    settlements: tuple[Any, ...]
    cities: tuple[Any, ...]
    roads: tuple[tuple[Any, Any], ...]


@dataclass(frozen=True)
class LegalMoves:
    """Conjunto de acciones legales validadas por la API del motor catan-rl."""

    settlements: tuple[Any, ...] = ()
    cities: tuple[Any, ...] = ()
    roads: tuple[tuple[Any, Any], ...] = ()
    robber_hexes: tuple[int, ...] = ()
    robber_targets_by_hex: dict[int, tuple[str, ...]] = field(default_factory=dict)
    bank_trade_ratios: dict[str, int] = field(default_factory=dict)
    cards_to_discard: int = 0
    can_buy_dev_card: bool = False
    playable_dev_cards: tuple[str, ...] = ()


@dataclass(frozen=True)
class PublicEvent:
    """Evento público ocurrido en la mesa (útil para trazas y para creencias BDI en Fase 2)."""

    turn_number: int
    event_type: str
    actor: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Percept:
    """Percepción completa entregada a `agent.perceive(percept)` en cada ciclo."""

    agent_name: str
    turn_number: int
    round_number: int
    current_player: str
    phase: GamePhase
    last_dice_roll: int | None
    self_state: SelfState
    public_players: dict[str, PublicPlayerInfo]
    board: BoardView
    legal_moves: LegalMoves
    recent_events: tuple[PublicEvent, ...] = ()

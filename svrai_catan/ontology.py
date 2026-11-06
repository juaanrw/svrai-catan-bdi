"""Ontología compartida del dominio Catán y primitivos de comunicación FIPA-ACL.

Alineado con el Tema 4 de SVRAI (Sección 2.2: Ontología compartida y actos de habla):
define el vocabulario común (recursos, costes, fases, acciones y performativas)
que comparten el entorno y los 4 agentes autónomos para garantizar acuerdo semántico.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# Vocabulario de recursos (idéntico a las claves internas del motor catan-rl)
RESOURCES: tuple[str, ...] = ("BRICK", "WOOD", "WHEAT", "SHEEP", "ORE")

RESOURCE_NAMES_ES: dict[str, str] = {
    "BRICK": "Arcilla",
    "WOOD": "Madera",
    "WHEAT": "Trigo",
    "SHEEP": "Oveja",
    "ORE": "Mineral",
    "DESERT": "Desierto",
}

# Costes oficiales de construcción en Catán
BUILD_COSTS: dict[str, dict[str, int]] = {
    "ROAD": {"BRICK": 1, "WOOD": 1},
    "SETTLEMENT": {"BRICK": 1, "WOOD": 1, "WHEAT": 1, "SHEEP": 1},
    "CITY": {"ORE": 3, "WHEAT": 2},
    "DEV_CARD": {"ORE": 1, "WHEAT": 1, "SHEEP": 1},
}

# Puntos de probabilidad (pips = numerador sobre 36 combinaciones de 2d6)
DICE_PIPS: dict[int | None, int] = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 0,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
    None: 0,
}


class GamePhase(str, Enum):
    """Subfase en la que el entorno solicita una decisión al agente."""

    SETUP_SETTLEMENT = "SETUP_SETTLEMENT"
    SETUP_ROAD = "SETUP_ROAD"
    DISCARD = "DISCARD"
    MOVE_ROBBER = "MOVE_ROBBER"
    TURN_PLAY = "TURN_PLAY"


class ActionType(str, Enum):
    """Acciones físicas y comunicativas ejecutables sobre el entorno."""

    BUILD_SETTLEMENT = "BUILD_SETTLEMENT"
    BUILD_CITY = "BUILD_CITY"
    BUILD_ROAD = "BUILD_ROAD"
    BUY_DEV_CARD = "BUY_DEV_CARD"
    PLAY_DEV_CARD = "PLAY_DEV_CARD"
    PROPOSE_PLAYER_TRADE = "PROPOSE_PLAYER_TRADE"
    TRADE_BANK = "TRADE_BANK"
    DISCARD_CARDS = "DISCARD_CARDS"
    MOVE_ROBBER = "MOVE_ROBBER"
    END_TURN = "END_TURN"


class Performative(str, Enum):
    """Subconjunto de actos de habla FIPA-ACL (Tema 4) para negociación comercial."""

    CFP = "cfp"                          # Call for Proposal (oferta de trueque difundida)
    PROPOSE = "propose"                  # Aceptación / propuesta de un contratista
    REFUSE = "refuse"                    # Rechazo de la oferta
    ACCEPT_PROPOSAL = "accept-proposal"  # Adjudicación del trueque por el iniciador
    REJECT_PROPOSAL = "reject-proposal"  # Descarte de propuestas sobrantes
    INFORM = "inform"                    # Confirmación de ejecución del intercambio


@dataclass(frozen=True)
class TradeOffer:
    """Contenido semántico de una oferta comercial entre agentes o con la banca.

    Interpretación desde el punto de vista del emisor (iniciador):
    entrega `give_amount` unidades de `give_resource` a cambio de recibir
    `want_amount` unidades de `want_resource`.
    """

    give_resource: str
    give_amount: int
    want_resource: str
    want_amount: int

    def is_valid_1x1(self) -> bool:
        return (
            self.give_resource in RESOURCES
            and self.want_resource in RESOURCES
            and self.give_resource != self.want_resource
            and self.give_amount == 1
            and self.want_amount == 1
        )


@dataclass(frozen=True)
class ACLMessage:
    """Mensaje formal entre agentes inspirado en FIPA-ACL (Tema 4 SVRAI)."""

    msg_id: int
    performative: Performative
    sender: str
    receiver: str
    content: TradeOffer
    conversation_id: int
    protocol: str = "cnp-reactive-mvp"
    turn_number: int = 0
    utility: float | None = None  # Reservado para Fase 2 (evaluación de utilidad CNP)


@dataclass(frozen=True)
class Action:
    """Acción emitida por el método `decide()` de un agente."""

    action_type: ActionType
    vertex: Any = None
    edge: tuple[Any, Any] | None = None
    hex_index: int | None = None
    target_player: str | None = None
    dev_card: str | None = None
    dev_card_args: dict[str, Any] = field(default_factory=dict)
    trade_offer: TradeOffer | None = None
    discard_resources: dict[str, int] = field(default_factory=dict)
    reason: str = ""  # Explicabilidad / trazabilidad de qué regla o capa tomó la decisión

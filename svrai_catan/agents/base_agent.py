"""Interfaz abstracta del Agente y su bucle Percepción-Decisión-Acción.

Alineado con el Tema 1 y Tema 2 de SVRAI:
- Separa estrictamente la fase de percepción (`perceive`) de la deliberación/selección
  de acción (`decide`), exponiendo el paso atómico `step(percept) -> Action`.
- Define la interfaz de habilidad social (`evaluate_trade_cfp`, `select_trade_partner`)
  de modo que tanto el agente reactivo del MVP (Fase 1) como los futuros agentes
  BDI + Contract Net (Fase 2) y Maquiavélicos (Fase 3) compartan exactamente el
  mismo contrato con el simulador.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..ontology import ACLMessage, Action
from ..percepts import Percept


class BaseAgent(ABC):
    """Clase base abstracta para cualquier agente jugador de Catán."""

    def __init__(self, name: str, agent_type: str = "BaseAgent") -> None:
        self.name = name
        self.agent_type = agent_type
        self.last_percept: Percept | None = None

    def reset_for_new_game(self) -> None:
        """Reinicia la memoria episódica del agente antes de comenzar una nueva partida."""
        self.last_percept = None

    def perceive(self, percept: Percept) -> None:
        """Actualiza el estado interno del agente a partir de la percepción actual.

        En un agente reactivo (MVP) solo almacena la percepción inmediata; en la
        Fase 2 (BDI) actualizará además la base de creencias deductiva/probabilística.
        """
        self.last_percept = percept

    @abstractmethod
    def decide(self) -> Action:
        """Selecciona la siguiente acción a ejecutar según el estado percibido."""

    def step(self, percept: Percept) -> Action:
        """Ejecuta el ciclo completo Percepción -> Decisión -> Acción."""
        self.perceive(percept)
        return self.decide()

    @abstractmethod
    def evaluate_trade_cfp(
        self,
        cfp_message: ACLMessage,
        percept: Percept,
    ) -> tuple[bool, float | None]:
        """Evalúa un Call for Proposal (`[cfp]`) de comercio emitido por otro agente.

        Devuelve `(acepta, utilidad_estimada)`.
        """

    def select_trade_partner(
        self,
        proposals: list[ACLMessage],
        percept: Percept,
    ) -> ACLMessage | None:
        """Adjudica el intercambio entre los contratistas que respondieron `[propose]`.

        Por defecto (MVP) selecciona la primera propuesta válida sin discriminar
        quién va ganando. En Fases 2 y 3 podrá sobrecargarse con criterios de
        utilidad y bloqueo estratégico (Teoría de Juegos).
        """
        if not proposals:
            return None
        return proposals[0]

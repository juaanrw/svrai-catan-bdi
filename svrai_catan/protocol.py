"""Mediador del protocolo de negociación comercial entre agentes (Contract Net / 1x1).

Alineado con el Tema 4 de SVRAI (`c4-comunicacion.jl`):
- Ningún agente accede directamente a `other_player.resources`.
- Toda negociación transcurre mediante envío de mensajes `ACLMessage` con performativas
  FIPA-ACL (`cfp`, `propose`, `refuse`, `accept-proposal`, `reject-proposal`, `inform`).
- Registra en bitácora cada acto de habla para medir sobrecarga de comunicación y
  eficiencia de negociación tanto en el MVP como en las ampliaciones de Fases 2 y 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .agents.base_agent import BaseAgent
from .ontology import ACLMessage, Performative, TradeOffer
from .percepts import Percept


@dataclass(frozen=True)
class TradeResolution:
    """Resultado completo de una ronda de negociación iniciada por un agente."""

    conversation_id: int
    initiator: str
    offer: TradeOffer
    accepted: bool
    partner: str | None
    messages: tuple[ACLMessage, ...]


class TradeChannel:
    """Canal formal de paso de mensajes para la fase de comercio entre los 4 agentes."""

    def __init__(self, protocol_name: str = "cnp-reactive-mvp") -> None:
        self.protocol_name = protocol_name
        self._next_msg_id: int = 1
        self._next_conv_id: int = 1
        self.message_log: list[ACLMessage] = []

    def _emit(
        self,
        performative: Performative,
        sender: str,
        receiver: str,
        content: TradeOffer,
        conv_id: int,
        turn_number: int,
        utility: float | None = None,
    ) -> ACLMessage:
        msg = ACLMessage(
            msg_id=self._next_msg_id,
            performative=performative,
            sender=sender,
            receiver=receiver,
            content=content,
            conversation_id=conv_id,
            protocol=self.protocol_name,
            turn_number=turn_number,
            utility=utility,
        )
        self._next_msg_id += 1
        self.message_log.append(msg)
        return msg

    def run_trade_negotiation(
        self,
        initiator: BaseAgent,
        contractors: list[BaseAgent],
        offer: TradeOffer,
        turn_number: int,
        percept_provider: Callable[[str], Percept],
    ) -> TradeResolution:
        """Ejecuta una ronda de negociación siguiendo la estructura Contract Net:

        1. Difusión (`CFP`): el iniciador envía la oferta a los otros 3 agentes.
        2. Evaluación (`PROPOSE` / `REFUSE`): cada receptor evalúa con su propia percepción.
        3. Adjudicación (`ACCEPT_PROPOSAL` / `REJECT_PROPOSAL`): el iniciador selecciona
           al contratista ganador (si al menos uno propuso aceptar).
        """
        conv_id = self._next_conv_id
        self._next_conv_id += 1
        round_messages: list[ACLMessage] = []

        # Fase 1: Anuncio (Call for Proposal) a cada contratista
        cfp_by_contractor: dict[str, ACLMessage] = {}
        for contractor in contractors:
            cfp_msg = self._emit(
                performative=Performative.CFP,
                sender=initiator.name,
                receiver=contractor.name,
                content=offer,
                conv_id=conv_id,
                turn_number=turn_number,
            )
            cfp_by_contractor[contractor.name] = cfp_msg
            round_messages.append(cfp_msg)

        # Fase 2: Evaluación autónoma de cada contratista (sin memoria compartida)
        proposals: list[ACLMessage] = []
        for contractor in contractors:
            contractor_percept = percept_provider(contractor.name)
            accepted, utility = contractor.evaluate_trade_cfp(
                cfp_by_contractor[contractor.name],
                contractor_percept,
            )
            resp_perf = Performative.PROPOSE if accepted else Performative.REFUSE
            resp_msg = self._emit(
                performative=resp_perf,
                sender=contractor.name,
                receiver=initiator.name,
                content=offer,
                conv_id=conv_id,
                turn_number=turn_number,
                utility=utility,
            )
            round_messages.append(resp_msg)
            if accepted:
                proposals.append(resp_msg)

        if not proposals:
            return TradeResolution(
                conversation_id=conv_id,
                initiator=initiator.name,
                offer=offer,
                accepted=False,
                partner=None,
                messages=tuple(round_messages),
            )

        # Fase 3: Adjudicación (Award) por parte del iniciador
        initiator_percept = percept_provider(initiator.name)
        chosen = initiator.select_trade_partner(proposals, initiator_percept)
        if chosen is None:
            for prop in proposals:
                rej = self._emit(
                    performative=Performative.REJECT_PROPOSAL,
                    sender=initiator.name,
                    receiver=prop.sender,
                    content=offer,
                    conv_id=conv_id,
                    turn_number=turn_number,
                )
                round_messages.append(rej)
            return TradeResolution(
                conversation_id=conv_id,
                initiator=initiator.name,
                offer=offer,
                accepted=False,
                partner=None,
                messages=tuple(round_messages),
            )

        winner_name = chosen.sender
        for prop in proposals:
            perf = (
                Performative.ACCEPT_PROPOSAL
                if prop.sender == winner_name
                else Performative.REJECT_PROPOSAL
            )
            award_msg = self._emit(
                performative=perf,
                sender=initiator.name,
                receiver=prop.sender,
                content=offer,
                conv_id=conv_id,
                turn_number=turn_number,
            )
            round_messages.append(award_msg)

        return TradeResolution(
            conversation_id=conv_id,
            initiator=initiator.name,
            offer=offer,
            accepted=True,
            partner=winner_name,
            messages=tuple(round_messages),
        )

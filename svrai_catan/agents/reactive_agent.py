"""Agente Reactivo/Heurístico correspondiente a la Fase 1 (MVP) del proyecto.

Características del diseño (según Propuesta SVRAI - Fase 1):
1. Arquitectura reactiva basada en reglas condición-acción ordenadas por prioridad
   (inspirada en Subsumption Architecture, Tema 2).
2. Construcción espacial guiada por heurísticas estáticas locales (probabilidad de
   dados en vértices y vecindad inmediata en caminos), evitando búsqueda pesada en grafos.
3. Comercio reactivo 1x1: propone y acepta trueques 1 por 1 únicamente en función de
   si necesita el recurso ofertado y le sobra el solicitado, SIN mantener modelo de
   creencias de manos rivales y SIN evaluar quién va ganando la partida.
"""

from __future__ import annotations

from typing import Any

from ..heuristics import (
    can_afford,
    score_city_vertex,
    score_road_edge,
    score_settlement_vertex,
    select_cards_to_discard,
    select_robber_placement,
)
from ..ontology import (
    BUILD_COSTS,
    RESOURCES,
    ACLMessage,
    Action,
    ActionType,
    GamePhase,
    TradeOffer,
)
from ..percepts import Percept
from .base_agent import BaseAgent


class ReactiveHeuristicAgent(BaseAgent):
    """Agente base reactivo/heurístico del MVP."""

    def __init__(
        self,
        name: str,
        max_trade_proposals_per_turn: int = 4,
    ) -> None:
        super().__init__(name=name, agent_type="ReactiveMVP")
        self.max_trade_proposals_per_turn = max_trade_proposals_per_turn
        self._setup_covered_resources: set[str] = set()
        self._last_active_turn: int = -1
        self._attempted_trades_this_turn: set[tuple[str, str]] = set()
        self._trades_proposed_this_turn: int = 0

    def reset_for_new_game(self) -> None:
        super().reset_for_new_game()
        self._setup_covered_resources = set()
        self._last_active_turn = -1
        self._attempted_trades_this_turn = set()
        self._trades_proposed_this_turn = 0

    # ------------------------------------------------------------------
    # Bucle principal de Decisión (por subfase del entorno)
    # ------------------------------------------------------------------
    def decide(self) -> Action:
        if self.last_percept is None:
            raise RuntimeError(f"El agente {self.name} llamó a decide() sin percepción previa.")

        percept = self.last_percept
        phase = percept.phase

        if phase == GamePhase.SETUP_SETTLEMENT:
            return self._decide_setup_settlement(percept)
        if phase == GamePhase.SETUP_ROAD:
            return self._decide_setup_road(percept)
        if phase == GamePhase.DISCARD:
            return self._decide_discard(percept)
        if phase == GamePhase.MOVE_ROBBER:
            return self._decide_robber(percept)
        if phase == GamePhase.TURN_PLAY:
            return self._decide_turn_play(percept)

        return Action(ActionType.END_TURN, reason="Fase desconocida")

    # ------------------------------------------------------------------
    # Subfases de preparación inicial, descarte y ladrón
    # ------------------------------------------------------------------
    def _decide_setup_settlement(self, percept: Percept) -> Action:
        candidates = percept.legal_moves.settlements
        best_vertex = max(
            candidates,
            key=lambda v: score_settlement_vertex(
                v, percept.board, covered_resources=self._setup_covered_resources
            ),
        )
        for hex_idx in percept.board.vertices[best_vertex].adjacent_hexes:
            res_type = percept.board.hexes[hex_idx].resource_type
            if res_type != "DESERT":
                self._setup_covered_resources.add(res_type)

        return Action(
            action_type=ActionType.BUILD_SETTLEMENT,
            vertex=best_vertex,
            reason="Setup: vértice con mayor esperanza de dados y diversidad",
        )

    def _decide_setup_road(self, percept: Percept) -> Action:
        own_nodes = self._get_own_network_nodes(percept)
        best_edge = max(
            percept.legal_moves.roads,
            key=lambda e: score_road_edge(e, percept.board, self.name, own_nodes),
        )
        return Action(
            action_type=ActionType.BUILD_ROAD,
            edge=best_edge,
            reason="Setup: camino orientado hacia intersección colonizable",
        )

    def _decide_discard(self, percept: Percept) -> Action:
        recipe = self._current_target_recipe(percept)
        to_discard = select_cards_to_discard(
            percept.self_state.resources,
            percept.legal_moves.cards_to_discard,
            recipe,
        )
        return Action(
            action_type=ActionType.DISCARD_CARDS,
            discard_resources=to_discard,
            reason="Regla del 7: descarte reactivo de recursos excedentes",
        )

    def _decide_robber(self, percept: Percept) -> Action:
        hex_idx, victim = select_robber_placement(percept)
        return Action(
            action_type=ActionType.MOVE_ROBBER,
            hex_index=hex_idx,
            target_player=victim,
            reason="Ladrón: bloqueo de hexágono rival de alta probabilidad",
        )

    # ------------------------------------------------------------------
    # Turno principal: arquitectura reactiva por prioridad de reglas
    # ------------------------------------------------------------------
    def _decide_turn_play(self, percept: Percept) -> Action:
        if percept.turn_number != self._last_active_turn:
            self._last_active_turn = percept.turn_number
            self._attempted_trades_this_turn.clear()
            self._trades_proposed_this_turn = 0

        # Prioridad 1: Jugar carta de desarrollo disponible si aporta ventaja directa
        dev_action = self._try_play_dev_card(percept)
        if dev_action is not None:
            return dev_action

        # Prioridad 2: Construir inmediatamente si se dispone de recursos y emplazamiento legal
        build_action = self._try_immediate_build(percept)
        if build_action is not None:
            return build_action

        # Prioridad 3: Proponer trueque reactivo 1x1 a los demás agentes
        trade_player_action = self._try_propose_reactive_trade(percept)
        if trade_player_action is not None:
            return trade_player_action

        # Prioridad 4: Comercio marítimo / banca (4:1, 3:1 o 2:1) con excedentes
        trade_bank_action = self._try_bank_trade(percept)
        if trade_bank_action is not None:
            return trade_bank_action

        # Prioridad 5: Pasar turno
        return Action(ActionType.END_TURN, reason="Sin construcciones ni trueques viables")

    def _try_play_dev_card(self, percept: Percept) -> Action | None:
        playable = percept.legal_moves.playable_dev_cards
        if not playable:
            return None

        me = percept.self_state
        recipe = self._current_target_recipe(percept)
        missing = [r for r, req in recipe.items() if me.resources[r] < req]

        if "KNIGHT" in playable:
            hex_idx, victim = select_robber_placement(percept)
            return Action(
                action_type=ActionType.PLAY_DEV_CARD,
                dev_card="KNIGHT",
                dev_card_args={"hex_index": hex_idx, "target_player": victim},
                reason="Jugar Caballero para desplazar ladrón y sumar Gran Ejército",
            )

        if "ROADBUILDER" in playable and me.roads_left > 0 and percept.legal_moves.roads:
            return Action(
                action_type=ActionType.PLAY_DEV_CARD,
                dev_card="ROADBUILDER",
                reason="Jugar Construcción de Carreteras para expandir red gratis",
            )

        if "YEAROFPLENTY" in playable:
            picks: list[str] = []
            temp_hand = dict(me.resources)
            for _ in range(2):
                deficits = [r for r, req in recipe.items() if temp_hand[r] < req]
                chosen = deficits[0] if deficits else ("ORE" if me.cities_left > 0 else "WHEAT")
                picks.append(chosen)
                temp_hand[chosen] += 1
            return Action(
                action_type=ActionType.PLAY_DEV_CARD,
                dev_card="YEAROFPLENTY",
                dev_card_args={"resources": tuple(picks)},
                reason="Jugar Invento para obtener recursos deficitarios",
            )

        if "MONOPOLY" in playable:
            target_res = missing[0] if missing else ("ORE" if me.cities_left > 0 else "WHEAT")
            return Action(
                action_type=ActionType.PLAY_DEV_CARD,
                dev_card="MONOPOLY",
                dev_card_args={"resource": target_res},
                reason=f"Jugar Monopolio sobre {target_res}",
            )

        return None

    def _try_immediate_build(self, percept: Percept) -> Action | None:
        me = percept.self_state
        hand = me.resources
        legal = percept.legal_moves

        # 1. Ciudad (+1 PV y duplica producción de un vértice fuerte)
        if legal.cities and can_afford(hand, "CITY"):
            best_city = max(legal.cities, key=lambda v: score_city_vertex(v, percept.board))
            return Action(
                action_type=ActionType.BUILD_CITY,
                vertex=best_city,
                reason="Construcción heurística: mejorar mejor poblado a ciudad",
            )

        # 2. Poblado (+1 PV en el vértice legal con mayor probabilidad de dados)
        if legal.settlements and can_afford(hand, "SETTLEMENT"):
            best_settlement = max(
                legal.settlements,
                key=lambda v: score_settlement_vertex(v, percept.board),
            )
            return Action(
                action_type=ActionType.BUILD_SETTLEMENT,
                vertex=best_settlement,
                reason="Construcción heurística: poblado en intersección de mayor probabilidad",
            )

        # 3. Camino (solo si necesitamos abrir hueco para poblado, sobran materiales o disputamos Gran Ruta)
        if legal.roads and can_afford(hand, "ROAD"):
            needs_expansion = len(legal.settlements) == 0 and me.settlements_left > 0
            has_road_surplus = hand["BRICK"] >= 2 and hand["WOOD"] >= 2
            chasing_longest_road = (
                me.settlements_left == 0
                and not me.longest_road_flag
                and me.max_road_length >= 4
            )
            if needs_expansion or has_road_surplus or chasing_longest_road:
                own_nodes = self._get_own_network_nodes(percept)
                best_road = max(
                    legal.roads,
                    key=lambda e: score_road_edge(e, percept.board, self.name, own_nodes),
                )
                return Action(
                    action_type=ActionType.BUILD_ROAD,
                    edge=best_road,
                    reason="Construcción heurística: extender carretera hacia vértice libre",
                )

        # 4. Carta de desarrollo (si no estamos a las puertas de pagar una ciudad)
        if legal.can_buy_dev_card:
            almost_city = bool(legal.cities) and hand["ORE"] >= 2 and hand["WHEAT"] >= 2
            if not almost_city:
                return Action(
                    action_type=ActionType.BUY_DEV_CARD,
                    reason="Compra reactiva de carta de desarrollo",
                )

        return None

    def _try_propose_reactive_trade(self, percept: Percept) -> Action | None:
        if self._trades_proposed_this_turn >= self.max_trade_proposals_per_turn:
            return None

        hand = percept.self_state.resources
        recipe = self._current_target_recipe(percept)

        missing = [r for r, req in recipe.items() if hand[r] < req]
        surplus = sorted(
            [r for r in RESOURCES if hand[r] > recipe.get(r, 0)],
            key=lambda r: hand[r] - recipe.get(r, 0),
            reverse=True,
        )

        for r_give in surplus:
            for r_want in missing:
                if r_give == r_want:
                    continue
                pair = (r_give, r_want)
                if pair not in self._attempted_trades_this_turn:
                    self._attempted_trades_this_turn.add(pair)
                    self._trades_proposed_this_turn += 1
                    return Action(
                        action_type=ActionType.PROPOSE_PLAYER_TRADE,
                        trade_offer=TradeOffer(
                            give_resource=r_give,
                            give_amount=1,
                            want_resource=r_want,
                            want_amount=1,
                        ),
                        reason=f"Oferta reactiva 1x1: doy 1 {r_give}, pido 1 {r_want}",
                    )

        return None

    def _try_bank_trade(self, percept: Percept) -> Action | None:
        me = percept.self_state
        hand = me.resources
        ratios = percept.legal_moves.bank_trade_ratios
        recipe = self._current_target_recipe(percept)

        missing = [r for r, req in recipe.items() if hand[r] < req]
        # Si acumulamos 8+ cartas sin objetivo inmediato, aliviamos exceso hacia el recurso más escaso
        if not missing and me.total_resources >= 8:
            scarcest = min(RESOURCES, key=lambda r: hand[r])
            missing = [scarcest]

        if not missing:
            return None

        r_want = missing[0]
        for r_give in sorted(RESOURCES, key=lambda r: hand[r], reverse=True):
            if r_give == r_want:
                continue
            ratio = ratios.get(r_give, 4)
            keep = recipe.get(r_give, 0)
            # Permitir comerciar si el excedente sobre la receta alcanza el ratio del puerto/banca
            if hand[r_give] - keep >= ratio or (me.total_resources >= 8 and hand[r_give] >= ratio):
                return Action(
                    action_type=ActionType.TRADE_BANK,
                    trade_offer=TradeOffer(
                        give_resource=r_give,
                        give_amount=ratio,
                        want_resource=r_want,
                        want_amount=1,
                    ),
                    reason=f"Comercio marítimo/banca ({ratio}:1): doy {ratio} {r_give} por 1 {r_want}",
                )

        return None

    # ------------------------------------------------------------------
    # Interacción social: Respuesta reactiva a ofertas 1x1 (Fase 1 MVP)
    # ------------------------------------------------------------------
    def evaluate_trade_cfp(
        self,
        cfp_message: ACLMessage,
        percept: Percept,
    ) -> tuple[bool, float | None]:
        """Acepta un intercambio 1x1 si necesita lo ofertado y le sobra lo pedido.

        IMPORTANTE (Límite explícito de Fase 1 MVP):
        - NO consulta los puntos de victoria del emisor (`cfp_message.sender`),
          por lo que comerciará incluso con un rival a punto de ganar (contraste
          experimental directo con las Fases 2 y 3).
        """
        self.perceive(percept)
        offer = cfp_message.content
        if not offer.is_valid_1x1():
            return False, None

        r_offered = offer.give_resource     # Lo que recibirá este agente si acepta
        r_requested = offer.want_resource   # Lo que entregará este agente si acepta
        hand = percept.self_state.resources

        if hand.get(r_requested, 0) < 1:
            return False, 0.0

        recipe = self._current_target_recipe(percept)
        needs_offered = (
            hand.get(r_offered, 0) < recipe.get(r_offered, 0)
            or (hand.get(r_offered, 0) == 0 and hand.get(r_requested, 0) >= 3)
        )
        can_spare_requested = hand.get(r_requested, 0) > recipe.get(r_requested, 0)

        if needs_offered and can_spare_requested:
            return True, 1.0

        return False, 0.0

    # ------------------------------------------------------------------
    # Utilidades internas estáticas
    # ------------------------------------------------------------------
    def _current_target_recipe(self, percept: Percept) -> dict[str, int]:
        """Determina de forma reactiva la receta de construcción inmediata."""
        me = percept.self_state
        hand = me.resources
        legal = percept.legal_moves

        if legal.cities and (
            hand["ORE"] >= 2
            or (hand["ORE"] >= 1 and hand["WHEAT"] >= 2)
            or me.settlements_left == 0
        ):
            return BUILD_COSTS["CITY"]

        if legal.settlements and me.settlements_left > 0:
            return BUILD_COSTS["SETTLEMENT"]

        if not legal.settlements and me.settlements_left > 0 and me.roads_left > 0 and legal.roads:
            return BUILD_COSTS["ROAD"]

        if legal.cities:
            return BUILD_COSTS["CITY"]

        return BUILD_COSTS["DEV_CARD"]

    @staticmethod
    def _get_own_network_nodes(percept: Percept) -> set[Any]:
        nodes: set[Any] = set(percept.self_state.settlements) | set(percept.self_state.cities)
        for v1, v2 in percept.self_state.roads:
            nodes.add(v1)
            nodes.add(v2)
        return nodes

"""Adaptador Headless y Motor de Simulación Multi-Agente sobre `catan-rl`.

Cumple estrictamente la especificación de Entorno de Simulación de la Propuesta SVRAI:
- Utiliza `catan-rl` (`catanBoard` y `player`) como motor "caja negra" intacto para el
  grafo hexagonal, puertos, mazo de desarrollo, validación espacial y actualización de estado,
  descartando por completo su GUI (`gameView.py`) y sus agentes originales (`heuristicAIPlayer.py`, `RLGame.py`).
- Abstrae la capa gráfica (`pygame`) para correr cientos de partidas rápidas por consola.
- Corrige de forma no intrusiva (en la capa adaptadora) limitaciones del repositorio base:
  descarte obligatorio con tirada de '7', cálculo exacto de Gran Ruta con ramificaciones
  y sincronización de puntos de victoria visibles vs. ocultos.
- Garantiza observabilidad parcial estricta: construye objetos `Percept` inmutables sin
  filtrar las manos de los rivales.
"""

from __future__ import annotations

import contextlib
import io
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

# 1. Inyección no intrusiva de stub para `pygame` si no está instalado (ejecución headless)
if "pygame" not in sys.modules:
    try:
        import pygame  # noqa: F401
    except ModuleNotFoundError:
        _pg_stub = types.ModuleType("pygame")
        _pg_stub.init = lambda: None  # type: ignore[attr-defined]
        sys.modules["pygame"] = _pg_stub

# 2. Carga de la API base de `catan-rl/code` manteniendo el repositorio original intacto
_CATAN_RL_CODE_DIR = Path(__file__).resolve().parent.parent / "catan-rl" / "code"
if str(_CATAN_RL_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CATAN_RL_CODE_DIR))

from board import catanBoard  # type: ignore[import-not-found]  # noqa: E402
from hexLib import polygon_corners  # type: ignore[import-not-found]  # noqa: E402
from player import player as CatanEnginePlayer  # type: ignore[import-not-found]  # noqa: E402

from .agents.base_agent import BaseAgent  # noqa: E402
from .heuristics import can_afford, score_road_edge  # noqa: E402
from .metrics import AgentGameStats, GameMetrics  # noqa: E402
from .ontology import (  # noqa: E402
    RESOURCES,
    Action,
    ActionType,
    GamePhase,
)
from .percepts import (  # noqa: E402
    BoardView,
    HexInfo,
    LegalMoves,
    Percept,
    PublicEvent,
    PublicPlayerInfo,
    SelfState,
    VertexInfo,
)
from .protocol import TradeChannel  # noqa: E402

PLAYER_COLORS: tuple[str, ...] = ("black", "darkslateblue", "magenta4", "orange1")


@contextlib.contextmanager
def _silence_engine(enabled: bool = True):
    """Silencia los `print()` internos del motor `catan-rl` para simulaciones rápidas."""
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class CatanSimulationEnvironment:
    """Entorno de simulación de Catán para 4 agentes autónomos."""

    def __init__(
        self,
        agents: list[BaseAgent],
        max_points: int = 10,
        max_rounds: int = 250,
        max_actions_per_turn: int = 25,
        seed: int | None = None,
        verbose: bool = False,
    ) -> None:
        if len(agents) != 4:
            raise ValueError(f"El escenario requiere exactamente 4 agentes (recibidos: {len(agents)}).")
        if len({a.name for a in agents}) != 4:
            raise ValueError("Los 4 agentes deben tener nombres identificadores únicos.")

        self.agents: list[BaseAgent] = agents
        self.agents_by_name: dict[str, BaseAgent] = {a.name: a for a in agents}
        self.max_points = max_points
        self.max_rounds = max_rounds
        self.max_actions_per_turn = max_actions_per_turn
        self.seed = seed
        self.verbose = verbose

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        with _silence_engine(True):
            self.board = catanBoard()
            self.engine_players: dict[str, CatanEnginePlayer] = {}
            for idx, agent in enumerate(self.agents):
                eng_p = CatanEnginePlayer(agent.name, PLAYER_COLORS[idx])
                # Recursos exactos para construir los 2 poblados y 2 caminos iniciales
                eng_p.resources = {
                    "ORE": 0,
                    "BRICK": 4,
                    "WHEAT": 2,
                    "WOOD": 4,
                    "SHEEP": 2,
                }
                self.engine_players[agent.name] = eng_p

        # Cachear vértices geométricos de cada hexágono para consultas O(1)
        self._hex_vertices: dict[int, tuple[Any, ...]] = {
            h_idx: tuple(polygon_corners(self.board.flat, h_tile.hex))
            for h_idx, h_tile in self.board.hexTileDict.items()
        }

        self.trade_channel = TradeChannel()
        self.public_events: list[PublicEvent] = []
        self._last_seen_event_idx: dict[str, int] = {a.name: 0 for a in agents}
        self.turn_number: int = 0
        self.round_number: int = 0
        self.last_dice_roll: int | None = None
        self.current_player_name: str = self.agents[0].name

    # ------------------------------------------------------------------
    # Ciclo completo de una partida
    # ------------------------------------------------------------------
    def run_game(self, game_id: int = 1) -> GameMetrics:
        """Ejecuta una partida completa (Setup serpiente + rondas por turnos hasta 10 PV)."""
        t0 = time.perf_counter()
        for agent in self.agents:
            agent.reset_for_new_game()

        metrics = GameMetrics(
            game_id=game_id,
            seed=self.seed,
            winner=None,
            winner_type=None,
            per_agent={
                a.name: AgentGameStats(agent_name=a.name, agent_type=a.agent_type)
                for a in self.agents
            },
        )

        self._log_header(f"INICIO DE PARTIDA #{game_id} (Semilla: {self.seed})")
        self._run_initial_setup_phase()

        game_over = False
        while not game_over and self.round_number < self.max_rounds:
            self.round_number += 1
            for agent in self.agents:
                self.turn_number += 1
                self.current_player_name = agent.name
                eng_p = self.engine_players[agent.name]
                metrics.per_agent[agent.name].turns_taken += 1

                # Actualizar cartas de desarrollo compradas en el turno previo
                eng_p.updateDevCards()
                eng_p.devCardPlayedThisTurn = False

                # 1. Tirada de dados y actualización del entorno
                dice_roll = self._roll_dice()
                self.last_dice_roll = dice_roll
                self._record_event("DICE_ROLLED", agent.name, {"roll": dice_roll})
                self._log(
                    f"\n[Ronda {self.round_number:02d} | Turno {self.turn_number:03d}] "
                    f"Juega {agent.name} ({eng_p.victoryPoints} PV) -> Dados: {dice_roll}"
                )

                if dice_roll != 7:
                    self._distribute_resources(dice_roll)
                else:
                    self._handle_seven_roll(agent, metrics)

                # 2. Bucle Percepción -> Decisión -> Acción durante el turno del agente
                actions_taken = 0
                had_trade_this_turn = False

                while actions_taken < self.max_actions_per_turn:
                    if self._check_victory(agent, metrics):
                        game_over = True
                        break

                    percept = self.build_percept(agent.name, GamePhase.TURN_PLAY)
                    action = agent.step(percept)
                    actions_taken += 1

                    if action.action_type == ActionType.END_TURN:
                        break

                    if action.action_type == ActionType.PROPOSE_PLAYER_TRADE:
                        had_trade_this_turn = True

                    self._execute_turn_action(agent, action, metrics)

                    if self._check_victory(agent, metrics):
                        game_over = True
                        break

                if had_trade_this_turn:
                    metrics.trade_turns += 1

                if game_over:
                    break

        # Cierre estadístico de la partida
        metrics.total_turns = self.turn_number
        metrics.total_rounds = self.round_number
        metrics.total_messages = len(self.trade_channel.message_log)
        for name, eng_p in self.engine_players.items():
            metrics.per_agent[name].final_victory_points = eng_p.victoryPoints

        for msg in self.trade_channel.message_log:
            if msg.sender in metrics.per_agent:
                metrics.per_agent[msg.sender].messages_sent += 1

        metrics.elapsed_seconds = time.perf_counter() - t0
        if metrics.winner:
            self._log_header(
                f"¡VICTORIA DE {metrics.winner} ({metrics.winner_type}) EN "
                f"{metrics.total_rounds} RONDAS ({metrics.total_turns} TURNOS)!"
            )
        else:
            self._log_header(
                f"LÍMITE DE {self.max_rounds} RONDAS ALCANZADO SIN GANADOR A {self.max_points} PV."
            )

        return metrics

    # ------------------------------------------------------------------
    # Fase de colocación inicial (orden directo 1..4 y orden inverso 4..1)
    # ------------------------------------------------------------------
    def _run_initial_setup_phase(self) -> None:
        # Primera vuelta: 1 -> 2 -> 3 -> 4
        for agent in self.agents:
            self.current_player_name = agent.name
            self._run_single_setup_placement(agent, collect_initial_resources=False)

        # Segunda vuelta (serpiente): 4 -> 3 -> 2 -> 1 + recolección inicial
        for agent in reversed(self.agents):
            self.current_player_name = agent.name
            self._run_single_setup_placement(agent, collect_initial_resources=True)

    def _run_single_setup_placement(
        self,
        agent: BaseAgent,
        collect_initial_resources: bool,
    ) -> None:
        eng_p = self.engine_players[agent.name]

        # 1. Poblado inicial mediante bucle percibir -> decidir -> actuar
        p_settle = self.build_percept(agent.name, GamePhase.SETUP_SETTLEMENT)
        a_settle = agent.step(p_settle)
        legal_settlements = p_settle.legal_moves.settlements
        v_chosen = (
            a_settle.vertex
            if a_settle.action_type == ActionType.BUILD_SETTLEMENT
            and a_settle.vertex in legal_settlements
            else legal_settlements[0]
        )
        with _silence_engine(True):
            eng_p.build_settlement(v_chosen, self.board)
        self._sync_visible_vp(eng_p)

        # 2. Camino inicial adyacente al último poblado
        p_road = self.build_percept(agent.name, GamePhase.SETUP_ROAD)
        a_road = agent.step(p_road)
        legal_roads = p_road.legal_moves.roads
        e_chosen = (
            a_road.edge
            if a_road.action_type == ActionType.BUILD_ROAD and a_road.edge in legal_roads
            else legal_roads[0]
        )
        with _silence_engine(True):
            eng_p.build_road(e_chosen[0], e_chosen[1], self.board)
        eng_p.maxRoadLength = self._compute_longest_road(eng_p)

        v_idx = self.board.boardGraph[v_chosen].vertexIndex
        self._log(f"  [Setup] {agent.name} coloca poblado en v#{v_idx} y carretera inicial.")

        # 3. Recursos iniciales adyacentes al segundo poblado
        if collect_initial_resources:
            gained: dict[str, int] = {}
            for adj_hex in self.board.boardGraph[v_chosen].adjacentHexList:
                res_type = self.board.hexTileDict[adj_hex].resource.type
                if res_type != "DESERT":
                    eng_p.resources[res_type] += 1
                    gained[res_type] = gained.get(res_type, 0) + 1
            self._record_event("INITIAL_RESOURCES", agent.name, {"resources": gained})

    # ------------------------------------------------------------------
    # Mecánicas de dados, producción y ladrón (7)
    # ------------------------------------------------------------------
    @staticmethod
    def _roll_dice() -> int:
        return int(np.random.randint(1, 7) + np.random.randint(1, 7))

    def _distribute_resources(self, dice_roll: int) -> None:
        hexes_rolled = self.board.getHexResourceRolled(dice_roll)
        for agent in self.agents:
            eng_p = self.engine_players[agent.name]
            gained: dict[str, int] = {}

            for v_coord in eng_p.buildGraph["SETTLEMENTS"]:
                for adj_hex in self.board.boardGraph[v_coord].adjacentHexList:
                    if adj_hex in hexes_rolled and not self.board.hexTileDict[adj_hex].robber:
                        res = self.board.hexTileDict[adj_hex].resource.type
                        eng_p.resources[res] += 1
                        gained[res] = gained.get(res, 0) + 1

            for v_coord in eng_p.buildGraph["CITIES"]:
                for adj_hex in self.board.boardGraph[v_coord].adjacentHexList:
                    if adj_hex in hexes_rolled and not self.board.hexTileDict[adj_hex].robber:
                        res = self.board.hexTileDict[adj_hex].resource.type
                        eng_p.resources[res] += 2
                        gained[res] = gained.get(res, 0) + 2

            if gained:
                self._record_event("RESOURCES_PRODUCED", agent.name, {"resources": gained})
                self._log(f"    + {agent.name} recolecta {gained}")

    def _handle_seven_roll(self, active_agent: BaseAgent, metrics: GameMetrics) -> None:
        # 1. Fase de descarte obligatorio para todo jugador con > 7 cartas
        for agent in self.agents:
            eng_p = self.engine_players[agent.name]
            total_cards = sum(eng_p.resources.values())
            if total_cards > 7:
                to_discard_count = total_cards // 2
                p_disc = self.build_percept(
                    agent.name, GamePhase.DISCARD, cards_to_discard=to_discard_count
                )
                action = agent.step(p_disc)
                discarded = self._validate_and_apply_discard(
                    eng_p, action.discard_resources, to_discard_count
                )
                metrics.per_agent[agent.name].cards_discarded_to_robber += to_discard_count
                metrics.total_cards_discarded += to_discard_count
                self._record_event(
                    "CARDS_DISCARDED",
                    agent.name,
                    {"count": to_discard_count, "resources": discarded},
                )
                self._log(
                    f"    ! {agent.name} tenía {total_cards} cartas y descarta "
                    f"{to_discard_count}: {discarded}"
                )

        # 2. Movimiento del ladrón por el agente activo
        p_rob = self.build_percept(active_agent.name, GamePhase.MOVE_ROBBER)
        rob_action = active_agent.step(p_rob)
        self._apply_robber_move(active_agent.name, rob_action.hex_index, rob_action.target_player)

    def _validate_and_apply_discard(
        self,
        eng_p: CatanEnginePlayer,
        proposed: dict[str, int],
        required_count: int,
    ) -> dict[str, int]:
        applied: dict[str, int] = {}
        remaining = required_count

        # Aplicar primero la selección válida del agente
        for r, amt in proposed.items():
            if r in RESOURCES and amt > 0 and remaining > 0:
                take = min(amt, eng_p.resources[r], remaining)
                if take > 0:
                    eng_p.resources[r] -= take
                    applied[r] = applied.get(r, 0) + take
                    remaining -= take

        # Salvaguarda por si un agente experimental devolviera menos cartas de las exigidas
        while remaining > 0:
            available = [r for r in RESOURCES if eng_p.resources[r] > 0]
            if not available:
                break
            r_pick = max(available, key=lambda r: eng_p.resources[r])
            eng_p.resources[r_pick] -= 1
            applied[r_pick] = applied.get(r_pick, 0) + 1
            remaining -= 1

        return applied

    def _apply_robber_move(
        self,
        actor_name: str,
        hex_index: int | None,
        target_player_name: str | None,
    ) -> None:
        eng_actor = self.engine_players[actor_name]
        robber_spots = self.board.get_robber_spots()
        chosen_hex = hex_index if hex_index in robber_spots else next(iter(robber_spots.keys()))
        self.board.updateBoardGraph_robber(chosen_hex)

        valid_targets = self._get_robber_targets_on_hex(chosen_hex, exclude_player=actor_name)
        victim_name: str | None = None
        if target_player_name in valid_targets:
            victim_name = target_player_name
        elif valid_targets:
            victim_name = valid_targets[0]

        stolen_happened = False
        if victim_name is not None:
            victim_eng = self.engine_players[victim_name]
            if sum(victim_eng.resources.values()) > 0:
                with _silence_engine(True):
                    eng_actor.steal_resource(victim_eng)
                stolen_happened = True

        self._record_event(
            "ROBBER_MOVED",
            actor_name,
            {"hex": chosen_hex, "victim": victim_name, "stole_card": stolen_happened},
        )
        self._log(
            f"    * {actor_name} mueve el ladrón al hex #{chosen_hex}"
            + (f" y roba 1 carta a {victim_name}" if stolen_happened else "")
        )

    # ------------------------------------------------------------------
    # Ejecución validada de acciones de turno
    # ------------------------------------------------------------------
    def _execute_turn_action(
        self,
        agent: BaseAgent,
        action: Action,
        metrics: GameMetrics,
    ) -> None:
        eng_p = self.engine_players[agent.name]
        atype = action.action_type

        if atype == ActionType.BUILD_SETTLEMENT:
            legal_v = self._get_legal_settlements(eng_p)
            if (
                action.vertex in legal_v
                and eng_p.settlementsLeft > 0
                and can_afford(eng_p.resources, "SETTLEMENT")
            ):
                with _silence_engine(True):
                    eng_p.build_settlement(action.vertex, self.board)
                self._sync_visible_vp(eng_p)
                metrics.per_agent[agent.name].settlements_built += 1
                # Un nuevo poblado puede cortar la carretera más larga de un rival
                self._recompute_all_road_lengths_and_award()
                v_idx = self.board.boardGraph[action.vertex].vertexIndex
                self._record_event("BUILT_SETTLEMENT", agent.name, {"vertex_index": v_idx})
                self._log(
                    f"    [Acción] {agent.name} construye POBLADO en v#{v_idx} "
                    f"({eng_p.victoryPoints} PV)"
                )

        elif atype == ActionType.BUILD_CITY:
            legal_c = tuple(self.board.get_potential_cities(eng_p).keys())
            if (
                action.vertex in legal_c
                and eng_p.citiesLeft > 0
                and can_afford(eng_p.resources, "CITY")
            ):
                with _silence_engine(True):
                    eng_p.build_city(action.vertex, self.board)
                self._sync_visible_vp(eng_p)
                metrics.per_agent[agent.name].cities_built += 1
                v_idx = self.board.boardGraph[action.vertex].vertexIndex
                self._record_event("BUILT_CITY", agent.name, {"vertex_index": v_idx})
                self._log(
                    f"    [Acción] {agent.name} mejora a CIUDAD en v#{v_idx} "
                    f"({eng_p.victoryPoints} PV)"
                )

        elif atype == ActionType.BUILD_ROAD:
            legal_r = self._get_legal_roads(eng_p)
            edge_match = self._match_legal_edge(action.edge, legal_r)
            if edge_match is not None and eng_p.roadsLeft > 0 and can_afford(eng_p.resources, "ROAD"):
                with _silence_engine(True):
                    eng_p.build_road(edge_match[0], edge_match[1], self.board)
                metrics.per_agent[agent.name].roads_built += 1
                self._recompute_all_road_lengths_and_award()
                self._record_event("BUILT_ROAD", agent.name, {})
                self._log(
                    f"    [Acción] {agent.name} construye CAMINO "
                    f"(long. máx: {eng_p.maxRoadLength})"
                )

        elif atype == ActionType.BUY_DEV_CARD:
            if (
                sum(self.board.devCardStack.values()) > 0
                and can_afford(eng_p.resources, "DEV_CARD")
            ):
                with _silence_engine(True):
                    eng_p.draw_devCard(self.board)
                self._sync_visible_vp(eng_p)
                metrics.per_agent[agent.name].dev_cards_bought += 1
                self._record_event("BOUGHT_DEV_CARD", agent.name, {})
                self._log(
                    f"    [Acción] {agent.name} compra CARTA DE DESARROLLO "
                    f"({eng_p.victoryPoints} PV)"
                )

        elif atype == ActionType.PLAY_DEV_CARD:
            self._execute_play_dev_card(agent, action)

        elif atype == ActionType.TRADE_BANK:
            offer = action.trade_offer
            if offer is not None and offer.give_resource in RESOURCES and offer.want_resource in RESOURCES:
                before_want = eng_p.resources[offer.want_resource]
                with _silence_engine(True):
                    eng_p.trade_with_bank(offer.give_resource, offer.want_resource)
                if eng_p.resources[offer.want_resource] > before_want:
                    metrics.per_agent[agent.name].bank_trades += 1
                    self._record_event(
                        "BANK_TRADE",
                        agent.name,
                        {
                            "give_resource": offer.give_resource,
                            "give_amount": offer.give_amount,
                            "want_resource": offer.want_resource,
                        },
                    )
                    self._log(
                        f"    [Banca] {agent.name} cambia {offer.give_amount} "
                        f"{offer.give_resource} -> 1 {offer.want_resource}"
                    )

        elif atype == ActionType.PROPOSE_PLAYER_TRADE:
            self._execute_player_trade_proposal(agent, action, metrics)

    def _execute_play_dev_card(self, agent: BaseAgent, action: Action) -> None:
        eng_p = self.engine_players[agent.name]
        card = action.dev_card
        if (
            eng_p.devCardPlayedThisTurn
            or card is None
            or card == "VP"
            or eng_p.devCards.get(card, 0) < 1
        ):
            return

        eng_p.devCards[card] -= 1
        eng_p.devCardPlayedThisTurn = True

        if card == "KNIGHT":
            eng_p.knightsPlayed += 1
            self._apply_robber_move(
                agent.name,
                action.dev_card_args.get("hex_index"),
                action.dev_card_args.get("target_player"),
            )
            self._check_and_update_largest_army(eng_p)
            self._record_event("PLAYED_KNIGHT", agent.name, {"knights": eng_p.knightsPlayed})

        elif card == "ROADBUILDER":
            for _ in range(2):
                if eng_p.roadsLeft <= 0:
                    break
                legal_r = self._get_legal_roads(eng_p)
                if not legal_r:
                    break
                p_view = self.build_percept(agent.name, GamePhase.TURN_PLAY)
                own_nodes = set(p_view.self_state.settlements) | set(p_view.self_state.cities)
                for v1, v2 in p_view.self_state.roads:
                    own_nodes.add(v1)
                    own_nodes.add(v2)
                best_edge = max(
                    legal_r,
                    key=lambda e: score_road_edge(e, p_view.board, agent.name, own_nodes),
                )
                # Carta gratuita: abonar temporalmente coste para reutilizar API `build_road`
                eng_p.resources["BRICK"] += 1
                eng_p.resources["WOOD"] += 1
                with _silence_engine(True):
                    eng_p.build_road(best_edge[0], best_edge[1], self.board)
            self._recompute_all_road_lengths_and_award()
            self._record_event("PLAYED_ROADBUILDER", agent.name, {})
            self._log(f"    [Carta] {agent.name} juega CONSTRUCCIÓN DE CARRETERAS")

        elif card == "YEAROFPLENTY":
            picks = action.dev_card_args.get("resources", ("ORE", "WHEAT"))
            chosen_list = [r for r in picks if r in RESOURCES][:2]
            while len(chosen_list) < 2:
                chosen_list.append("ORE")
            for r in chosen_list:
                eng_p.resources[r] += 1
            self._record_event("PLAYED_YEAROFPLENTY", agent.name, {"resources": chosen_list})
            self._log(f"    [Carta] {agent.name} juega INVENTO y toma {chosen_list}")

        elif card == "MONOPOLY":
            res = action.dev_card_args.get("resource", "ORE")
            if res not in RESOURCES:
                res = "ORE"
            total_taken = 0
            for other_name, other_eng in self.engine_players.items():
                if other_name != agent.name:
                    amt = other_eng.resources[res]
                    other_eng.resources[res] = 0
                    eng_p.resources[res] += amt
                    total_taken += amt
            self._record_event("PLAYED_MONOPOLY", agent.name, {"resource": res, "taken": total_taken})
            self._log(
                f"    [Carta] {agent.name} juega MONOPOLIO sobre {res} (captura {total_taken})"
            )

    def _execute_player_trade_proposal(
        self,
        initiator: BaseAgent,
        action: Action,
        metrics: GameMetrics,
    ) -> None:
        offer = action.trade_offer
        eng_init = self.engine_players[initiator.name]
        if (
            offer is None
            or offer.give_resource not in RESOURCES
            or offer.want_resource not in RESOURCES
            or offer.give_resource == offer.want_resource
            or offer.give_amount < 1
            or offer.want_amount < 1
            or eng_init.resources[offer.give_resource] < offer.give_amount
        ):
            return

        contractors = [a for a in self.agents if a.name != initiator.name]
        metrics.proposals_initiated += 1
        metrics.per_agent[initiator.name].trades_initiated += 1

        resolution = self.trade_channel.run_trade_negotiation(
            initiator=initiator,
            contractors=contractors,
            offer=offer,
            turn_number=self.turn_number,
            percept_provider=lambda name: self.build_percept(name, GamePhase.TURN_PLAY),
        )

        if resolution.accepted and resolution.partner is not None:
            eng_partner = self.engine_players[resolution.partner]
            if (
                eng_init.resources[offer.give_resource] >= offer.give_amount
                and eng_partner.resources[offer.want_resource] >= offer.want_amount
            ):
                eng_init.resources[offer.give_resource] -= offer.give_amount
                eng_init.resources[offer.want_resource] += offer.want_amount
                eng_partner.resources[offer.want_resource] -= offer.want_amount
                eng_partner.resources[offer.give_resource] += offer.give_amount

                metrics.proposals_accepted += 1
                metrics.per_agent[initiator.name].trades_accepted_as_initiator += 1
                metrics.per_agent[resolution.partner].trades_accepted_as_partner += 1

                self._record_event(
                    "PLAYER_TRADE",
                    initiator.name,
                    {
                        "partner": resolution.partner,
                        "give_resource": offer.give_resource,
                        "give_amount": offer.give_amount,
                        "want_resource": offer.want_resource,
                        "want_amount": offer.want_amount,
                    },
                )
                self._log(
                    f"    [Trueque ACEPTADO] {initiator.name} cambia "
                    f"{offer.give_amount} {offer.give_resource} <-> "
                    f"{offer.want_amount} {offer.want_resource} con {resolution.partner}"
                )
                return

        metrics.proposals_rejected += 1
        self._log(
            f"    [Trueque rechazado] {initiator.name} ofreció "
            f"{offer.give_amount} {offer.give_resource} por {offer.want_amount} {offer.want_resource}"
        )

    # ------------------------------------------------------------------
    # Construcción de Percepciones con Información Imperfecta
    # ------------------------------------------------------------------
    def build_percept(
        self,
        agent_name: str,
        phase: GamePhase,
        cards_to_discard: int = 0,
    ) -> Percept:
        eng_me = self.engine_players[agent_name]
        board_view = self._build_board_view()

        self_state = SelfState(
            name=agent_name,
            resources=dict(eng_me.resources),
            victory_points=eng_me.victoryPoints,
            visible_victory_points=eng_me.visibleVictoryPoints,
            dev_cards=dict(eng_me.devCards),
            new_dev_cards=tuple(eng_me.newDevCards),
            dev_card_played_this_turn=eng_me.devCardPlayedThisTurn,
            roads_left=eng_me.roadsLeft,
            settlements_left=eng_me.settlementsLeft,
            cities_left=eng_me.citiesLeft,
            ports=tuple(eng_me.portList),
            knights_played=eng_me.knightsPlayed,
            max_road_length=eng_me.maxRoadLength,
            longest_road_flag=eng_me.longestRoadFlag,
            largest_army_flag=eng_me.largestArmyFlag,
            settlements=tuple(eng_me.buildGraph["SETTLEMENTS"]),
            cities=tuple(eng_me.buildGraph["CITIES"]),
            roads=tuple(eng_me.buildGraph["ROADS"]),
        )

        public_players: dict[str, PublicPlayerInfo] = {}
        for other_name, eng_other in self.engine_players.items():
            if other_name == agent_name:
                continue
            hidden_devs = sum(eng_other.devCards.values()) + len(eng_other.newDevCards)
            public_players[other_name] = PublicPlayerInfo(
                name=other_name,
                visible_victory_points=eng_other.visibleVictoryPoints,
                total_resource_cards=sum(eng_other.resources.values()),
                total_hidden_dev_cards=hidden_devs,
                knights_played=eng_other.knightsPlayed,
                max_road_length=eng_other.maxRoadLength,
                longest_road_flag=eng_other.longestRoadFlag,
                largest_army_flag=eng_other.largestArmyFlag,
                roads_left=eng_other.roadsLeft,
                settlements_left=eng_other.settlementsLeft,
                cities_left=eng_other.citiesLeft,
                ports=tuple(eng_other.portList),
                settlements=tuple(eng_other.buildGraph["SETTLEMENTS"]),
                cities=tuple(eng_other.buildGraph["CITIES"]),
                roads=tuple(eng_other.buildGraph["ROADS"]),
            )

        legal_moves = self._build_legal_moves(eng_me, phase, cards_to_discard)

        last_idx = self._last_seen_event_idx[agent_name]
        recent_events = tuple(self.public_events[last_idx:])
        self._last_seen_event_idx[agent_name] = len(self.public_events)

        return Percept(
            agent_name=agent_name,
            turn_number=self.turn_number,
            round_number=self.round_number,
            current_player=self.current_player_name,
            phase=phase,
            last_dice_roll=self.last_dice_roll,
            self_state=self_state,
            public_players=public_players,
            board=board_view,
            legal_moves=legal_moves,
            recent_events=recent_events,
        )

    def _build_board_view(self) -> BoardView:
        hexes: dict[int, HexInfo] = {}
        for h_idx, h_tile in self.board.hexTileDict.items():
            num = int(h_tile.resource.num) if h_tile.resource.num is not None else None
            hexes[h_idx] = HexInfo(
                index=h_idx,
                resource_type=h_tile.resource.type,
                dice_num=num,
                has_robber=bool(h_tile.robber),
                vertices=self._hex_vertices[h_idx],
            )

        vertices: dict[Any, VertexInfo] = {}
        for v_coord, v_obj in self.board.boardGraph.items():
            owner_obj = v_obj.state["Player"]
            owner_name = owner_obj.name if owner_obj is not None else None
            if v_obj.state["City"]:
                b_type: str | None = "CITY"
            elif v_obj.state["Settlement"]:
                b_type = "SETTLEMENT"
            else:
                b_type = None

            vertices[v_coord] = VertexInfo(
                coord=v_coord,
                vertex_index=v_obj.vertexIndex,
                adjacent_hexes=tuple(v_obj.adjacentHexList),
                neighbors=tuple(v_obj.edgeList),
                owner=owner_name,
                building=b_type,
                port=v_obj.port if v_obj.port else None,
                is_colonised=bool(v_obj.isColonised),
            )

        roads_public: list[tuple[Any, Any, str]] = []
        for p_name, eng_p in self.engine_players.items():
            for v1, v2 in eng_p.buildGraph["ROADS"]:
                roads_public.append((v1, v2, p_name))

        return BoardView(
            hexes=hexes,
            vertices=vertices,
            roads=tuple(roads_public),
            dev_cards_remaining=sum(self.board.devCardStack.values()),
        )

    def _build_legal_moves(
        self,
        eng_p: CatanEnginePlayer,
        phase: GamePhase,
        cards_to_discard: int,
    ) -> LegalMoves:
        if phase == GamePhase.SETUP_SETTLEMENT:
            return LegalMoves(settlements=tuple(self.board.get_setup_settlements(eng_p).keys()))

        if phase == GamePhase.SETUP_ROAD:
            return LegalMoves(roads=tuple(self.board.get_setup_roads(eng_p).keys()))

        if phase == GamePhase.DISCARD:
            return LegalMoves(cards_to_discard=cards_to_discard)

        robber_hexes = tuple(self.board.get_robber_spots().keys())
        robber_targets = {
            h_idx: self._get_robber_targets_on_hex(h_idx, exclude_player=eng_p.name)
            for h_idx in robber_hexes
        }

        if phase == GamePhase.MOVE_ROBBER:
            return LegalMoves(
                robber_hexes=robber_hexes,
                robber_targets_by_hex=robber_targets,
            )

        # Fase TURN_PLAY
        legal_settlements = self._get_legal_settlements(eng_p) if eng_p.settlementsLeft > 0 else ()
        legal_cities = (
            tuple(self.board.get_potential_cities(eng_p).keys()) if eng_p.citiesLeft > 0 else ()
        )
        legal_roads = self._get_legal_roads(eng_p) if eng_p.roadsLeft > 0 else ()

        bank_ratios: dict[str, int] = {}
        for r in RESOURCES:
            if f"2:1 {r}" in eng_p.portList:
                bank_ratios[r] = 2
            elif "3:1 PORT" in eng_p.portList:
                bank_ratios[r] = 3
            else:
                bank_ratios[r] = 4

        can_buy_dev = (
            sum(self.board.devCardStack.values()) > 0
            and can_afford(eng_p.resources, "DEV_CARD")
        )
        playable_devs: tuple[str, ...] = ()
        if not eng_p.devCardPlayedThisTurn:
            playable_devs = tuple(
                card
                for card, amt in eng_p.devCards.items()
                if card != "VP" and amt >= 1
            )

        return LegalMoves(
            settlements=legal_settlements,
            cities=legal_cities,
            roads=legal_roads,
            robber_hexes=robber_hexes,
            robber_targets_by_hex=robber_targets,
            bank_trade_ratios=bank_ratios,
            can_buy_dev_card=can_buy_dev,
            playable_dev_cards=playable_devs,
        )

    # ------------------------------------------------------------------
    # Validaciones espaciales y bonificaciones (Gran Ruta / Gran Ejército)
    # ------------------------------------------------------------------
    def _get_legal_settlements(self, eng_p: CatanEnginePlayer) -> tuple[Any, ...]:
        """Versión segura de `board.get_potential_settlements` sin el bug de `break`."""
        colonisable: list[Any] = []
        seen: set[Any] = set()
        for v1, v2 in eng_p.buildGraph["ROADS"]:
            for v_coord in (v1, v2):
                if v_coord in seen:
                    continue
                seen.add(v_coord)
                v_obj = self.board.boardGraph[v_coord]
                if v_obj.isColonised:
                    continue
                if all(not self.board.boardGraph[nbr].isColonised for nbr in v_obj.edgeList):
                    colonisable.append(v_coord)
        return tuple(colonisable)

    def _get_legal_roads(self, eng_p: CatanEnginePlayer) -> tuple[tuple[Any, Any], ...]:
        return tuple(self.board.get_potential_roads(eng_p).keys())

    @staticmethod
    def _match_legal_edge(
        proposed_edge: tuple[Any, Any] | None,
        legal_roads: tuple[tuple[Any, Any], ...],
    ) -> tuple[Any, Any] | None:
        if proposed_edge is None:
            return None
        if proposed_edge in legal_roads:
            return proposed_edge
        rev = (proposed_edge[1], proposed_edge[0])
        if rev in legal_roads:
            return rev
        return None

    def _get_robber_targets_on_hex(
        self,
        hex_index: int,
        exclude_player: str,
    ) -> tuple[str, ...]:
        players_dict = self.board.get_players_to_rob(hex_index)
        targets: list[str] = []
        for p_obj in players_dict.keys():
            if p_obj.name != exclude_player and sum(p_obj.resources.values()) > 0:
                targets.append(p_obj.name)
        return tuple(targets)

    def _compute_longest_road(self, eng_p: CatanEnginePlayer) -> int:
        """DFS exacto con backtracking sobre las aristas del jugador, respetando cortes rivales."""
        roads = eng_p.buildGraph["ROADS"]
        if not roads:
            return 0

        adj: dict[Any, list[tuple[Any, int]]] = {}
        for idx, (u, v) in enumerate(roads):
            adj.setdefault(u, []).append((v, idx))
            adj.setdefault(v, []).append((u, idx))

        def dfs(curr_v: Any, used_edges: set[int]) -> int:
            best = 0
            for nxt_v, edge_id in adj.get(curr_v, ()):
                if edge_id in used_edges:
                    continue
                used_edges.add(edge_id)
                # Un asentamiento rival en `nxt_v` impide continuar atravesándolo
                nxt_owner = self.board.boardGraph[nxt_v].state["Player"]
                if nxt_owner in (None, eng_p):
                    best = max(best, 1 + dfs(nxt_v, used_edges))
                else:
                    best = max(best, 1)
                used_edges.remove(edge_id)
            return best

        return max(dfs(v, set()) for v in adj)

    def _recompute_all_road_lengths_and_award(self) -> None:
        for eng_p in self.engine_players.values():
            eng_p.maxRoadLength = self._compute_longest_road(eng_p)

        holder: CatanEnginePlayer | None = None
        for eng_p in self.engine_players.values():
            if eng_p.longestRoadFlag:
                holder = eng_p
                break

        eligible = [p for p in self.engine_players.values() if p.maxRoadLength >= 5]
        if not eligible:
            if holder is not None:
                holder.longestRoadFlag = False
                holder.victoryPoints -= 2
                self._sync_visible_vp(holder)
            return

        max_len = max(p.maxRoadLength for p in eligible)
        leaders = [p for p in eligible if p.maxRoadLength == max_len]

        if holder in leaders:
            return  # El poseedor actual conserva Gran Ruta en caso de empate

        if holder is not None:
            holder.longestRoadFlag = False
            holder.victoryPoints -= 2
            self._sync_visible_vp(holder)

        if len(leaders) == 1:
            new_holder = leaders[0]
            new_holder.longestRoadFlag = True
            new_holder.victoryPoints += 2
            self._sync_visible_vp(new_holder)
            self._record_event("LONGEST_ROAD", new_holder.name, {"length": max_len})
            self._log(f"    ★ {new_holder.name} obtiene GRAN RUTA COMERCIAL ({max_len} tramos)")

    def _check_and_update_largest_army(self, eng_p: CatanEnginePlayer) -> None:
        if eng_p.knightsPlayed < 3:
            return
        others_max = max(
            (p.knightsPlayed for p in self.engine_players.values() if p != eng_p),
            default=0,
        )
        if eng_p.knightsPlayed > others_max and not eng_p.largestArmyFlag:
            for p in self.engine_players.values():
                if p.largestArmyFlag:
                    p.largestArmyFlag = False
                    p.victoryPoints -= 2
                    self._sync_visible_vp(p)
            eng_p.largestArmyFlag = True
            eng_p.victoryPoints += 2
            self._sync_visible_vp(eng_p)
            self._record_event("LARGEST_ARMY", eng_p.name, {"knights": eng_p.knightsPlayed})
            self._log(
                f"    ★ {eng_p.name} obtiene GRAN EJÉRCITO DE CABALLERÍA ({eng_p.knightsPlayed})"
            )

    @staticmethod
    def _sync_visible_vp(eng_p: CatanEnginePlayer) -> None:
        eng_p.visibleVictoryPoints = eng_p.victoryPoints - eng_p.devCards.get("VP", 0)

    def _check_victory(self, agent: BaseAgent, metrics: GameMetrics) -> bool:
        eng_p = self.engine_players[agent.name]
        self._sync_visible_vp(eng_p)
        if eng_p.victoryPoints >= self.max_points:
            metrics.winner = agent.name
            metrics.winner_type = agent.agent_type
            metrics.per_agent[agent.name].won = True
            return True
        return False

    # ------------------------------------------------------------------
    # Registro de eventos públicos y trazas por consola
    # ------------------------------------------------------------------
    def _record_event(self, event_type: str, actor: str, details: dict[str, Any]) -> None:
        self.public_events.append(
            PublicEvent(
                turn_number=self.turn_number,
                event_type=event_type,
                actor=actor,
                details=details,
            )
        )

    def _log(self, text: str) -> None:
        if self.verbose:
            print(text)

    def _log_header(self, title: str) -> None:
        if self.verbose:
            print("\n" + "=" * 74)
            print(f" {title}")
            print("=" * 74)

"""Suite de pruebas automatizadas de verificación para la Fase 1 (MVP SVRAI).

Ejecución desde la carpeta `proyecto/`:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from svrai_catan.agents import ReactiveHeuristicAgent
from svrai_catan.environment import CatanSimulationEnvironment
from svrai_catan.ontology import (
    ACLMessage,
    ActionType,
    GamePhase,
    Performative,
    TradeOffer,
)


def _make_four_agents() -> list[ReactiveHeuristicAgent]:
    return [ReactiveHeuristicAgent(f"Agente_{i}") for i in range(1, 5)]


class TestCatanMVP(unittest.TestCase):
    """Pruebas de integración con `catan-rl`, información imperfecta, bucle PDA y comercio."""

    def test_board_initialization_headless(self) -> None:
        """Verifica que `catan-rl` inicializa 19 hexágonos y 54 vértices sin GUI."""
        env = CatanSimulationEnvironment(_make_four_agents(), seed=7)
        self.assertEqual(len(env.board.hexTileDict), 19)
        self.assertEqual(len(env.board.boardGraph), 54)
        robber_hexes = [h for h in env.board.hexTileDict.values() if h.robber]
        self.assertEqual(len(robber_hexes), 1)
        self.assertEqual(robber_hexes[0].resource.type, "DESERT")

    def test_imperfect_information_and_no_shared_memory(self) -> None:
        """Garantiza que `Percept` oculta manos rivales y puntos de victoria secretos."""
        agents = _make_four_agents()
        env = CatanSimulationEnvironment(agents, seed=11)
        env._run_initial_setup_phase()

        # Asignar recursos conocidos y 1 carta oculta de PV a Agente_2
        eng_p2 = env.engine_players["Agente_2"]
        eng_p2.resources = {"ORE": 3, "BRICK": 0, "WHEAT": 2, "WOOD": 0, "SHEEP": 1}
        eng_p2.devCards["VP"] = 1
        eng_p2.victoryPoints += 1
        env._sync_visible_vp(eng_p2)

        percept_p1 = env.build_percept("Agente_1", GamePhase.TURN_PLAY)
        pub_p2 = percept_p1.public_players["Agente_2"]

        # Agente_1 solo ve el total de cartas (6) y los PV visibles (2, no 3)
        self.assertEqual(pub_p2.total_resource_cards, 6)
        self.assertEqual(pub_p2.visible_victory_points, 2)
        self.assertFalse(hasattr(pub_p2, "resources"))

        # Mutar el diccionario de la propia percepción no altera el estado real del motor
        original_wood = env.engine_players["Agente_1"].resources["WOOD"]
        percept_p1.self_state.resources["WOOD"] = 999
        self.assertEqual(env.engine_players["Agente_1"].resources["WOOD"], original_wood)

    def test_reactive_trade_accepts_1x1_even_when_sender_near_victory(self) -> None:
        """Verifica el requisito del MVP: comercio 1x1 reactivo sin bloquear al líder."""
        agents = _make_four_agents()
        env = CatanSimulationEnvironment(agents, seed=21)
        env._run_initial_setup_phase()

        # Agente_1 va ganando con 9 PV visibles
        eng_p1 = env.engine_players["Agente_1"]
        eng_p1.victoryPoints = 9
        env._sync_visible_vp(eng_p1)

        # Agente_2 quiere construir CIUDAD (tiene 2 ORE, 2 WHEAT; necesita ORE y le sobra WOOD)
        eng_p2 = env.engine_players["Agente_2"]
        eng_p2.resources = {"ORE": 2, "WHEAT": 2, "WOOD": 2, "BRICK": 0, "SHEEP": 0}

        cfp = ACLMessage(
            msg_id=1,
            performative=Performative.CFP,
            sender="Agente_1",
            receiver="Agente_2",
            content=TradeOffer(
                give_resource="ORE",
                give_amount=1,
                want_resource="WOOD",
                want_amount=1,
            ),
            conversation_id=1,
        )
        percept_p2 = env.build_percept("Agente_2", GamePhase.TURN_PLAY)
        accepted, _ = agents[1].evaluate_trade_cfp(cfp, percept_p2)
        self.assertTrue(
            accepted,
            "El agente reactivo del MVP debe aceptar un 1x1 beneficioso sin mirar PV rivales.",
        )

    def test_trade_channel_message_protocol(self) -> None:
        """Comprueba la secuencia CFP -> PROPOSE/REFUSE -> ACCEPT_PROPOSAL y el trueque real."""
        agents = _make_four_agents()
        env = CatanSimulationEnvironment(agents, seed=33)
        env._run_initial_setup_phase()

        # Agente_1 ofrece 1 ORE pidiendo 1 SHEEP
        env.engine_players["Agente_1"].resources = {
            "ORE": 2, "BRICK": 1, "WOOD": 1, "WHEAT": 1, "SHEEP": 0
        }
        # Agente_3 busca ciudad (necesita ORE, le sobra SHEEP) -> aceptará
        env.engine_players["Agente_3"].resources = {
            "ORE": 2, "WHEAT": 2, "SHEEP": 2, "BRICK": 0, "WOOD": 0
        }
        # Agentes 2 y 4 sin SHEEP -> rechazarán
        env.engine_players["Agente_2"].resources = {
            "ORE": 0, "WHEAT": 0, "SHEEP": 0, "BRICK": 0, "WOOD": 0
        }
        env.engine_players["Agente_4"].resources = {
            "ORE": 0, "WHEAT": 0, "SHEEP": 0, "BRICK": 0, "WOOD": 0
        }

        resolution = env.trade_channel.run_trade_negotiation(
            initiator=agents[0],
            contractors=agents[1:],
            offer=TradeOffer("ORE", 1, "SHEEP", 1),
            turn_number=1,
            percept_provider=lambda name: env.build_percept(name, GamePhase.TURN_PLAY),
        )

        self.assertTrue(resolution.accepted)
        self.assertEqual(resolution.partner, "Agente_3")
        # 3 CFP + 3 respuestas (1 PROPOSE, 2 REFUSE) + 1 ACCEPT_PROPOSAL = 7 mensajes
        self.assertEqual(len(resolution.messages), 7)

    def test_seven_roll_discard_rule(self) -> None:
        """Comprueba que con >7 cartas al salir un 7 se descarta exactamente la mitad."""
        agents = _make_four_agents()
        env = CatanSimulationEnvironment(agents, seed=15)
        env._run_initial_setup_phase()

        # Dar 10 cartas a Agente_2 -> debe descartar 5
        env.engine_players["Agente_2"].resources = {
            "BRICK": 4, "WOOD": 3, "WHEAT": 1, "SHEEP": 1, "ORE": 1
        }
        from svrai_catan.metrics import AgentGameStats, GameMetrics

        metrics = GameMetrics(
            game_id=1,
            seed=15,
            winner=None,
            winner_type=None,
            per_agent={a.name: AgentGameStats(a.name, a.agent_type) for a in agents},
        )
        env._handle_seven_roll(agents[0], metrics)
        self.assertEqual(metrics.per_agent["Agente_2"].cards_discarded_to_robber, 5)

    def test_full_game_reaches_victory(self) -> None:
        """Ejecuta una partida completa de 4 agentes y verifica que concluye en >= 10 PV."""
        agents = _make_four_agents()
        env = CatanSimulationEnvironment(agents, seed=42, verbose=False)
        metrics = env.run_game(game_id=1)

        self.assertIsNotNone(metrics.winner)
        winner_vp = metrics.per_agent[metrics.winner].final_victory_points  # type: ignore[index]
        self.assertGreaterEqual(winner_vp, 10)
        self.assertGreater(metrics.proposals_initiated, 0)
        self.assertGreater(metrics.total_messages, 0)

    def test_pda_step_cycle(self) -> None:
        """Verifica explícitamente el ciclo percibir -> decidir -> acción en `BaseAgent.step`."""
        agents = _make_four_agents()
        env = CatanSimulationEnvironment(agents, seed=5)
        percept = env.build_percept("Agente_1", GamePhase.SETUP_SETTLEMENT)
        action = agents[0].step(percept)
        self.assertEqual(action.action_type, ActionType.BUILD_SETTLEMENT)
        self.assertIn(action.vertex, percept.legal_moves.settlements)


if __name__ == "__main__":
    unittest.main()

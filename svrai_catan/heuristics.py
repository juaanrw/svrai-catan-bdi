"""Heurísticas espaciales y de gestión estáticas para la Capa Reactiva (MVP).

Tal como establece la propuesta para el MVP:
- Prioriza intersecciones según la esperanza estática de generación de dados (pips)
  y diversidad básica de recursos, sin búsqueda de caminos costosa en grafos.
- Evalúa aristas locales (1-2 saltos) para orientar caminos hacia huecos colonizables
  y no malgastar madera/arcilla en callejones sin salida.
- Selecciona dónde colocar el ladrón evitando bloquear producción propia.
"""

from __future__ import annotations

from typing import Any

from .ontology import BUILD_COSTS, DICE_PIPS, RESOURCES
from .percepts import BoardView, Percept


def score_settlement_vertex(
    vertex_coord: Any,
    board: BoardView,
    covered_resources: set[str] | None = None,
) -> float:
    """Puntúa estáticamente un vértice candidato para construir un poblado.

    Combina:
    1. Suma de probabilidades de dados (`DICE_PIPS`, de 0 a 15 pips).
    2. Diversidad local de recursos distintos en el vértice.
    3. Bonificación si aporta un recurso aún no cubierto por el agente (clave en setup).
    4. Pequeño plus si el vértice dispone de puerto marítimo.
    """
    v_info = board.vertices[vertex_coord]
    pip_score = 0.0
    distinct_resources: set[str] = set()

    for hex_idx in v_info.adjacent_hexes:
        hex_info = board.hexes[hex_idx]
        if hex_info.resource_type != "DESERT":
            pip_score += DICE_PIPS.get(hex_info.dice_num, 0)
            distinct_resources.add(hex_info.resource_type)

    diversity_bonus = 1.5 * len(distinct_resources)

    new_resource_bonus = 0.0
    if covered_resources is not None:
        for res in distinct_resources:
            if res not in covered_resources:
                new_resource_bonus += 2.0

    port_bonus = 0.75 if v_info.port else 0.0
    return pip_score + diversity_bonus + new_resource_bonus + port_bonus


def score_city_vertex(vertex_coord: Any, board: BoardView) -> float:
    """Puntúa cuál de los poblados propios conviene más mejorar a ciudad.

    Al duplicar la recolección de recursos del vértice, prioriza la suma de pips
    con un ligero incentivo hacia trigo y mineral (motor de ciudades/desarrollo).
    """
    v_info = board.vertices[vertex_coord]
    score = 0.0
    for hex_idx in v_info.adjacent_hexes:
        hex_info = board.hexes[hex_idx]
        pips = DICE_PIPS.get(hex_info.dice_num, 0)
        weight = 1.15 if hex_info.resource_type in ("ORE", "WHEAT") else 1.0
        score += pips * weight
    return score


def is_vertex_colonisable_on_board(vertex_coord: Any, board: BoardView) -> bool:
    """Comprueba la regla de distancia de Catán sobre una vista pública del tablero."""
    v_info = board.vertices[vertex_coord]
    if v_info.is_colonised:
        return False
    return all(not board.vertices[nbr].is_colonised for nbr in v_info.neighbors)


def score_road_edge(
    edge: tuple[Any, Any],
    board: BoardView,
    agent_name: str,
    own_vertices_and_road_nodes: set[Any],
) -> float:
    """Heurística espacial local (O(1), sin pathfinding global) para elegir un camino.

    Prioriza:
    1. Aristas cuyo extremo nuevo ya es directamente colonizable (+100 + valor vértice).
    2. Aristas cuyo extremo nuevo es vecino inmediato de un vértice colonizable (+50 + valor).
    3. Extensión general hacia intersecciones con buena producción (+10 + 0.5 * valor).
    """
    v1, v2 = edge
    # Identificar cuál de los dos extremos nos proyecta hacia territorio nuevo
    if v1 in own_vertices_and_road_nodes and v2 not in own_vertices_and_road_nodes:
        frontier = v2
    elif v2 in own_vertices_and_road_nodes and v1 not in own_vertices_and_road_nodes:
        frontier = v1
    else:
        frontier = v2

    # Si el vértice frontera pertenece a un rival, el camino queda bloqueado ahí
    frontier_info = board.vertices[frontier]
    if frontier_info.owner not in (None, agent_name):
        return -10.0

    if is_vertex_colonisable_on_board(frontier, board):
        return 100.0 + score_settlement_vertex(frontier, board)

    best_next_step = 0.0
    for nbr in frontier_info.neighbors:
        if is_vertex_colonisable_on_board(nbr, board):
            best_next_step = max(best_next_step, score_settlement_vertex(nbr, board))

    if best_next_step > 0.0:
        return 50.0 + best_next_step

    return 10.0 + 0.5 * score_settlement_vertex(frontier, board)


def select_robber_placement(percept: Percept) -> tuple[int, str | None]:
    """Heurística estática para mover el ladrón sin perjudicar al propio agente.

    Elige un hexágono legal donde el propio agente NO tenga asentamientos,
    maximizando la probabilidad de tirada multiplicada por la presencia rival,
    y roba a un jugador adyacente que tenga cartas en mano.
    """
    me = percept.agent_name
    board = percept.board
    best_hex = percept.legal_moves.robber_hexes[0]
    best_score = -1e9
    best_victim: str | None = None

    for hex_idx in percept.legal_moves.robber_hexes:
        hex_info = board.hexes[hex_idx]
        pips = DICE_PIPS.get(hex_info.dice_num, 0)

        self_exposure = 0
        rival_exposure = 0
        for v_coord in hex_info.vertices:
            v_info = board.vertices[v_coord]
            if v_info.owner == me:
                self_exposure += 2 if v_info.building == "CITY" else 1
            elif v_info.owner is not None:
                rival_exposure += 2 if v_info.building == "CITY" else 1

        # Penalizar fuertemente bloquearnos a nosotros mismos
        if self_exposure > 0:
            hex_score = -100.0 * self_exposure + pips
        elif hex_info.resource_type == "DESERT":
            hex_score = -5.0
        else:
            hex_score = pips * (1.0 + 1.5 * rival_exposure)

        candidates = percept.legal_moves.robber_targets_by_hex.get(hex_idx, ())
        victim: str | None = None
        if candidates:
            # En el MVP reactivo elegimos al rival en el hexágono con más cartas en mano
            victim = max(
                candidates,
                key=lambda p_name: percept.public_players[p_name].total_resource_cards,
            )
            hex_score += 5.0

        if hex_score > best_score:
            best_score = hex_score
            best_hex = hex_idx
            best_victim = victim

    return best_hex, best_victim


def select_cards_to_discard(
    resources: dict[str, int],
    num_to_discard: int,
    target_recipe: dict[str, int],
) -> dict[str, int]:
    """Selecciona `num_to_discard` cartas priorizando descartar excedentes."""
    working = dict(resources)
    discarded: dict[str, int] = {r: 0 for r in RESOURCES}

    for _ in range(num_to_discard):
        available = [r for r in RESOURCES if working[r] > 0]
        if not available:
            break
        # Descartar primero el recurso con mayor superávit respecto a la receta objetivo
        worst_res = max(
            available,
            key=lambda r: (working[r] - target_recipe.get(r, 0), working[r]),
        )
        working[worst_res] -= 1
        discarded[worst_res] += 1

    return {r: amt for r, amt in discarded.items() if amt > 0}


def can_afford(resources: dict[str, int], item_name: str) -> bool:
    """Verifica si un diccionario de recursos cubre el coste de `item_name`."""
    cost = BUILD_COSTS[item_name]
    return all(resources.get(r, 0) >= req for r, req in cost.items())

# Coordinación y Negociación Multi-Agente sobre Catán (SVRAI)

Proyecto final de la asignatura **Síntesis, Verificación y Razonamiento de Agentes Inteligentes (SVRAI)**.

Este repositorio contiene la **Fase 1 (MVP)** funcional: un ecosistema de **4 agentes autónomos** integrados sobre el motor open-source [`catan-rl`](./catan-rl) actuando como *caja negra* de validación de reglas y tablero hexagonal, ejecutándose de forma rápida por consola (*headless*, sin dependencia de `pygame`).

---

## Estructura del Proyecto

```text
proyecto/
├── catan-rl/                        # Motor base open-source (intacto, usado como caja negra)
├── svrai_catan/                     # Paquete Python del sistema multi-agente SVRAI
│   ├── ontology.py                  # Ontología compartida: recursos, costes, acciones y actos FIPA-ACL
│   ├── percepts.py                  # Percepciones inmutables con observabilidad parcial (sin ver manos rivales)
│   ├── heuristics.py                # Heurísticas espaciales estáticas (pips de dados, caminos, ladrón, descarte)
│   ├── protocol.py                  # Canal de negociación mediante paso de mensajes (CFP / PROPOSE / REFUSE / AWARD)
│   ├── environment.py               # Adaptador headless de catan-rl y bucle del simulador de 4 jugadores
│   ├── metrics.py                   # Recolección de las 4 métricas experimentales de la propuesta
│   └── agents/
│       ├── base_agent.py            # Interfaz abstracta BaseAgent: ciclo percibir -> decidir -> actuar
│       └── reactive_agent.py        # Agente de Fase 1 (MVP): heurísticas estáticas + comercio reactivo 1x1
├── tests/
│   └── test_mvp.py                  # Suite de tests unitarios y de integración del MVP
└── run_mvp.py                       # Script CLI para ejecutar partidas trazadas o torneos por consola
```

---

## Alineación con la Asignatura (Decisiones de Diseño del MVP)

1. **Entorno con Información Imperfecta (Tema 1):** `CatanSimulationEnvironment` nunca expone las instancias internas de los rivales. Cada agente recibe un objeto `Percept` de solo lectura con su propia mano privada, el estado público del grafo y solo el **tamaño total** de la mano de cada rival (ocultando tipos de recursos y cartas de Punto de Victoria secretas).
2. **Bucle Percepción–Decisión–Acción explícito (Tema 1 y Tema 2):** `BaseAgent.step(percept)` separa `perceive(percept)` y `decide() -> Action` en todas las subfases (`SETUP_SETTLEMENT`, `SETUP_ROAD`, `DISCARD`, `MOVE_ROBBER`, `TURN_PLAY`).
3. **Capa Reactiva / Heurística por prioridad (Tema 2):** `ReactiveHeuristicAgent` organiza sus decisiones jerárquicamente (cartas de desarrollo $\rightarrow$ construcción inmediata de ciudades/poblados/caminos según probabilidad estática de dados $\rightarrow$ comercio 1x1 entre jugadores $\rightarrow$ comercio marítimo/banca $\rightarrow$ fin de turno).
4. **Ontología y Canal de Mensajes preparados para Contract Net (Tema 4):** Los intercambios circulan exclusivamente a través de `TradeChannel` usando performativas FIPA-ACL (`cfp`, `propose`, `refuse`, `accept-proposal`, `reject-proposal`). En este MVP, los agentes aceptan trueques `1x1` de forma puramente reactiva (según necesidad/excedente inmediato, **sin evaluar quién va ganando**), sirviendo como *baseline* experimental exacta frente a las Fases 2 y 3.

---

## Requisitos y Ejecución

Solo requiere **Python 3.10+** y **`numpy`** (no necesita instalar `pygame` ni `tensorflow`).

Desde la carpeta `proyecto/`:

```powershell
# 1. Ejecutar 1 partida completa mostrando toda la traza paso a paso por consola
python run_mvp.py

# 2. Ejecutar un torneo rápido de 20 partidas (modo silencioso + tabla de métricas)
python run_mvp.py --games 20 --seed 42

# 3. Ejecutar la suite de tests automatizados de verificación
python -m unittest discover -s tests -v
```

---

## Hoja de Ruta hacia Fases 2 y 3

La arquitectura del MVP deja listos todos los puntos de extensión sin necesidad de reescribir el simulador:
- **Fase 2 (BDI + Contract Net con Utilidad):** Subclasificar `BaseAgent` en `BDIAgent`, actualizando creencias sobre las manos rivales en `perceive()` a partir de `percept.recent_events` y evaluando escasez/utilidad en `evaluate_trade_cfp()` y `select_trade_partner()`.
- **Fase 3 (Teoría de Juegos / Inteligencia Maquiavélica):** Incorporar penalización estratégica en la utilidad cuando `percept.public_players[rival].visible_victory_points` se acerque a 10 PV, y lanzar torneos asimétricos (1–2 agentes BDI frente a `ReactiveHeuristicAgent`).

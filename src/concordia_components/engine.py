from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from collections.abc import Callable, Mapping, Sequence
from concordia.typing.entity import DEFAULT_ACTION_SPEC
from typing import Any
from concordia_components.type_aliases import Thread
import numpy as np

class SimEngine(engine_lib.Engine):
    """Engine interface."""

    def __init__(self,
                 threads: list[Thread] = None
                 ):
        self._threads = [] if threads is None else threads

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
    ) -> Thread:
        """Make an observation for an entity."""
        
        # Select a random thread as observation
        if np.random.rand() < 0.3 and len(self._threads) < 5:
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe([new_thread])
            return new_thread

        if not self._threads:
            self._threads.append(Thread(id=0, content=[]))
        random_thread = np.random.choice(self._threads)
        entity.observe([random_thread])
        return random_thread

    def next_acting(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
        """Return the next entity or entities to act."""
        
        # Select a random entity to act
        acting_entity = np.random.choice(entities)
        return acting_entity, DEFAULT_ACTION_SPEC

    def resolve(
        self,
        game_master: entity_lib.Entity,
        event: str,
    ) -> None:
        """Resolve the event."""
        return

    def terminate(
        self,
        game_master: entity_lib.Entity,
    ) -> bool:
        """Decide if the episode should terminate or continue."""
        return False

    def next_game_master(
        self,
        game_master: entity_lib.Entity,
        game_masters: Sequence[entity_lib.Entity],
    ) -> entity_lib.Entity:
        """Return the game master that will be responsible for the next step."""
        return game_master

    def run_loop(
        self,
        game_masters: Sequence[entity_lib.Entity],
        entities: Sequence[entity_lib.Entity],
        premise: str,
        max_steps: int,
        verbose: bool,
        log: list[Mapping[str, Any]] | None,
        checkpoint_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Run a game loop."""

        steps = 0
        game_master = game_masters[0]
        while not self.terminate(game_master) and steps < max_steps:
            if verbose:
                print(f"Step {steps}")
            acting_entity, action_spec = self.next_acting(
                game_master, entities)
            observation = self.make_observation(game_master, acting_entity)
            action = acting_entity.act(action_spec)
            observation.content.append({'role': acting_entity.name, 'content': action})
            self.resolve(game_master, action)
            game_master = self.next_game_master(game_master, game_masters)
            steps += 1

        return

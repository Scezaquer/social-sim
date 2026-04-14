
from collections.abc import Callable, Mapping
from typing import Any
import numpy as np
import copy
from simulation_components.engine import SimEngine
from simulation_components.unsloth_model import UnslothLanguageModel
from simulation_components.entities import Entity

class SocialMediaSim:
    def __init__(
      self,
      model: UnslothLanguageModel,
      entities: list[Entity] = None,
      engine: SimEngine = SimEngine(),
    ):
        """Initialize the simulation object."""
        self._model = model
        self._engine = engine
        self.game_masters = [None]
        self.entities = entities if entities is not None else []
        self._raw_log = []
        self._checkpoints_path = None
        self._checkpoint_counter = 0

    def get_raw_log(self) -> list[Mapping[str, Any]]:
        """Get the raw log of the simulation."""
        return copy.deepcopy(self._raw_log)

    def get_game_masters(self) -> list[Entity]:
        """Get the game masters.

        The function returns a copy of the game masters list to avoid modifying the
        original list. However, the game masters are not deep copied, so changes
        to the game masters will be reflected in the simulation.

        Returns:
        A list of game master entities.
        """
        return copy.copy(self.game_masters)

    def get_entities(self) -> list[Entity]:
        """Get the entities.

        The function returns a copy of the entities list to avoid modifying the
        original list. However, the entities are not deep copied, so changes
        to the entities will be reflected in the simulation.

        Returns:
        A list of entities.
        """
        return copy.copy(self.entities)

    def add_game_master(self, game_master: Entity):
        """Add a game master to the simulation."""
        raise NotImplementedError

    def add_entity(self, entity: Entity):
        """Add an entity to the simulation."""
        raise NotImplementedError

    def play(
        self,
        premise: str | None = None,
        max_steps: int | None = None,
        start_time: float | None = None,
        duration: float | None = None,
    ) -> list[Mapping[str, Any]]:
        """Run the simulation.

        Args:
        premise: A string to use as the initial premise of the simulation.
        max_steps: The maximum number of steps to run the simulation for.
        start_time: The start time of the simulation (unix timestamp).
        duration: The duration of the simulation in seconds.

        Returns:
        html_results_log: browseable log of the simulation in HTML format
        """
        
        self._engine.run_loop(
            game_masters=self.game_masters,
            entities=self.entities,
            premise=premise,
            max_steps=max_steps,
            verbose=True,
            log=self._raw_log,
            checkpoint_callback=None,
            start_time=start_time,
            duration=duration,
        )

        return copy.deepcopy(self._raw_log)

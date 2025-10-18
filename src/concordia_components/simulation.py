
from collections.abc import Callable, Mapping
from typing import Any

from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.typing import prefab as prefab_lib
from concordia.typing import simulation as simulation_lib
from concordia.environment import engine as engine_lib
from concordia.prefabs.entity import minimal
from concordia.language_model.no_language_model import NoLanguageModel
import numpy as np
import copy
from concordia_components.engine import SimEngine

Config = prefab_lib.Config
Role = prefab_lib.Role

class SocialMediaSim(simulation_lib.Simulation):
    def __init__(
      self,
      config: Config,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      entities: list[entity_lib.Entity] = None,
      engine: engine_lib.Engine = SimEngine(),
    ):
        """Initialize the simulation object."""
        self._config = config
        self._model = model
        self._embedder = embedder
        self._engine = engine
        self.game_masters = [None]
        self.entities = entities if entities is not None else []
        self._raw_log = []
        self._entity_to_prefab_config: dict[str, prefab_lib.InstanceConfig] = {}
        self._checkpoints_path = None
        self._checkpoint_counter = 0

    def get_raw_log(self) -> list[Mapping[str, Any]]:
        """Get the raw log of the simulation."""
        return copy.deepcopy(self._raw_log)

    def get_game_masters(self) -> list[entity_lib.Entity]:
        """Get the game masters.

        The function returns a copy of the game masters list to avoid modifying the
        original list. However, the game masters are not deep copied, so changes
        to the game masters will be reflected in the simulation.

        Returns:
        A list of game master entities.
        """
        return copy.copy(self.game_masters)

    def get_entities(self) -> list[entity_lib.Entity]:
        """Get the entities.

        The function returns a copy of the entities list to avoid modifying the
        original list. However, the entities are not deep copied, so changes
        to the entities will be reflected in the simulation.

        Returns:
        A list of entities.
        """
        return copy.copy(self.entities)

    def add_game_master(self, game_master: entity_lib.Entity):
        """Add a game master to the simulation."""
        raise NotImplementedError

    def add_entity(self, entity: entity_lib.Entity):
        """Add an entity to the simulation."""
        raise NotImplementedError

    def play(
        self,
        premise: str | None = None,
        max_steps: int | None = None,
    ) -> list[Mapping[str, Any]]:
        """Run the simulation.

        Args:
        premise: A string to use as the initial premise of the simulation.
        max_steps: The maximum number of steps to run the simulation for.

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
        )

        return copy.deepcopy(self._raw_log)

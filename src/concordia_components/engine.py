from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from collections.abc import Callable, Mapping, Sequence
from concordia.typing.entity import DEFAULT_ACTION_SPEC
from typing import Any
from concordia_components.type_aliases import Thread
import numpy as np
from tqdm import tqdm
import networkx as nx

class SimEngine(engine_lib.Engine):
    """Engine interface."""

    def __init__(self,
                 threads: list[Thread] = None
                 ):
        self._threads = [] if threads is None else threads
        self._entity_activity_probs = None
        self._social_graph = None
        self._acting_entities = None
        self._acting_probs = None

    def _initialize_social_context(self, entities: Sequence[entity_lib.Entity]):
        n_entities = len(entities)
        entity_names = [e.name for e in entities]
        
        # Power-law activity distribution
        ranks = np.arange(1, n_entities + 1)
        activity_weights = 1.0 / (ranks ** 1.5) # Zipf's law
        activity_weights /= activity_weights.sum()
        np.random.shuffle(activity_weights)
        self._entity_activity_probs = dict(zip(entity_names, activity_weights))
        
        # Cache for next_acting efficiency
        self._acting_entities = list(entities)
        self._acting_probs = activity_weights
        
        # Social Graph using NetworkX
        # Scale-free graph for realistic social media topology (hubs/poles)
        G = nx.scale_free_graph(n_entities, seed=42)
        G = nx.DiGraph(G) # Remove parallel edges
        G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops
        
        self._social_graph = {}
        for i, name in enumerate(entity_names):
            if i in G:
                following_indices = list(G.successors(i))
                # Ensure indices are within bounds
                following_names = {entity_names[j] for j in following_indices if j < n_entities}
                self._social_graph[name] = following_names
            else:
                self._social_graph[name] = set()

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
    ) -> Thread:
        """Make an observation for an entity."""
        
        # Select a random thread as observation
        if np.random.rand() < 0.3:
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            return new_thread

        if not self._threads:
            self._threads.append(Thread(id=0, content=[]))

        # Calculate weights inversely proportional to thread length
        thread_lengths = np.array([len(thread.content) for thread in self._threads])
        # Add 1 to avoid division by zero and ensure non-zero weights
        weights = 1.0 / (thread_lengths + 1)
        
        # Social boost
        if self._social_graph and entity.name in self._social_graph:
            following = self._social_graph[entity.name]
            social_boost = np.ones(len(self._threads))
            for i, thread in enumerate(self._threads):
                participants = {msg['role'] for msg in thread.content}
                if not following.isdisjoint(participants):
                    social_boost[i] = 10.0 # Significant boost
            weights *= social_boost

        # Zero all weights except for the last 20 threads to focus on recent activity
        if len(weights) > 20:
            weights[:-20] = 0
        # Normalize weights to sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        random_thread = np.random.choice(self._threads, p=weights)
        entity.observe(random_thread)
        return random_thread

    def next_acting(
        self,
        game_master: entity_lib.Entity,
        entities: Sequence[entity_lib.Entity],
    ) -> tuple[entity_lib.Entity, entity_lib.ActionSpec]:
        """Return the next entity or entities to act."""
        
        if self._entity_activity_probs is None:
            self._initialize_social_context(entities)

        # Use cached probabilities for efficiency
        acting_entity = np.random.choice(self._acting_entities, p=self._acting_probs)
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
        with tqdm(total=max_steps, disable=not verbose, desc="Game Steps") as pbar:
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
                pbar.update(1)

        return

from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from collections.abc import Callable, Mapping, Sequence
from concordia.typing.entity import DEFAULT_ACTION_SPEC
from typing import Any
from concordia_components.type_aliases import Thread
import numpy as np
from tqdm import tqdm
import networkx as nx
import time

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
        self._name_to_idx = None

    def _initialize_social_context(self, entities: Sequence[entity_lib.Entity]):
        n_entities = len(entities)
        entity_names = [e.name for e in entities]
        self._name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
        # Power-law activity distribution
        ranks = np.arange(1, n_entities + 1)
        activity_weights = 1.0 / (ranks ** 0.4) # Zipf's law
        activity_weights /= activity_weights.sum()
        np.random.shuffle(activity_weights)
        self._entity_activity_probs = dict(zip(entity_names, activity_weights))
        
        # Cache for next_acting efficiency
        self._acting_entities = list(entities)
        self._acting_probs = activity_weights
        
        # Social Graph using NetworkX
        # Scale-free graph for realistic social media topology (hubs/poles)
        G = nx.scale_free_graph(n_entities, seed=42)
        
        self._social_graph = [set() for _ in range(n_entities)]
        for u, v in G.edges():
            if u != v and u < n_entities and v < n_entities:
                self._social_graph[u].add(v)
        
        # Clear G to free memory
        del G

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
        make_new_thread: bool = True
    ) -> Thread:
        """Make an observation for an entity."""
        
        if make_new_thread and np.random.rand() < 0.3:
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            return new_thread

        if not self._threads:
            if make_new_thread:
                self._threads.append(Thread(id=0, content=[]))
            else:
                return None

        # Only consider the last 200 threads for efficiency
        relevant_threads = self._threads[-200:]

        # Calculate weights inversely proportional to thread length
        thread_lengths = np.array([len(thread.content) for thread in relevant_threads])
        # Add 1 to avoid division by zero and ensure non-zero weights
        weights = 1.0 / (thread_lengths + 1)

        # Prevent self-replies
        for i, thread in enumerate(relevant_threads):
            if thread.content and thread.content[-1]['role'] == entity.name:
                weights[i] = 0.0
        
        # Social and Participation boost
        boosts = np.ones(len(relevant_threads))
        
        entity_idx = self._name_to_idx.get(entity.name)
        following = set()
        if entity_idx is not None and self._social_graph:
            following = self._social_graph[entity_idx]
            
        for i, thread in enumerate(relevant_threads):
            participants_indices = set()
            for msg in thread.content:
                p_idx = self._name_to_idx.get(msg['role'])
                if p_idx is not None:
                    participants_indices.add(p_idx)
            
            # Social boost
            if not following.isdisjoint(participants_indices):
                boosts[i] *= 10.0
            
            # Participation boost
            if entity_idx in participants_indices:
                age = 20 - i # Age relative to the most recent thread
                boosts[i] *= (1.0 + (20.0 / age))
        
        weights *= boosts

        # Normalize weights to sum to 1
        weights.clip(min=0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        elif make_new_thread:
            # If no valid threads (e.g. all are self-replies), create a new one
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            return new_thread
        else:
            return None
        
        random_thread_idx = np.random.choice(len(relevant_threads), p=weights)
        random_thread = relevant_threads[random_thread_idx]
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
        idx = np.random.choice(len(self._acting_entities), p=self._acting_probs)
        acting_entity = self._acting_entities[idx]
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
        start_time: float | None = None,
        duration: float | None = None,
    ) -> None:
        """Run a game loop."""

        steps = 0
        game_master = game_masters[0]
        with tqdm(total=max_steps, disable=not verbose, desc="Game Steps") as pbar:
            while not self.terminate(game_master) and steps < max_steps:
                if start_time is not None and duration is not None:
                    if time.time() - start_time > duration:
                        if verbose:
                            print(f"Time limit of {duration} seconds reached. Stopping simulation.")
                        break

                if verbose:
                    print(f"Step {steps}")
                # 10 observations for every action
                for i in range(10):
                    acting_entity, action_spec = self.next_acting(
                        game_master, entities)
                    
                    make_new_thread = ( i == 9 )  # Only create new thread on the last observation
                    observation = self.make_observation(game_master, acting_entity, make_new_thread)
                action = acting_entity.act(action_spec)
                observation.content.append({'role': acting_entity.name, 'content': action})
                self.resolve(game_master, action)
                game_master = self.next_game_master(game_master, game_masters)
                steps += 1
                pbar.update(1)

        return

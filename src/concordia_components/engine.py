from concordia.environment import engine as engine_lib
from concordia.typing import entity as entity_lib
from collections.abc import Callable, Mapping, Sequence
from concordia.typing.entity import DEFAULT_ACTION_SPEC
from typing import Any
from concordia_components.type_aliases import Thread
from concordia_components.entities import NewsSource, User
import numpy as np
from tqdm import tqdm
import networkx as nx
import time

class SimEngine(engine_lib.Engine):
    """Engine interface."""

    def __init__(self,
                 threads: list[Thread] = None,
                 graph: Any = None,
                 survey_config: dict[str, Any] = None
                 ):
        self._threads = [] if threads is None else threads
        self._entity_activity_probs = None
        self._social_graph = None
        self._acting_entities = None
        self._acting_probs = None
        self._name_to_idx = None
        self._initial_graph = graph
        self._survey_config = survey_config
        self._step_count = 0
        self._survey_results = []

    def get_survey_results(self) -> list[dict[str, Any]]:
        """Returns the results of the surveys."""
        return self._survey_results

    def _initialize_social_context(self, entities: Sequence[entity_lib.Entity]):
        n_entities = len(entities)
        entity_names = [e.name for e in entities]
        self._name_to_idx = {name: i for i, name in enumerate(entity_names)}
        
        # Identify NewsSource
        news_source_indices = [i for i, e in enumerate(entities) if isinstance(e, NewsSource)]

        # Power-law activity distribution
        ranks = np.arange(1, n_entities + 1)
        activity_weights = 1.0 / (ranks ** 0.5) # Zipf's law
        activity_weights /= activity_weights.sum()
        
        # Assign highest weights to NewsSource
        activity_weights = np.sort(activity_weights)[::-1]
        final_weights = np.zeros(n_entities)
        
        for i, idx in enumerate(news_source_indices):
            if i < len(activity_weights):
                final_weights[idx] = activity_weights[i]
        
        remaining_indices = [i for i in range(n_entities) if i not in news_source_indices]
        remaining_weights = activity_weights[len(news_source_indices):]
        np.random.shuffle(remaining_weights)
        
        for i, idx in enumerate(remaining_indices):
            if i < len(remaining_weights):
                final_weights[idx] = remaining_weights[i]

        self._entity_activity_probs = dict(zip(entity_names, final_weights))
        
        # Cache for next_acting efficiency
        self._acting_entities = list(entities)
        self._acting_probs = final_weights
        
        # Social Graph using NetworkX
        self._social_graph = [set() for _ in range(n_entities)]

        if self._initial_graph is not None:
            G = self._initial_graph
            
            # Calculate degrees to identify hubs
            if G.is_directed():
                degree_dict = dict(G.in_degree())
            else:
                degree_dict = dict(G.degree())
                
            sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
            
            node_to_entity_idx = {}
            used_nodes = set()
            
            # Assign NewsSources to highest degree nodes
            for i, ns_idx in enumerate(news_source_indices):
                if i < len(sorted_nodes):
                    node = sorted_nodes[i]
                    node_to_entity_idx[node] = ns_idx
                    used_nodes.add(node)
            
            # Assign remaining entities
            remaining_indices = [i for i in range(n_entities) if i not in news_source_indices]
            remaining_nodes = [n for n in sorted_nodes if n not in used_nodes]
            
            # Shuffle remaining indices to avoid correlation
            np.random.shuffle(remaining_indices)
            
            for i, r_idx in enumerate(remaining_indices):
                if i < len(remaining_nodes):
                    node = remaining_nodes[i]
                    node_to_entity_idx[node] = r_idx
            
            # Build social graph from G edges
            for u, v in G.edges():
                if u in node_to_entity_idx and v in node_to_entity_idx:
                    u_idx = node_to_entity_idx[u]
                    v_idx = node_to_entity_idx[v]
                    if u_idx != v_idx:
                        self._social_graph[u_idx].add(v_idx)
        else:
            # Scale-free graph for realistic social media topology (hubs/poles)
            G = nx.scale_free_graph(n_entities, seed=42)
            
            for u, v in G.edges():
                if u != v and u < n_entities and v < n_entities:
                    self._social_graph[u].add(v)
            
            # Force many users to follow NewsSource
            for ns_idx in news_source_indices:
                for i in range(n_entities):
                    if i != ns_idx and np.random.rand() < 0.8:
                        self._social_graph[i].add(ns_idx)
            
            # Clear G to free memory
            del G

    def make_observation(
        self,
        game_master: entity_lib.Entity,
        entity: entity_lib.Entity,
        make_new_thread: bool = True
    ) -> Thread:
        """Make an observation for an entity."""
        
        if make_new_thread and isinstance(entity, NewsSource):
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            return new_thread

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

        # Calculate weights proportional to thread length
        thread_lengths = np.array([float(len(thread.content)) for thread in relevant_threads])
        # Add 1 to avoid division by zero and ensure non-zero weights
        weights = thread_lengths

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
                age = len(relevant_threads) - i # Age relative to the most recent thread
                boosts[i] *= (1.0 + (20.0 / age))
        
        weights *= boosts

        # Normalize weights to sum to 1
        weights = weights.clip(min=0)
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
        for _ in range(100):
            idx = np.random.choice(len(self._acting_entities), p=self._acting_probs)
            acting_entity = self._acting_entities[idx]
            
            if isinstance(acting_entity, NewsSource) and not acting_entity.has_news():
                continue
            
            return acting_entity, DEFAULT_ACTION_SPEC
            
        # Fallback: pick a non-NewsSource entity if possible
        non_news_indices = [i for i, e in enumerate(self._acting_entities) if not isinstance(e, NewsSource)]
        if non_news_indices:
             probs = self._acting_probs[non_news_indices]
             if probs.sum() > 0:
                 probs /= probs.sum()
                 idx = np.random.choice(len(non_news_indices), p=probs)
                 return self._acting_entities[non_news_indices[idx]], DEFAULT_ACTION_SPEC
        
        return self._acting_entities[0], DEFAULT_ACTION_SPEC

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

                # Survey Logic
                if self._survey_config and steps % self._survey_config['interval'] == 0:
                    question = self._survey_config['question']
                    options = self._survey_config['options']
                    results = {}
                    print(f"\n--- Running Survey at Step {steps} ---")
                    for entity in entities:
                        if isinstance(entity, User):
                            response = entity.survey_response(question, options)
                            results[entity.name] = response
                    
                    self._survey_results.append({
                        'step': steps,
                        'question': question,
                        'results': results
                    })
                    print(f"--- Survey Completed ---\n")

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

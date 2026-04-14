from collections.abc import Callable, Mapping, Sequence
from typing import Any
from simulation_components.type_aliases import Thread
from simulation_components.entities import NewsSource, User, Entity
from simulation_components.metrics import (
    compute_behavioral_metrics,
    compute_echo_chamber_metrics,
    compute_echo_chamber_metrics_for_survey,
    compute_herd_effect_metrics,
    compute_homophily_metrics,
    compute_model_opinion_scores,
)
import numpy as np
from tqdm import tqdm
import networkx as nx
import time

class SimEngine:
    """Engine interface."""

    def __init__(self,
                 threads: list[Thread] = None,
                 graph: Any = None,
                 survey_config: dict[str, Any] = None,
                 homophily: bool = False,
                 model_probabilities: list[dict[str, float]] | dict[str, dict[str, float]] | None = None,
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
        self._homophily = homophily
        self._model_probabilities = model_probabilities
        self._homophily_metrics = None
        self._behavioral_metrics = None
        self._visualizer_data = {
            "nodes": [],
            "edges": [],
            "threads": [],
            "observations": [],
            "survey_results": [],
            "news_posts": [],
            "behavioral_metrics": {},
        }

    def _compute_echo_chamber_metrics(self) -> dict[str, Any]:
        return compute_echo_chamber_metrics(
            self._survey_results,
            self._visualizer_data,
            self._threads,
        )

    def _compute_echo_chamber_metrics_for_survey(
        self,
        survey: dict[str, Any],
    ) -> dict[str, Any]:
        return compute_echo_chamber_metrics_for_survey(
            survey,
            self._visualizer_data,
            self._threads,
        )

    def _compute_herd_effect_metrics(self) -> dict[str, Any]:
        return compute_herd_effect_metrics(
            self._survey_results,
            self._visualizer_data,
            self._social_graph,
            self._name_to_idx,
        )

    def _compute_behavioral_metrics(self) -> dict[str, Any]:
        return compute_behavioral_metrics(
            self._survey_results,
            self._visualizer_data,
            self._threads,
            self._social_graph,
            self._name_to_idx,
        )

    def get_behavioral_metrics(self) -> dict[str, Any]:
        """Returns computed echo-chamber and herd-effect metrics."""
        self._behavioral_metrics = self._compute_behavioral_metrics()
        return self._behavioral_metrics

    def _compute_model_opinion_scores(self) -> dict[int, float]:
        return compute_model_opinion_scores(self._model_probabilities)

    def _assign_with_homophily(
        self,
        graph: Any,
        remaining_nodes: list[int],
        remaining_indices: list[int],
        entities: Sequence[Entity],
    ) -> dict[int, int]:
        if not remaining_nodes or not remaining_indices:
            return {}

        if graph.is_directed():
            undirected_graph = graph.to_undirected(as_view=False)
        else:
            undirected_graph = graph

        graph_subset = undirected_graph.subgraph(remaining_nodes).copy()
        positions = nx.spring_layout(graph_subset, seed=42, dim=2)
        ordered_nodes = sorted(
            remaining_nodes,
            key=lambda node: float(np.ravel(np.asarray(positions.get(node, [0.0])))[0])
        )

        model_scores = self._compute_model_opinion_scores()
        model_to_entity_indices: dict[int | None, list[int]] = {}
        for entity_idx in remaining_indices:
            model_id = getattr(entities[entity_idx], 'model_id', None)
            model_to_entity_indices.setdefault(model_id, []).append(entity_idx)

        rng = np.random.default_rng(42)
        for indices in model_to_entity_indices.values():
            rng.shuffle(indices)

        ordered_model_ids = sorted(
            model_to_entity_indices.keys(),
            key=lambda model_id: (model_scores.get(model_id, float('inf')), str(model_id))
        )

        ordered_entity_indices = []
        for model_id in ordered_model_ids:
            ordered_entity_indices.extend(model_to_entity_indices[model_id])

        node_to_entity_idx = {}
        for node, entity_idx in zip(ordered_nodes, ordered_entity_indices):
            node_to_entity_idx[node] = entity_idx

        return node_to_entity_idx

    def _compute_homophily_metrics(self, entities: Sequence[Entity]) -> dict[str, float]:
        return compute_homophily_metrics(
            list(entities),
            self._social_graph,
            self._model_probabilities,
        )

    def get_survey_results(self) -> list[dict[str, Any]]:
        """Returns the results of the surveys."""
        return self._survey_results

    def get_visualizer_data(self) -> dict[str, Any]:
        """Returns the data needed for the visualizer."""
        self._visualizer_data["threads"] = [
            {"id": t.id, "messages": t.content} for t in self._threads
        ]
        self._behavioral_metrics = self._compute_behavioral_metrics()
        self._visualizer_data["behavioral_metrics"] = self._behavioral_metrics
        return self._visualizer_data

    def _initialize_social_context(self, entities: Sequence[Entity]):
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

            if self._homophily:
                homophily_assignment = self._assign_with_homophily(
                    graph=G,
                    remaining_nodes=remaining_nodes,
                    remaining_indices=remaining_indices,
                    entities=entities,
                )
                node_to_entity_idx.update(homophily_assignment)
            else:
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

        # Populate visualizer data for nodes and edges
        for i, entity in enumerate(entities):
            self._visualizer_data["nodes"].append({
                "id": i,
                "name": entity.name,
                "type": "NewsSource" if isinstance(entity, NewsSource) else "User",
                "model_id": getattr(entity, "model_id", None)
            })
        for u_idx, neighbors in enumerate(self._social_graph):
            for v_idx in neighbors:
                self._visualizer_data["edges"].append({
                    "source": u_idx,
                    "target": v_idx
                })

        self._homophily_metrics = self._compute_homophily_metrics(entities)

    def make_observation(
        self,
        game_master: Entity,
        entity: Entity,
        make_new_thread: bool = True,
        step: int = 0
    ) -> Thread:
        """Make an observation for an entity."""
        
        if make_new_thread and isinstance(entity, NewsSource):
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            self._visualizer_data["observations"].append({
                "step": step,
                "entity_name": entity.name,
                "thread_id": new_thread.id
            })
            return new_thread

        if make_new_thread and np.random.rand() < 0.3:
            new_thread = Thread(id=len(self._threads), content=[])
            self._threads.append(new_thread)
            entity.observe(new_thread)
            self._visualizer_data["observations"].append({
                "step": step,
                "entity_name": entity.name,
                "thread_id": new_thread.id
            })
            return new_thread

        if not self._threads:
            if make_new_thread:
                new_thread = Thread(id=0, content=[])
                self._threads.append(new_thread)
                self._visualizer_data["observations"].append({
                    "step": step,
                    "entity_name": entity.name,
                    "thread_id": new_thread.id
                })
                return new_thread
            else:
                return None

        # Only consider the last 50 threads for efficiency
        relevant_threads = self._threads[-50:]

        # Calculate weights proportional to thread length
        thread_lengths = np.array([float(len(thread.content) + 2) for thread in relevant_threads])
        # Add 1 to normalize and prevent zero-length threads from having zero probability, and +1 more to ensure even very short threads have some chance of being selected.
        weights = thread_lengths

        # Zero out weights for threads with more than 20 messages
        weights[np.array([len(thread.content) > 20 for thread in relevant_threads])] = 0.0

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
            self._visualizer_data["observations"].append({
                "step": step,
                "entity_name": entity.name,
                "thread_id": new_thread.id
            })
            return new_thread
        else:
            return None
        
        random_thread_idx = np.random.choice(len(relevant_threads), p=weights)
        random_thread = relevant_threads[random_thread_idx]
        entity.observe(random_thread)
        self._visualizer_data["observations"].append({
            "step": step,
            "entity_name": entity.name,
            "thread_id": random_thread.id
        })
        return random_thread

    def next_acting(
        self,
        game_master: Entity,
        entities: Sequence[Entity],
    ) -> Entity:
        """Return the next entity or entities to act."""
        
        if self._entity_activity_probs is None:
            self._initialize_social_context(entities)

        # Use cached probabilities for efficiency
        for _ in range(100):
            idx = np.random.choice(len(self._acting_entities), p=self._acting_probs)
            acting_entity = self._acting_entities[idx]
            
            if isinstance(acting_entity, NewsSource) and not acting_entity.has_news():
                continue
            
            return acting_entity
            
        # Fallback: pick a non-NewsSource entity if possible
        non_news_indices = [i for i, e in enumerate(self._acting_entities) if not isinstance(e, NewsSource)]
        if non_news_indices:
             probs = self._acting_probs[non_news_indices]
             if probs.sum() > 0:
                 probs /= probs.sum()
                 idx = np.random.choice(len(non_news_indices), p=probs)
                 return self._acting_entities[non_news_indices[idx]]
        
        return self._acting_entities[0]

    def run_loop(
        self,
        game_masters: Sequence[Entity],
        entities: Sequence[Entity],
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

        if self._entity_activity_probs is None:
            self._initialize_social_context(entities)

        if self._homophily_metrics is not None:
            metrics = self._homophily_metrics
            print(
                "Homophily metrics: "
                f"edge_similarity={metrics['edge_similarity']:.4f}, "
                f"random_similarity={metrics['random_similarity']:.4f}, "
                f"excess_similarity={metrics['excess_similarity']:.4f}, "
                f"same_model_edge_fraction={metrics['same_model_edge_fraction']:.4f}, "
                f"edges={metrics['edge_count']}"
            )

        with tqdm(total=max_steps, disable=not verbose, desc="Game Steps") as pbar:
            while steps <= max_steps:
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
                    for entity in tqdm(entities, desc="Surveying Entities", leave=False):
                        if isinstance(entity, User):
                            response = entity.survey_response(question, options)
                            results[entity.name] = response
                    
                    self._survey_results.append({
                        'step': steps,
                        'question': question,
                        'results': results
                    })
                    self._visualizer_data["survey_results"].append({
                        'step': steps,
                        'question': question,
                        'results': results
                    })
                    print(f"--- Survey Completed ---\n")
                    if steps == max_steps:
                        break

                # 10 observations for every action
                for i in range(10):
                    acting_entity = self.next_acting(
                        game_master, entities)
                    
                    make_new_thread = ( i == 9 )  # Only create new thread on the last observation
                    observation = self.make_observation(game_master, acting_entity, make_new_thread, step=steps)
                action = acting_entity.act()
                observation.content.append({'role': acting_entity.name, 'content': action, 'step': steps})
                
                if isinstance(acting_entity, NewsSource):
                    self._visualizer_data["news_posts"].append({
                        "step": steps,
                        "thread_id": observation.id,
                        "content": action
                    })

                steps += 1
                pbar.update(1)

        return

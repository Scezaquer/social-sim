
import networkx as nx
import json
import time

class MetricTracker:
    def __init__(self):
        # A directed graph where nodes are tweets and edges represent "caused by"
        # e.g. Reply -> Original Tweet
        self.propagation_forest = nx.DiGraph()
        self.events_log = []
        
    def log_event(self, timestamp, agent_name, action, content, source_tweet_id=None, tweet_id=None):
        """
        Logs an interaction event.
        """
        event = {
            "timestamp": timestamp,
            "agent": agent_name,
            "action": action,
            "content": content,
            "source_tweet_id": source_tweet_id,
            "tweet_id": tweet_id
        }
        self.events_log.append(event)

        if tweet_id:
            # If this action created a new artifact (tweet, reply, quote)
            self.propagation_forest.add_node(tweet_id, 
                                           author=agent_name, 
                                           timestamp=timestamp, 
                                           content=content[:50]) # store partial content to save RAM
            
            if source_tweet_id:
                # If it's a response to something, add an edge
                self.propagation_forest.add_edge(source_tweet_id, tweet_id, type=action)
        
        elif action in ["Retweet", "Like"] and source_tweet_id:
            # For non-generative actions, we might just want to track them as properties of the source node
            # or as terminal nodes in the graph
            interaction_id = f"{action}_{agent_name}_{time.time()}"
            self.propagation_forest.add_node(interaction_id, author=agent_name, type=action, timestamp=timestamp)
            self.propagation_forest.add_edge(source_tweet_id, interaction_id, type=action)

    def get_propagation_stats(self):
        """
        Returns stats about the largest cascades.
        """
        if self.propagation_forest.number_of_nodes() == 0:
            return {}

        roots = [n for n, d in self.propagation_forest.in_degree() if d == 0]
        cascades = []
        
        for root in roots:
            tree_size = len(nx.descendants(self.propagation_forest, root)) + 1
            cascades.append({"root_id": root, "size": tree_size})
            
        return {
            "total_events": len(self.events_log),
            "total_cascades": len(roots),
            "max_cascade_size": max([c["size"] for c in cascades]) if cascades else 0,
            "avg_cascade_size": sum([c["size"] for c in cascades]) / len(cascades) if cascades else 0
        }

    def save_logs(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.events_log, f, indent=2)

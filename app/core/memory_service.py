import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import time
import logging

from app.models.memory import Entity, Relation, KnowledgeGraph
from app.utils.config import get_config

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self, memory_file_path: str = None):
        config = get_config()
        
        # Set up file path for storage
        memory_file_path = memory_file_path or config.memory_file_path
        self.memory_file_path = Path(
            memory_file_path
            if Path(memory_file_path).is_absolute()
            else Path(os.getcwd()) / memory_file_path
        )
        
        # Ensure directory exists
        self.memory_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up user preferences path
        self.user_prefs_dir = self.memory_file_path.parent / "user_preferences"
        self.user_prefs_dir.mkdir(exist_ok=True)
        
        # Add option to use graph database instead of file-based storage
        self.use_graph_db = config.use_graph_db
        if self.use_graph_db:
            import networkx as nx
            self.graph = nx.DiGraph()
            self._load_graph_from_file()
    
    def _read_graph_file(self) -> KnowledgeGraph:
        """Read the knowledge graph from disk"""
        if not self.memory_file_path.exists():
            return KnowledgeGraph(entities=[], relations=[])
        
        try:
            with open(self.memory_file_path, "r", encoding="utf-8") as f:
                lines = [line for line in f if line.strip()]
                entities = []
                relations = []
                
                for line in lines:
                    item = json.loads(line)
                    if item.get("type") == "entity":
                        entities.append(Entity(
                            name=item["name"],
                            entity_type=item["entity_type"],
                            observations=item.get("observations", [])
                        ))
                    elif item.get("type") == "relation":
                        relations.append(Relation(
                            **{k: v for k, v in item.items() if k != "type"}
                        ))
                
                return KnowledgeGraph(entities=entities, relations=relations)
        except Exception as e:
            print(f"Error reading graph file: {e}")
            return KnowledgeGraph(entities=[], relations=[])
    
    def _save_graph(self, graph: KnowledgeGraph):
        """Save the knowledge graph to disk"""
        lines = []
        
        # Save entities
        for e in graph.entities:
            entity_dict = e.dict()
            entity_dict["type"] = "entity"
            lines.append(json.dumps(entity_dict))
        
        # Save relations
        for r in graph.relations:
            relation_dict = r.dict(by_alias=True)
            relation_dict["type"] = "relation"
            lines.append(json.dumps(relation_dict))
        
        # Write to file
        with open(self.memory_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    
    def create_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create new entities in the graph"""
        graph = self._read_graph_file()
        existing_names = {e.name for e in graph.entities}
        
        # Convert input dictionaries to Entity objects
        entity_objects = []
        for entity_dict in entities:
            if isinstance(entity_dict, dict):
                entity_objects.append(Entity(**entity_dict))
            else:
                entity_objects.append(entity_dict)
        
        # Filter out existing entities
        new_entities = [e for e in entity_objects if e.name not in existing_names]
        
        # Add new entities to graph
        graph.entities.extend(new_entities)
        self._save_graph(graph)
        
        # Return the added entities
        return [e.dict() for e in new_entities]
    
    def create_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create new relations in the graph"""
        graph = self._read_graph_file()
        
        # Get existing relations for deduplication
        existing_relations = {(r.from_, r.to, r.relation_type) for r in graph.relations}
        
        # Convert input dictionaries to Relation objects
        relation_objects = []
        for relation_dict in relations:
            if isinstance(relation_dict, dict):
                # Handle inconsistent field naming in input
                if "relation_type" in relation_dict and "relationType" not in relation_dict:
                    relation_dict["relationType"] = relation_dict["relation_type"]
                if "from_" in relation_dict and "from" not in relation_dict:
                    relation_dict["from"] = relation_dict["from_"]
                relation_objects.append(Relation(**relation_dict))
            else:
                relation_objects.append(relation_dict)
        
        # Filter out existing relations
        new_relations = [r for r in relation_objects 
                        if (r.from_, r.to, r.relation_type) not in existing_relations]
        
        # Add new relations to graph
        graph.relations.extend(new_relations)
        self._save_graph(graph)
        
        # Return the added relations
        return [r.dict(by_alias=True) for r in new_relations]
    
    def add_observations(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add observations to entities"""
        graph = self._read_graph_file()
        results = []
        
        # Process each observation item
        for obs_item in observations:
            entity_name = obs_item["entity_name"]
            contents = obs_item["contents"]
            
            # Find the entity
            entity = next((e for e in graph.entities if e.name == entity_name), None)
            
            if entity:
                # Get observations that are not already in the entity
                new_observations = [c for c in contents if c not in entity.observations]
                
                # Add new observations
                entity.observations.extend(new_observations)
                
                # Record result
                results.append({
                    "entity_name": entity_name,
                    "added_observations": new_observations
                })
        
        # Save the updated graph
        self._save_graph(graph)
        
        return results
    
    def delete_entities(self, entity_names: List[str]) -> Dict[str, Any]:
        """Delete entities and their relations"""
        graph = self._read_graph_file()
        
        # Remove entities
        initial_count = len(graph.entities)
        graph.entities = [e for e in graph.entities if e.name not in entity_names]
        entities_removed = initial_count - len(graph.entities)
        
        # Remove relations involving the deleted entities
        initial_relations_count = len(graph.relations)
        graph.relations = [r for r in graph.relations 
                         if r.from_ not in entity_names and r.to not in entity_names]
        relations_removed = initial_relations_count - len(graph.relations)
        
        # Save the updated graph
        self._save_graph(graph)
        
        return {
            "entities_removed": entities_removed,
            "relations_removed": relations_removed
        }
    
    def delete_relations(self, relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Delete specific relations"""
        graph = self._read_graph_file()
        
        # Convert input dictionaries to relation tuples for comparison
        relation_tuples = []
        for relation in relations:
            from_entity = relation.get("from", relation.get("from_"))
            to_entity = relation.get("to")
            relation_type = relation.get("relation_type", relation.get("relationType"))
            if from_entity and to_entity and relation_type:
                relation_tuples.append((from_entity, to_entity, relation_type))
        
        # Remove matching relations
        initial_count = len(graph.relations)
        graph.relations = [r for r in graph.relations 
                         if (r.from_, r.to, r.relation_type) not in relation_tuples]
        relations_removed = initial_count - len(graph.relations)
        
        # Save the updated graph
        self._save_graph(graph)
        
        return {
            "relations_removed": relations_removed
        }
    
    def search_nodes(self, query: str) -> KnowledgeGraph:
        """Search for nodes matching the query"""
        graph = self._read_graph_file()
        
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Find entities matching the query
        matching_entities = []
        for entity in graph.entities:
            # Check name
            if query_lower in entity.name.lower():
                matching_entities.append(entity)
                continue
                
            # Check type
            if query_lower in entity.entity_type.lower():
                matching_entities.append(entity)
                continue
                
            # Check observations
            if any(query_lower in observation.lower() for observation in entity.observations):
                matching_entities.append(entity)
                continue
        
        # Get names of matching entities
        matching_names = {entity.name for entity in matching_entities}
        
        # Find relations between matching entities
        matching_relations = [r for r in graph.relations 
                            if r.from_ in matching_names and r.to in matching_names]
        
        return KnowledgeGraph(entities=matching_entities, relations=matching_relations)
    
    def open_nodes(self, names: List[str]) -> KnowledgeGraph:
        """Retrieve specific nodes by name"""
        graph = self._read_graph_file()
        
        # Find the specified entities
        entities = [e for e in graph.entities if e.name in names]
        
        # Get entity names
        entity_names = {e.name for e in entities}
        
        # Find relations between these entities
        relations = [r for r in graph.relations 
                   if r.from_ in entity_names and r.to in entity_names]
        
        return KnowledgeGraph(entities=entities, relations=relations)
    
    def get_user_preference(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user preferences"""
        try:
            pref_file = self.user_prefs_dir / f"{user_id}.json"
            if not pref_file.exists():
                logger.debug(f"No preferences found for user {user_id}")
                return {}
            
            with open(pref_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading user preferences for {user_id}: {e}", exc_info=True)
            return {}

    def set_user_preference(self, user_id: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Store user preferences"""
        try:
            # Validate user_id to prevent path traversal
            if not user_id or '/' in user_id or '\\' in user_id or '..' in user_id:
                logger.error(f"Invalid user ID: {user_id}")
                raise ValueError(f"Invalid user ID")
                
            pref_file = self.user_prefs_dir / f"{user_id}.json"
            
            # Merge with existing preferences if present
            existing_prefs = {}
            if pref_file.exists():
                try:
                    with open(pref_file, "r", encoding="utf-8") as f:
                        existing_prefs = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not read existing preferences for {user_id}: {e}")
            
            # Update with new preferences
            existing_prefs.update(preferences)
            
            # Write back to file
            with open(pref_file, "w", encoding="utf-8") as f:
                json.dump(existing_prefs, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Updated preferences for user {user_id}")
            return existing_prefs
        except Exception as e:
            logger.error(f"Failed to set preferences for {user_id}: {e}", exc_info=True)
            raise
    
    def get_full_graph(self) -> KnowledgeGraph:
        """Get the entire knowledge graph"""
        return self._read_graph_file()
    
    def find_similar_entities(self, entity_name: str, threshold: float = 0.8) -> List[str]:
        """Find entities with similar names"""
        graph = self._read_graph_file()
        similar = []
        
        # Use fuzzy matching to find similar entity names
        from difflib import SequenceMatcher
        
        for entity in graph.entities:
            similarity = SequenceMatcher(None, entity_name.lower(), entity.name.lower()).ratio()
            if similarity >= threshold and entity_name != entity.name:
                similar.append(entity.name)
                
        return similar

    def _load_graph_from_file(self):
        """Load graph data from file into networkx graph"""
        try:
            knowledge_graph = self._read_graph_file()
            
            # Clear existing graph
            self.graph.clear()
            
            # Add all entities as nodes
            for entity in knowledge_graph.entities:
                self.graph.add_node(
                    entity.name,
                    entity_type=entity.entity_type,
                    observations=entity.observations
                )
            
            # Add all relations as edges
            for relation in knowledge_graph.relations:
                self.graph.add_edge(
                    relation.from_,
                    relation.to,
                    relation_type=relation.relation_type
                )
                
            logger.info(f"Loaded {len(knowledge_graph.entities)} entities and {len(knowledge_graph.relations)} relations into graph")
            return True
        except Exception as e:
            logger.error(f"Error loading graph from file: {e}", exc_info=True)
            return False

    def find_paths(self, start_entity: str, end_entity: str, max_length: int = 3) -> List[List[Dict[str, Any]]]:
        """Find paths between two entities in the graph"""
        if not self.use_graph_db:
            raise ValueError("Graph database not enabled")
            
        try:
            import networkx as nx
            
            # Check if entities exist
            if start_entity not in self.graph.nodes:
                raise ValueError(f"Entity '{start_entity}' not found in graph")
            if end_entity not in self.graph.nodes:
                raise ValueError(f"Entity '{end_entity}' not found in graph")
            
            # Find all simple paths up to max_length
            paths = list(nx.all_simple_paths(self.graph, start_entity, end_entity, cutoff=max_length))
            
            # Format results
            result_paths = []
            for path in paths:
                path_info = []
                
                # Add nodes and edges to path
                for i in range(len(path)):
                    # Add node
                    node = path[i]
                    node_data = self.graph.nodes[node]
                    path_info.append({
                        "type": "entity",
                        "name": node,
                        "entity_type": node_data.get("entity_type", "unknown"),
                    })
                    
                    # Add edge if not last node
                    if i < len(path) - 1:
                        next_node = path[i+1]
                        edge_data = self.graph.get_edge_data(node, next_node)
                        path_info.append({
                            "type": "relation",
                            "from": node,
                            "to": next_node,
                            "relation_type": edge_data.get("relation_type", "unknown"),
                        })
                
                result_paths.append(path_info)
                
            return result_paths
        except ValueError as e:
            # Re-raise ValueError for specific error handling
            raise
        except Exception as e:
            logger.error(f"Error finding paths: {e}", exc_info=True)
            return []

    def get_similar_entities(self, entity_name: str, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Find entities with similar names"""
        try:
            import difflib
            
            if self.use_graph_db:
                # If graph DB is enabled, use graph nodes
                all_entities = list(self.graph.nodes)
            else:
                # Otherwise use entities from knowledge graph
                knowledge_graph = self._read_graph_file()
                all_entities = [entity.name for entity in knowledge_graph.entities]
            
            # Calculate similarity scores
            similarities = []
            for name in all_entities:
                score = difflib.SequenceMatcher(None, entity_name.lower(), name.lower()).ratio()
                if score >= threshold:
                    similarities.append({
                        "name": name,
                        "similarity": score
                    })
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similarities
        except Exception as e:
            logger.error(f"Error finding similar entities: {e}", exc_info=True)
            return []

    def get_entity_connections(self, entity_name: str) -> Dict[str, Any]:
        """Get all connections for a specific entity"""
        if not self.use_graph_db:
            raise ValueError("Graph database not enabled")
            
        try:
            if entity_name not in self.graph:
                raise ValueError(f"Entity '{entity_name}' does not exist in the graph")
                
            # Get all neighbors
            neighbors = list(self.graph.neighbors(entity_name))
            
            # Get incoming edges (entities that connect to this one)
            incoming = []
            for source in self.graph.predecessors(entity_name):
                if source != entity_name:  # Skip self-loops
                    edge_data = self.graph.get_edge_data(source, entity_name)
                    incoming.append({
                        "from": source,
                        "relation_type": edge_data.get("relation_type", "unknown")
                    })
                    
            # Get outgoing edges (entities this one connects to)
            outgoing = []
            for target in self.graph.successors(entity_name):
                if target != entity_name:  # Skip self-loops
                    edge_data = self.graph.get_edge_data(entity_name, target)
                    outgoing.append({
                        "to": target,
                        "relation_type": edge_data.get("relation_type", "unknown")
                    })
                    
            return {
                "name": entity_name,
                "properties": self.graph.nodes[entity_name],
                "connections": {
                    "incoming": incoming,
                    "outgoing": outgoing,
                    "total": len(incoming) + len(outgoing)
                }
            }
        except Exception as e:
            logger.error(f"Error getting entity connections: {e}", exc_info=True)
            return {"error": str(e)}

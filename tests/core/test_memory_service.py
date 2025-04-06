import pytest
import os
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.core.memory_service import MemoryService
from app.models.memory import Entity, Relation, KnowledgeGraph

@pytest.fixture
def temp_dir():
    # Create a temporary directory for testing
    temp_dir = Path("./test_data")
    temp_dir.mkdir(exist_ok=True)
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def memory_service(temp_dir):
    # Create a memory service with test file
    memory_file = temp_dir / "test_memory.json"
    
    with patch('app.core.memory_service.get_config') as mock_config:
        mock_config.return_value.use_graph_db = False
        service = MemoryService(str(memory_file))
        yield service

class TestMemoryService:
    def test_create_entities(self, memory_service):
        # Test creating entities
        entities = [
            {"name": "Person1", "entity_type": "person", "properties": {"age": 30}},
            {"name": "Place1", "entity_type": "location", "properties": {"country": "USA"}}
        ]
        
        result = memory_service.create_entities(entities)
        assert len(result) == 2
        assert result[0]["name"] == "Person1"
        assert result[1]["name"] == "Place1"
        
        # Test retrieving entities
        all_entities = memory_service.get_entities()
        assert len(all_entities) == 2
        assert any(e["name"] == "Person1" for e in all_entities)
        
    def test_create_relations(self, memory_service):
        # First create some entities
        entities = [
            {"name": "Person1", "entity_type": "person"},
            {"name": "Place1", "entity_type": "location"}
        ]
        memory_service.create_entities(entities)
        
        # Now create a relation between them
        relations = [
            {"from": "Person1", "to": "Place1", "relation_type": "visited"}
        ]
        
        result = memory_service.create_relations(relations)
        assert len(result) == 1
        assert result[0]["from_"] == "Person1"
        assert result[0]["to"] == "Place1"
        assert result[0]["relation_type"] == "visited"
        
        # Verify the relation exists
        all_relations = memory_service.get_relations()
        assert len(all_relations) == 1
        assert all_relations[0]["from_"] == "Person1"
        
    def test_query_graph(self, memory_service):
        # Create test data
        memory_service.create_entities([
            {"name": "Alice", "entity_type": "person"},
            {"name": "Bob", "entity_type": "person"},
            {"name": "CompanyX", "entity_type": "company"}
        ])
        
        memory_service.create_relations([
            {"from": "Alice", "to": "CompanyX", "relation_type": "works_at"},
            {"from": "Bob", "to": "CompanyX", "relation_type": "works_at"}
        ])
        
        # Query by entity type
        people = memory_service.query_entities(entity_type="person")
        assert len(people) == 2
        assert any(p["name"] == "Alice" for p in people)
        assert any(p["name"] == "Bob" for p in people)
        
        # Query relations
        company_relations = memory_service.query_relations(to_entity="CompanyX")
        assert len(company_relations) == 2
        
    def test_user_preferences(self, memory_service):
        # Test setting preferences
        user_id = "test_user"
        prefs = {"theme": "dark", "language": "en"}
        
        # Set preferences
        result = memory_service.set_user_preference(user_id, prefs)
        assert result == prefs
        
        # Get preferences
        retrieved = memory_service.get_user_preference(user_id)
        assert retrieved == prefs
        
        # Update preferences
        updated_prefs = {"theme": "light", "font_size": "large"}
        result = memory_service.set_user_preference(user_id, updated_prefs)
        assert result["theme"] == "light"
        assert result["language"] == "en"  # Should preserve existing values
        assert result["font_size"] == "large"  # Should add new values
        
    def test_delete_entities(self, memory_service):
        # Create test entities
        memory_service.create_entities([
            {"name": "ToDelete1", "entity_type": "test"},
            {"name": "ToDelete2", "entity_type": "test"},
            {"name": "ToKeep", "entity_type": "test"}
        ])
        
        # Create some relations
        memory_service.create_relations([
            {"from": "ToDelete1", "to": "ToKeep", "relation_type": "test_relation"}
        ])
        
        # Delete entities
        result = memory_service.delete_entities(["ToDelete1", "ToDelete2"])
        assert result["entities_removed"] == 2
        assert result["relations_removed"] == 1
        
        # Verify deletion
        remaining = memory_service.get_entities()
        assert len(remaining) == 1
        assert remaining[0]["name"] == "ToKeep"
        
    def test_delete_relations(self, memory_service):
        # Create test data
        memory_service.create_entities([
            {"name": "E1", "entity_type": "test"},
            {"name": "E2", "entity_type": "test"},
            {"name": "E3", "entity_type": "test"}
        ])
        
        memory_service.create_relations([
            {"from": "E1", "to": "E2", "relation_type": "rel1"},
            {"from": "E1", "to": "E3", "relation_type": "rel2"},
            {"from": "E2", "to": "E3", "relation_type": "rel3"}
        ])
        
        # Delete one relation
        result = memory_service.delete_relations([
            {"from": "E1", "to": "E2", "relation_type": "rel1"}
        ])
        
        assert result["relations_removed"] == 1
        
        # Verify remaining relations
        remaining = memory_service.get_relations()
        assert len(remaining) == 2
        
    def test_entity_connections(self, memory_service):
        # Create test data
        memory_service.create_entities([
            {"name": "Central", "entity_type": "test"},
            {"name": "Connected1", "entity_type": "test"},
            {"name": "Connected2", "entity_type": "test"}
        ])
        
        memory_service.create_relations([
            {"from": "Central", "to": "Connected1", "relation_type": "outgoing"},
            {"from": "Connected2", "to": "Central", "relation_type": "incoming"}
        ])
        
        # Get connections for Central
        connections = memory_service.get_entity_connections("Central")
        
        assert connections["name"] == "Central"
        assert len(connections["connections"]["incoming"]) == 1
        assert len(connections["connections"]["outgoing"]) == 1
        assert connections["connections"]["incoming"][0]["from"] == "Connected2"
        assert connections["connections"]["outgoing"][0]["to"] == "Connected1"
        
    @patch('networkx.DiGraph')
    def test_graph_db_mode(self, mock_digraph, temp_dir):
        # Test when using graph database mode
        memory_file = temp_dir / "graph_db_test.json"
        
        with patch('app.core.memory_service.get_config') as mock_config:
            mock_config.return_value.use_graph_db = True
            
            # This should initialize the networkx graph
            service = MemoryService(str(memory_file))
            
            # Verify DiGraph was created
            assert mock_digraph.called

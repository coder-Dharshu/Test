import pytest
import sqlite3
import pandas as pd
import tempfile
import os
from smart_triage.db_analyzer import (
    SQLiteSchemaLoader,
    PandasSchemaInferrer,
    RelationshipDetector,
    analyze_schema,
    RelationshipType
)

@pytest.fixture
def sqlite_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)")
    cursor.execute("CREATE TABLE posts (id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT, FOREIGN KEY(user_id) REFERENCES users(id))")
    cursor.execute("CREATE TABLE user_friends (user_id1 INTEGER, user_id2 INTEGER, FOREIGN KEY(user_id1) REFERENCES users(id), FOREIGN KEY(user_id2) REFERENCES users(id))")
    conn.commit()
    conn.close()
    yield path
    os.remove(path)

@pytest.fixture
def excel_path():
    fd, path = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]}).to_excel(writer, sheet_name="users", index=False)
        pd.DataFrame({"id": [1], "users_id": [1], "title": ["Post 1"]}).to_excel(writer, sheet_name="posts", index=False)
    
    yield path
    os.remove(path)

def test_sqlite_schema_loader(sqlite_db_path):
    loader = SQLiteSchemaLoader()
    tables, rels = loader.load(sqlite_db_path)
    assert len(tables) == 3
    table_names = [t.name for t in tables]
    assert "users" in table_names
    assert "posts" in table_names
    assert "user_friends" in table_names
    
    assert len(rels) == 3

def test_pandas_schema_inferrer(excel_path):
    loader = PandasSchemaInferrer()
    tables, rels = loader.load(excel_path)
    assert len(tables) == 2
    assert len(rels) == 1
    rel = rels[0]
    assert rel.source_table == "posts"
    assert rel.source_columns == ("users_id",)
    assert rel.target_table == "users"

def test_relationship_detector(sqlite_db_path):
    loader = SQLiteSchemaLoader()
    tables, rels = loader.load(sqlite_db_path)
    detector = RelationshipDetector()
    result = detector.analyze(tables, rels)
    
    assert "user_friends" in result.junction_tables
    junction_rels = [r for r in result.relationships if r.relationship_type == RelationshipType.JUNCTION]
    assert len(junction_rels) == 2

def test_analyze_schema_facade(sqlite_db_path):
    result = analyze_schema(sqlite_db_path)
    assert len(result.tables) == 3
    assert len(result.relationships) == 3

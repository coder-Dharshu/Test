"""
smart_triage/db_analyzer.py
============================
Production-quality Database Schema Analyzer and ER Diagram Renderer.

Architecture (SOLID, Clean Architecture):
  - SchemaLoader (ABC)           : abstract interface for loading schemas
  - SQLiteSchemaLoader           : reads FK constraints via PRAGMA
  - PandasSchemaInferrer         : infers FK from column naming conventions
  - Relationship (dataclass)     : immutable FK relationship descriptor
  - RelationshipDetector         : analyzes schema for all relationship types
  - ERDiagramRenderer            : renders interactive pyvis graph

Relationship types detected:
  - Explicit FOREIGN KEY (SQLite PRAGMA foreign_key_list)
  - Inferred FK (column name pattern: {table}_id or {table}Id → {table}.id)
  - Self-referencing (table refers to its own PK)
  - Junction/Bridge tables (table with exactly 2 FK cols to 2 distinct tables)
  - Composite FK (multiple columns forming one FK)

Design Decisions:
  - SchemaLoader is an ABC to allow injecting mock loaders in tests.
  - All relationship detection is pure (no I/O after schema is loaded).
  - pyvis is optional; falls back to textual representation if unavailable.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.warning("pyvis not installed — ER diagram will render as text table only.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnInfo:
    """Metadata for a single database column."""
    name: str
    data_type: str
    is_primary_key: bool = False
    is_nullable: bool = True
    default_value: Optional[str] = None


@dataclass(frozen=True)
class TableInfo:
    """Metadata for a database table."""
    name: str
    columns: tuple[ColumnInfo, ...]
    row_count: Optional[int] = None

    @property
    def primary_keys(self) -> list[str]:
        return [c.name for c in self.columns if c.is_primary_key]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


class RelationshipType(str):
    FOREIGN_KEY   = "foreign_key"
    INFERRED      = "inferred"
    SELF_REF      = "self_referencing"
    JUNCTION      = "junction"
    COMPOSITE_FK  = "composite_fk"


@dataclass(frozen=True)
class Relationship:
    """
    Immutable descriptor for a single table relationship.

    Attributes:
        source_table: The table that holds the FK column(s).
        source_columns: FK column name(s) in the source table.
        target_table: The referenced (parent) table.
        target_columns: Referenced column name(s) in the target table.
        relationship_type: One of RelationshipType constants.
        cardinality: e.g. 'many-to-one', 'one-to-one', 'many-to-many'.
        confidence: 0.0–1.0 — how confident this inferred relationship is.
        on_delete: DELETE action (CASCADE, SET NULL, RESTRICT, etc.)
        on_update: UPDATE action.
    """
    source_table: str
    source_columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]
    relationship_type: str
    cardinality: str = "many-to-one"
    confidence: float = 1.0
    on_delete: Optional[str] = None
    on_update: Optional[str] = None

    @property
    def label(self) -> str:
        src = ", ".join(self.source_columns)
        tgt = ", ".join(self.target_columns)
        return f"{self.source_table}.{src} → {self.target_table}.{tgt}"

    @property
    def is_self_referencing(self) -> bool:
        return self.source_table == self.target_table

    def to_dict(self) -> dict:
        return {
            "source_table"    : self.source_table,
            "source_columns"  : list(self.source_columns),
            "target_table"    : self.target_table,
            "target_columns"  : list(self.target_columns),
            "relationship_type": self.relationship_type,
            "cardinality"     : self.cardinality,
            "confidence"      : self.confidence,
            "on_delete"       : self.on_delete,
            "on_update"       : self.on_update,
        }


@dataclass
class SchemaAnalysisResult:
    """Complete result of a schema analysis."""
    tables: list[TableInfo]
    relationships: list[Relationship]
    junction_tables: list[str]
    warnings: list[str] = field(default_factory=list)
    source_type: str = "unknown"

    @property
    def has_relationships(self) -> bool:
        return len(self.relationships) > 0

    def summary(self) -> str:
        return (
            f"Schema: {len(self.tables)} tables, "
            f"{len(self.relationships)} relationships "
            f"({sum(1 for r in self.relationships if r.relationship_type == RelationshipType.FOREIGN_KEY)} explicit FK, "
            f"{sum(1 for r in self.relationships if r.relationship_type == RelationshipType.INFERRED)} inferred)"
        )


# ---------------------------------------------------------------------------
# SchemaLoader (Abstract Base)
# ---------------------------------------------------------------------------

class SchemaLoader(ABC):
    """
    Abstract interface for loading database schema information.
    Implementations: SQLiteSchemaLoader, PandasSchemaInferrer.
    """

    @abstractmethod
    def load(self, source: str) -> tuple[list[TableInfo], list[Relationship]]:
        """
        Load tables and explicit relationships from the source.

        Args:
            source: Path to file, connection string, etc.

        Returns:
            (tables, explicit_relationships)
        """


# ---------------------------------------------------------------------------
# SQLiteSchemaLoader
# ---------------------------------------------------------------------------

class SQLiteSchemaLoader(SchemaLoader):
    """
    Loads schema from a SQLite database file using PRAGMA introspection.

    Uses PRAGMA table_info for column metadata and PRAGMA foreign_key_list
    for explicit FK constraints.
    """

    def load(self, source: str) -> tuple[list[TableInfo], list[Relationship]]:
        """
        Args:
            source: Path to SQLite .db file.
        """
        db_path = Path(source)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite file not found: {source}")

        tables: list[TableInfo] = []
        relationships: list[Relationship] = []

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all user tables (exclude SQLite internals)
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            )
            table_names = [row["name"] for row in cursor.fetchall()]

            for tname in table_names:
                columns = self._load_columns(cursor, tname)
                row_count = self._get_row_count(cursor, tname)
                tables.append(TableInfo(
                    name=tname,
                    columns=tuple(columns),
                    row_count=row_count,
                ))

                fks = self._load_foreign_keys(cursor, tname)
                relationships.extend(fks)

            conn.close()

        except sqlite3.DatabaseError as exc:
            raise ValueError(f"Cannot read SQLite database '{source}': {exc}") from exc

        logger.info(
            "SQLiteSchemaLoader loaded %d tables, %d explicit FK relationships from %s",
            len(tables), len(relationships), db_path.name
        )
        return tables, relationships

    @staticmethod
    def _load_columns(cursor: sqlite3.Cursor, table_name: str) -> list[ColumnInfo]:
        cursor.execute(f"PRAGMA table_info({table_name})")
        rows = cursor.fetchall()
        return [
            ColumnInfo(
                name=row["name"],
                data_type=row["type"] or "TEXT",
                is_primary_key=bool(row["pk"]),
                is_nullable=not row["notnull"],
                default_value=str(row["dflt_value"]) if row["dflt_value"] is not None else None,
            )
            for row in rows
        ]

    @staticmethod
    def _get_row_count(cursor: sqlite3.Cursor, table_name: str) -> Optional[int]:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM [{table_name}]")
            return cursor.fetchone()[0]
        except Exception:
            return None

    @staticmethod
    def _load_foreign_keys(cursor: sqlite3.Cursor, table_name: str) -> list[Relationship]:
        cursor.execute(f"PRAGMA foreign_key_list({table_name})")
        rows = cursor.fetchall()
        if not rows:
            return []

        # Group by FK id to handle composite FKs
        fk_groups: dict[int, list] = {}
        for row in rows:
            fk_id = row["id"]
            fk_groups.setdefault(fk_id, []).append(row)

        relationships = []
        for fk_id, group in fk_groups.items():
            src_cols = tuple(r["from"] for r in group)
            tgt_cols = tuple(r["to"] for r in group)
            tgt_table = group[0]["table"]
            on_delete = group[0]["on_delete"]
            on_update = group[0]["on_update"]

            rel_type = (
                RelationshipType.COMPOSITE_FK if len(src_cols) > 1
                else RelationshipType.FOREIGN_KEY
            )
            cardinality = (
                "self-referencing" if table_name == tgt_table
                else "many-to-one"
            )

            relationships.append(Relationship(
                source_table=table_name,
                source_columns=src_cols,
                target_table=tgt_table,
                target_columns=tgt_cols,
                relationship_type=rel_type,
                cardinality=cardinality,
                confidence=1.0,
                on_delete=on_delete if on_delete != "NO ACTION" else None,
                on_update=on_update if on_update != "NO ACTION" else None,
            ))

        return relationships


# ---------------------------------------------------------------------------
# PandasSchemaInferrer
# ---------------------------------------------------------------------------

class PandasSchemaInferrer(SchemaLoader):
    """
    Infers table schema and FK relationships from Excel/CSV files.

    Convention-based FK inference:
      - Column named {table}_id or {table}Id → FK to table.id
      - Column named {table}_code → FK to table.code
      - Works across all sheets in a workbook (each sheet = a table).

    This is best-effort only; confidence is set to 0.6 for inferred FKs.
    """

    # Patterns that suggest a FK column
    FK_PATTERN = re.compile(
        r"^(?P<ref_table>[a-z][a-z0-9_]*?)(?:_id|_code|_key|Id|Code|Key)$",
        re.IGNORECASE,
    )

    def load(self, source: str) -> tuple[list[TableInfo], list[Relationship]]:
        """
        Args:
            source: Path to Excel (.xlsx/.xls) or CSV file.
        """
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")

        ext = path.suffix.lower()
        if ext == ".csv":
            dfs = {"data": pd.read_csv(source, dtype=str).fillna("")}
        elif ext in {".xlsx", ".xls"}:
            dfs = pd.read_excel(source, sheet_name=None, dtype=str)
            dfs = {k: v.fillna("") for k, v in dfs.items()}
        else:
            raise ValueError(f"Unsupported file type for schema inference: {ext}")

        # Build TableInfo for each sheet
        table_names = set(dfs.keys())
        tables: list[TableInfo] = []
        for sheet_name, df in dfs.items():
            cols = [
                ColumnInfo(
                    name=str(c),
                    data_type=self._infer_dtype(df[c]),
                    is_primary_key=(str(c).lower() in ("id", f"{sheet_name.lower()}_id")),
                )
                for c in df.columns
            ]
            tables.append(TableInfo(
                name=sheet_name,
                columns=tuple(cols),
                row_count=len(df),
            ))

        # Infer FK relationships
        relationships: list[Relationship] = []
        for table in tables:
            for col in table.columns:
                m = self.FK_PATTERN.match(col.name)
                if not m:
                    continue
                ref_table_name = m.group("ref_table")

                # Fuzzy match: find the actual sheet closest to ref_table_name
                matched = self._fuzzy_match_table(ref_table_name, table_names - {table.name})
                if matched is None:
                    continue

                # Find the PK of the referenced table
                ref_table = next((t for t in tables if t.name == matched), None)
                if ref_table is None:
                    continue

                tgt_cols = tuple(ref_table.primary_keys) or ("id",)

                is_self = table.name == matched
                relationships.append(Relationship(
                    source_table=table.name,
                    source_columns=(col.name,),
                    target_table=matched,
                    target_columns=tgt_cols,
                    relationship_type=(
                        RelationshipType.SELF_REF if is_self
                        else RelationshipType.INFERRED
                    ),
                    cardinality="self-referencing" if is_self else "many-to-one",
                    confidence=0.6,
                ))

        logger.info(
            "PandasSchemaInferrer: %d tables, %d inferred FK relationships from %s",
            len(tables), len(relationships), path.name
        )
        return tables, relationships

    @staticmethod
    def _infer_dtype(series: "pd.Series") -> str:  # type: ignore[type-arg]
        """Infer a SQL-like type name from a pandas Series."""
        sample = series.dropna()
        if sample.empty:
            return "TEXT"
        try:
            pd.to_numeric(sample)
            return "NUMERIC"
        except (ValueError, TypeError):
            pass
        try:
            pd.to_datetime(sample)
            return "DATETIME"
        except (ValueError, TypeError):
            pass
        return "TEXT"

    @staticmethod
    def _fuzzy_match_table(name: str, candidates: set[str]) -> Optional[str]:
        """
        Case-insensitive exact match first, then singular/plural variants.
        e.g. "order" matches "orders", "Orders", "Order".
        """
        name_lower = name.lower()
        for cand in candidates:
            if cand.lower() == name_lower:
                return cand
        # Try common plural/singular variants
        variants = {name_lower, name_lower + "s", name_lower.rstrip("s")}
        for cand in candidates:
            if cand.lower() in variants:
                return cand
        return None


# ---------------------------------------------------------------------------
# RelationshipDetector
# ---------------------------------------------------------------------------

class RelationshipDetector:
    """
    Augments schema analysis with additional relationship detection.

    Takes output from a SchemaLoader and:
      1. Detects junction/bridge tables (2 FK columns → 2 distinct tables)
      2. Detects self-referencing tables
      3. Deduplicates relationships
      4. Adds cardinality annotations
    """

    def analyze(
        self,
        tables: list[TableInfo],
        explicit_relationships: list[Relationship],
    ) -> SchemaAnalysisResult:
        """
        Perform full relationship analysis.

        Args:
            tables: List of TableInfo from a SchemaLoader.
            explicit_relationships: Relationships found by the loader.

        Returns:
            SchemaAnalysisResult with all detected relationships.
        """
        all_relationships = list(explicit_relationships)
        warnings: list[str] = []
        junction_tables: list[str] = []

        # Build FK map: table_name → list[Relationship]
        fk_map: dict[str, list[Relationship]] = {}
        for rel in all_relationships:
            fk_map.setdefault(rel.source_table, []).append(rel)

        # ── Detect junction tables ────────────────────────────────────────
        for table in tables:
            fks = fk_map.get(table.name, [])
            if len(fks) == 2:
                # Check all non-FK columns are minimal (just PK + 2 FKs)
                fk_col_set = set()
                for fk in fks:
                    fk_col_set.update(fk.source_columns)

                non_fk_non_pk = [
                    c for c in table.columns
                    if c.name not in fk_col_set and not c.is_primary_key
                ]
                if len(non_fk_non_pk) <= 2:  # Allow up to 2 payload columns
                    junction_tables.append(table.name)
                    # Annotate as junction (many-to-many)
                    for rel in fks:
                        # Re-create with updated cardinality
                        idx = all_relationships.index(rel)
                        all_relationships[idx] = Relationship(
                            source_table=rel.source_table,
                            source_columns=rel.source_columns,
                            target_table=rel.target_table,
                            target_columns=rel.target_columns,
                            relationship_type=RelationshipType.JUNCTION,
                            cardinality="many-to-many",
                            confidence=rel.confidence,
                            on_delete=rel.on_delete,
                            on_update=rel.on_update,
                        )
                    logger.info("Detected junction table: %s", table.name)

        # ── Detect self-referencing ───────────────────────────────────────
        for table in tables:
            for rel in fk_map.get(table.name, []):
                if rel.source_table == rel.target_table:
                    warnings.append(
                        f"Self-referencing FK detected in '{table.name}' "
                        f"({', '.join(rel.source_columns)} → {', '.join(rel.target_columns)})"
                    )

        # ── No relationships warning ──────────────────────────────────────
        if not all_relationships:
            if len(tables) > 1:
                warnings.append(
                    f"No relationships detected among {len(tables)} tables. "
                    "This may mean: (1) no FK constraints are defined, "
                    "(2) the schema uses naming conventions not recognized by the inferrer, "
                    "or (3) this is a denormalized / flat schema."
                )
            else:
                warnings.append("Only one table found — no relationships to display.")

        return SchemaAnalysisResult(
            tables=tables,
            relationships=all_relationships,
            junction_tables=junction_tables,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# ERDiagramRenderer
# ---------------------------------------------------------------------------

class ERDiagramRenderer:
    """
    Renders an interactive ER diagram using pyvis (or a text table fallback).

    The diagram uses color-coded nodes:
      - Blue  : regular tables
      - Green : tables with no incoming FKs (likely root entities)
      - Orange: junction/bridge tables
      - Red   : tables with circular self-references
    """

    # Node color palette
    COLOR_REGULAR  = "#3b82f6"   # Blue
    COLOR_ROOT     = "#22c55e"   # Green
    COLOR_JUNCTION = "#f59e0b"   # Amber
    COLOR_SELF_REF = "#ef4444"   # Red
    COLOR_EDGE_FK       = "#94a3b8"  # Gray (explicit FK)
    COLOR_EDGE_INFERRED = "#f59e0b"  # Amber (inferred)
    COLOR_EDGE_JUNCTION = "#818cf8"  # Indigo (junction)

    def render_html(
        self,
        result: SchemaAnalysisResult,
        height: str = "600px",
        notebook: bool = False,
    ) -> Optional[str]:
        """
        Render the ER diagram as an HTML string.

        Args:
            result: SchemaAnalysisResult from RelationshipDetector.
            height: CSS height of the diagram container.
            notebook: Set True when rendering in Jupyter.

        Returns:
            HTML string containing the interactive graph, or None if
            pyvis is unavailable.
        """
        if not PYVIS_AVAILABLE:
            logger.warning("pyvis not available — cannot render HTML ER diagram.")
            return None

        net = Network(
            height=height,
            width="100%",
            bgcolor="#0f172a",
            font_color="#e2e8f0",
            notebook=notebook,
        )
        net.force_atlas_2based(gravity=-50, spring_length=200)

        # Classify nodes for color coding
        tables_with_incoming_fks = {r.target_table for r in result.relationships}
        junction_set = set(result.junction_tables)

        self_ref_tables = {
            r.source_table for r in result.relationships if r.is_self_referencing
        }

        for table in result.tables:
            pk_labels = ", ".join(f"🔑 {pk}" for pk in table.primary_keys) or "—"
            col_labels = "\n".join(
                f"  {'🔑' if c.is_primary_key else '·'} {c.name} ({c.data_type})"
                for c in table.columns[:12]
            )
            if len(table.columns) > 12:
                col_labels += f"\n  ... +{len(table.columns) - 12} more"

            tooltip = (
                f"Table: {table.name}\n"
                f"PK: {pk_labels}\n"
                f"Columns ({len(table.columns)}):\n{col_labels}"
            )
            if table.row_count is not None:
                tooltip += f"\nRows: {table.row_count:,}"

            if table.name in self_ref_tables:
                color = self.COLOR_SELF_REF
            elif table.name in junction_set:
                color = self.COLOR_JUNCTION
            elif table.name not in tables_with_incoming_fks:
                color = self.COLOR_ROOT
            else:
                color = self.COLOR_REGULAR

            net.add_node(
                table.name,
                label=table.name,
                title=tooltip,
                color=color,
                size=30,
                font={"size": 14, "face": "Inter", "color": "#e2e8f0"},
                borderWidth=2,
                borderWidthSelected=4,
            )

        # Add edges
        for rel in result.relationships:
            if rel.relationship_type == RelationshipType.JUNCTION:
                edge_color = self.COLOR_EDGE_JUNCTION
                dashes = True
            elif rel.relationship_type == RelationshipType.INFERRED:
                edge_color = self.COLOR_EDGE_INFERRED
                dashes = True
            else:
                edge_color = self.COLOR_EDGE_FK
                dashes = False

            src_cols = ", ".join(rel.source_columns)
            tgt_cols = ", ".join(rel.target_columns)

            cardinality_label = {
                "many-to-one"  : "N:1",
                "one-to-one"   : "1:1",
                "many-to-many" : "N:M",
                "self-referencing": "self",
            }.get(rel.cardinality, "")

            edge_title = (
                f"Type: {rel.relationship_type}\n"
                f"{rel.source_table}.{src_cols} → {rel.target_table}.{tgt_cols}\n"
                f"Cardinality: {rel.cardinality}"
            )
            if rel.on_delete:
                edge_title += f"\nON DELETE: {rel.on_delete}"
            if rel.confidence < 1.0:
                edge_title += f"\nConfidence: {rel.confidence:.0%}"

            net.add_edge(
                rel.source_table,
                rel.target_table,
                title=edge_title,
                label=cardinality_label,
                color=edge_color,
                dashes=dashes,
                arrows={"to": {"enabled": True, "scaleFactor": 1.2}},
                width=2,
            )

        # Options
        net.set_options("""{
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "zoomView": true
            },
            "physics": {
                "enabled": true,
                "stabilization": {"iterations": 200}
            }
        }""")

        return net.generate_html(notebook=notebook)

    def render_text_table(self, result: SchemaAnalysisResult) -> str:
        """
        Render a plain-text tabular summary of relationships.
        Used as fallback when pyvis is unavailable, or for exports.
        """
        if not result.relationships:
            return "No relationships detected.\n\n" + "\n".join(result.warnings)

        lines = [
            f"{'Source Table':<25} {'FK Column(s)':<25} {'Target Table':<25} {'PK Column(s)':<20} {'Type':<15} {'Cardinality':<14} Conf",
            "-" * 130,
        ]
        for rel in result.relationships:
            src_cols = ", ".join(rel.source_columns)
            tgt_cols = ", ".join(rel.target_columns)
            lines.append(
                f"{rel.source_table:<25} {src_cols:<25} {rel.target_table:<25} "
                f"{tgt_cols:<20} {rel.relationship_type:<15} {rel.cardinality:<14} "
                f"{rel.confidence:.0%}"
            )

        if result.junction_tables:
            lines.append(f"\nJunction/Bridge tables: {', '.join(result.junction_tables)}")
        if result.warnings:
            lines.append("\nWarnings:")
            for w in result.warnings:
                lines.append(f"  ⚠ {w}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

def analyze_schema(source_path: str) -> SchemaAnalysisResult:
    """
    Convenience facade: auto-selects the correct loader and runs full analysis.

    Args:
        source_path: Path to a SQLite .db file, .xlsx, .xls, or .csv.

    Returns:
        SchemaAnalysisResult ready for rendering.

    Raises:
        ValueError: If the file type is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(source_path)
    ext = path.suffix.lower()

    if ext in {".db", ".sqlite", ".sqlite3"}:
        loader: SchemaLoader = SQLiteSchemaLoader()
    elif ext in {".xlsx", ".xls", ".csv"}:
        loader = PandasSchemaInferrer()
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            "Supported: .db, .sqlite, .sqlite3, .xlsx, .xls, .csv"
        )

    tables, explicit_rels = loader.load(source_path)
    detector = RelationshipDetector()
    result = detector.analyze(tables, explicit_rels)
    result.source_type = ext.lstrip(".")

    logger.info("Schema analysis complete: %s", result.summary())
    return result

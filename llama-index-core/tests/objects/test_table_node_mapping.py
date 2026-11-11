from uuid import UUID, NAMESPACE_DNS, uuid5

import pytest
from sqlalchemy import Column, Integer, MetaData, Table, create_engine

from llama_index.core import SQLDatabase
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.objects.base import ObjectIndex
from llama_index.core.objects.table_node_mapping import (
    SQLTableNodeMapping,
    SQLTableSchema,
)


@pytest.fixture()
def table_mapping() -> SQLTableNodeMapping:
    engine = create_engine("sqlite:///:memory:")
    metadata = MetaData()
    for name in ("sales", "sales_archive"):
        Table(name, metadata, Column("id", Integer, primary_key=True))
    metadata.create_all(engine)
    return SQLTableNodeMapping(SQLDatabase(engine))


def test_table_node_ids_preserve_field_boundaries(
    table_mapping: SQLTableNodeMapping,
) -> None:
    tables = [
        SQLTableSchema(table_name="sales", context_str="_archive records"),
        SQLTableSchema(table_name="sales_archive", context_str=" records"),
    ]
    nodes = table_mapping.to_nodes(tables)

    assert nodes[0].node_id != nodes[1].node_id
    assert [table_mapping.from_node(node) for node in nodes] == tables


def test_table_object_index_preserves_both_tables(
    table_mapping: SQLTableNodeMapping,
) -> None:
    tables = [
        SQLTableSchema(table_name="sales", context_str="_archive records"),
        SQLTableSchema(table_name="sales_archive", context_str=" records"),
    ]
    index = ObjectIndex.from_objects(tables, table_mapping, index_cls=SummaryIndex)

    assert index.as_retriever().retrieve("sales") == tables


def test_table_node_ids_distinguish_missing_and_literal_none_context(
    table_mapping: SQLTableNodeMapping,
) -> None:
    tables = [
        SQLTableSchema(table_name="sales", context_str=context)
        for context in (None, "None", "")
    ]

    assert len({table_mapping.to_node(table).node_id for table in tables}) == 3


@pytest.mark.parametrize("context", [None, "", "Sales records", 'a"b\\c\n销售'])
def test_table_node_ids_are_stable_uuids(
    table_mapping: SQLTableNodeMapping, context: str | None
) -> None:
    table = SQLTableSchema(table_name="sales", context_str=context)
    node = table_mapping.to_node(table)

    assert UUID(node.node_id).version == 5
    assert table_mapping.to_node(table.model_copy()).node_id == node.node_id
    assert table_mapping.from_node(node) == table
    assert node.metadata == (
        {"name": "sales"} if context is None else {"name": "sales", "context": context}
    )
    assert node.excluded_embed_metadata_keys == ["name", "context"]
    assert node.excluded_llm_metadata_keys == ["name", "context"]


def test_table_mapping_can_read_legacy_node_ids(
    table_mapping: SQLTableNodeMapping,
) -> None:
    table = SQLTableSchema(table_name="sales", context_str="Sales records")
    node = table_mapping.to_node(table)
    node.id_ = str(uuid5(NAMESPACE_DNS, f"{table.table_name}{table.context_str}"))

    assert table_mapping.from_node(node) == table


def test_table_node_id_encoding_is_stable(table_mapping: SQLTableNodeMapping) -> None:
    table = SQLTableSchema(table_name="sales", context_str="Sales records")

    assert (
        table_mapping.to_node(table).node_id == "10304553-04c5-5fe4-8052-ce895c06784a"
    )

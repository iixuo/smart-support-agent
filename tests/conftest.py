from datetime import date

import pytest

from support_agent import db
from support_agent.config import Settings
from support_agent.retriever import HybridRetriever
from support_agent.tools import ToolContext, build_registry


@pytest.fixture(scope="session")
def retriever():
    return HybridRetriever(use_dense=False)  # BM25 only: fast & offline


@pytest.fixture
def ctx(retriever):
    def make(customer_id="C001"):
        return ToolContext(db.init_db(":memory:"), customer_id, date(2026, 11, 18), retriever)
    return make


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def settings():
    return Settings(api_key="test", max_steps=4, timeout_s=30)

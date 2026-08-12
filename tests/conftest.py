from __future__ import annotations

import pytest

from tests.fixtures import synthetic


@pytest.fixture(scope="session")
def attributes():
    """40-row synthetic attribute table conforming to schemas.ATTRIBUTES."""
    return synthetic.make_attributes(n=40, seed=0)


@pytest.fixture(scope="session")
def attributes_large():
    """400-row table, closer to real PLP size: use where n matters (frames)."""
    return synthetic.make_attributes(n=400, seed=1)


@pytest.fixture(scope="session")
def holdout(attributes):
    return synthetic.make_holdout(attributes, seed=0)


@pytest.fixture(scope="session")
def elicitation(attributes):
    return synthetic.make_elicitation(attributes, seed=0)

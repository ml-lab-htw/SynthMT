import pytest


@pytest.fixture(scope="session")
def shared_tmp_path(tmp_path_factory):
    temp_path = tmp_path_factory.mktemp("shared")
    print(f"Shared temporary path created: {temp_path}")
    return temp_path

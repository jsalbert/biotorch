import os
import pytest


@pytest.fixture(scope='session')
def config_bp_path():
    return os.path.abspath(os.path.join('tests', 'fixtures', 'config_files', 'config_bp.yaml'))


@pytest.fixture(scope='session')
def config_usf_reproducible_path():
    return os.path.abspath(os.path.join('tests', 'fixtures', 'config_files', 'config_usf_reproducible.yaml'))

import importlib
import pytest

MODULES = [
    'synth_mt.benchmark',
    'synth_mt.config',
    'synth_mt.data_generation',
    'synth_mt.file_io',
    'synth_mt.model_hpo',
    'synth_mt.plotting',
    'synth_mt.utils',
]

def test_import_submodules():
    for module in MODULES:
        importlib.import_module(module)

def test_import_mt():
    import synth_mt


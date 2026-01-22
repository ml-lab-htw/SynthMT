import importlib
import pytest

# These are core modules that should always be importable
CORE_MODEL_MODULES = [
    "synth_mt.benchmark.models.base",
    "synth_mt.benchmark.models.anchor_point_model",
]

# These modules have optional dependencies that may not be installed
OPTIONAL_MODEL_MODULES = [
    "synth_mt.benchmark.models.cellpose_sam",
    "synth_mt.benchmark.models.cellsam",
    "synth_mt.benchmark.models.fiesta",
    "synth_mt.benchmark.models.micro_sam",
    "synth_mt.benchmark.models.sam",
    "synth_mt.benchmark.models.sam2",
    "synth_mt.benchmark.models.sam3",
    "synth_mt.benchmark.models.stardist",
    "synth_mt.benchmark.models.tardis",
]


def test_import_core_model_modules():
    """Test that core model modules can be imported."""
    for module in CORE_MODEL_MODULES:
        print(f"Testing import of core module: {module}")
        importlib.import_module(module)


@pytest.mark.parametrize("module", OPTIONAL_MODEL_MODULES)
def test_import_optional_model_modules(module):
    """Test that optional model modules can be imported (may skip if deps missing)."""
    try:
        importlib.import_module(module)
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"Optional dependency not available: {e}")

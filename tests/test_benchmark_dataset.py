import pytest

# Try to import the dataset object/class from synth_mt.benchmark.dataset
# Adjust the import if the class/object name is different

def test_import_dataset():
    try:
        from synth_mt.benchmark import dataset
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

    # Try to instantiate Dataset if it exists
    if hasattr(dataset, 'Dataset'):
        _ = dataset.Dataset()


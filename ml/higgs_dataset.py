# Compatibility shim for old model checkpoints
# When loading pickled models that expect "import higgs_dataset",
# forward that import to the new module path.

from ml.custom.higgs.higgs_dataset import HiggsDataModule

# Optionally expose everything from the real module
__all__ = ["HiggsDataModule"]
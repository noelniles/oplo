# oplo.plugins package: import submodules here so they can register
# on package import. Keep imports lazy and resilient to missing optional deps.
from . import dicom_reader  # type: ignore
from . import fits_reader  # type: ignore

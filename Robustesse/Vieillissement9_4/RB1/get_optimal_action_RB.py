"""Alias legacy du dossier ambigu ``RB1``.

Les nouveaux calculs doivent importer ``RB1_costopt_v8_020_035``. Cet alias
reste uniquement pour ne pas casser les anciens lanceurs V9_4.
"""

import importlib.util
from pathlib import Path


_TARGET = Path(__file__).resolve().parents[1] / "RB1_costopt_v8_020_035" / "get_optimal_action_RB.py"
_SPEC = importlib.util.spec_from_file_location("_genial_v94_rb1_costopt", _TARGET)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

VARIANT_ID = _MODULE.VARIANT_ID
SOC_LOW = _MODULE.SOC_LOW
SOC_HIGH = _MODULE.SOC_HIGH
get_optimal_action_RB = _MODULE.get_optimal_action_RB

"""base.py - Base class for Mars analog validation modules.

Consolidates the common patterns from:
- atacama_validation.py
- atacama_drone.py
- atacama_dust_dynamics.py
- cfd_dust_dynamics.py
- nrel_validation.py

All modules share patterns for:
- Dust similarity calibration
- Solar flux correction
- Efficiency projection
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from ..core import emit_receipt, dual_hash
from ..utils.constants import (
    ATACAMA_DUST_ANALOG_MATCH,
    ATACAMA_SOLAR_FLUX_W_M2,
    MARS_SOLAR_FLUX_W_M2,
    NREL_LAB_EFFICIENCY,
)


class MarsAnalogBase(ABC):
    """Abstract base class for Mars analog validation modules.

    Subclasses implement:
    - _load_config(): Load module-specific config
    - _validate(): Run module-specific validation
    """

    # Class-level defaults (overridden by subclasses)
    MODULE_NAME: str = ""
    TENANT_ID: str = "spaceproof-mars"

    def __init__(self):
        """Initialize Mars analog instance."""
        self._config = None

    @abstractmethod
    def _load_config(self) -> Dict[str, Any]:
        """Load module-specific configuration.

        Returns:
            Dict with module configuration
        """
        pass

    @abstractmethod
    def _validate(self, **kwargs) -> Dict[str, Any]:
        """Run module-specific validation.

        Returns:
            Dict with validation results
        """
        pass

    def load_config(self) -> Dict[str, Any]:
        """Load configuration with receipt emission.

        Returns:
            Dict with configuration
        """
        config = self._load_config()
        self._config = config

        emit_receipt(
            f"{self.MODULE_NAME.lower()}_config",
            {
                "receipt_type": f"{self.MODULE_NAME.lower()}_config",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                **{k: v for k, v in config.items() if not isinstance(v, (dict, list))},
                "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
            },
        )

        return config

    @staticmethod
    def compute_flux_correction(
        atacama_flux: float = ATACAMA_SOLAR_FLUX_W_M2,
        mars_flux: float = MARS_SOLAR_FLUX_W_M2,
    ) -> float:
        """Compute flux ratio correction factor.

        Args:
            atacama_flux: Atacama solar flux in W/m^2
            mars_flux: Mars solar flux in W/m^2

        Returns:
            Flux correction factor
        """
        if atacama_flux <= 0:
            return 0.0
        return round(mars_flux / atacama_flux, 4)

    @staticmethod
    def project_mars_efficiency(
        atacama_eff: float = NREL_LAB_EFFICIENCY,
        dust_correction: float = None,
    ) -> float:
        """Project Mars efficiency from Atacama data.

        Args:
            atacama_eff: Atacama-measured efficiency
            dust_correction: Optional flux correction factor

        Returns:
            Projected Mars efficiency
        """
        if dust_correction is None:
            dust_correction = MarsAnalogBase.compute_flux_correction()

        mars_eff = atacama_eff * dust_correction * ATACAMA_DUST_ANALOG_MATCH
        return round(mars_eff, 4)

    def run_validation(self, simulate: bool = True, **kwargs) -> Dict[str, Any]:
        """Run validation with standard receipt emission.

        Args:
            simulate: Whether to run in simulation mode
            **kwargs: Module-specific parameters

        Returns:
            Dict with validation results
        """
        config = self.load_config() if self._config is None else self._config
        result = self._validate(**kwargs)

        result["mode"] = "simulate" if simulate else "execute"
        result["config"] = config

        emit_receipt(
            f"{self.MODULE_NAME.lower()}_validation",
            {
                "receipt_type": f"{self.MODULE_NAME.lower()}_validation",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "mode": result["mode"],
                "validation_passed": result.get("validation_passed", False),
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )

        return result

    def get_info(self) -> Dict[str, Any]:
        """Get module info.

        Returns:
            Dict with module info
        """
        config = self.load_config() if self._config is None else self._config

        info = {
            "module": self.MODULE_NAME,
            "version": "1.0.0",
            "config": config,
            "constants": {
                "dust_analog_match": ATACAMA_DUST_ANALOG_MATCH,
                "atacama_solar_flux": ATACAMA_SOLAR_FLUX_W_M2,
                "mars_solar_flux": MARS_SOLAR_FLUX_W_M2,
                "flux_ratio": self.compute_flux_correction(),
            },
            "description": f"{self.MODULE_NAME} Mars analog validation",
        }

        emit_receipt(
            f"{self.MODULE_NAME.lower()}_info",
            {
                "receipt_type": f"{self.MODULE_NAME.lower()}_info",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "version": info["version"],
                "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
            },
        )

        return info

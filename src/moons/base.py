"""base.py - Abstract base class for Jovian moon hybrid modules.

Consolidates the common patterns from:
- titan_methane_hybrid.py
- europa_ice_hybrid.py
- ganymede_mag_hybrid.py
- callisto_ice.py

All moon modules share ~60% identical structure. This base class
extracts that commonality.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from ..core import emit_receipt, dual_hash
from ..utils.autonomy import compute_autonomy_from_latency, compute_combined_slo
from ..utils.constants import JOVIAN_MOONS


class MoonHybridBase(ABC):
    """Abstract base class for Jovian moon hybrid simulations.

    Subclasses implement:
    - _load_moon_config(): Load moon-specific config from spec
    - _simulate(): Run moon-specific simulation
    - _compute_autonomy(): Moon-specific autonomy calculation
    """

    # Class-level defaults (overridden by subclasses)
    MOON_NAME: str = ""
    TENANT_ID: str = ""
    DEPTH: int = 6
    SPEC_KEY: str = ""

    def __init__(self):
        """Initialize moon hybrid instance."""
        self._config = None
        self._moon_info = JOVIAN_MOONS.get(self.MOON_NAME.lower(), {})

    @property
    def autonomy_requirement(self) -> float:
        """Get autonomy requirement for this moon."""
        return self._moon_info.get("autonomy_requirement", 0.95)

    @property
    def latency_min(self) -> list:
        """Get latency bounds [min, max] in minutes."""
        return self._moon_info.get("latency_min", [33, 53])

    @property
    def earth_callback_max_pct(self) -> float:
        """Get maximum Earth callback percentage."""
        return self._moon_info.get("earth_callback_max_pct", 0.05)

    @abstractmethod
    def _load_moon_config(self) -> Dict[str, Any]:
        """Load moon-specific configuration.

        Returns:
            Dict with moon configuration
        """
        pass

    @abstractmethod
    def _simulate(self, duration_days: int, **kwargs) -> Dict[str, Any]:
        """Run moon-specific simulation.

        Args:
            duration_days: Simulation duration in days
            **kwargs: Moon-specific parameters

        Returns:
            Dict with simulation results
        """
        pass

    def load_config(self) -> Dict[str, Any]:
        """Load moon configuration with receipt emission.

        Returns:
            Dict with moon configuration
        """
        config = self._load_moon_config()
        self._config = config

        emit_receipt(
            f"{self.MOON_NAME.lower()}_config",
            {
                "receipt_type": f"{self.MOON_NAME.lower()}_config",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                **{k: v for k, v in config.items() if not isinstance(v, dict)},
                "payload_hash": dual_hash(json.dumps(config, sort_keys=True)),
            },
        )

        return config

    def compute_autonomy(self, duration_hours: float) -> Dict[str, Any]:
        """Compute autonomy based on latency constraints.

        This is the common pattern used across all moon modules.

        Args:
            duration_hours: Duration of operation in hours

        Returns:
            Dict with autonomy metrics
        """
        result = compute_autonomy_from_latency(
            duration_hours,
            self.latency_min,
            self.earth_callback_max_pct,
        )

        result["autonomy_met"] = (
            result["autonomy_achieved"] >= self.autonomy_requirement
        )
        result["autonomy_requirement"] = self.autonomy_requirement

        emit_receipt(
            f"{self.MOON_NAME.lower()}_autonomy",
            {
                "receipt_type": f"{self.MOON_NAME.lower()}_autonomy",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                **result,
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )

        return result

    def run_simulation(self, duration_days: int = 30, **kwargs) -> Dict[str, Any]:
        """Run simulation with standard receipt emission.

        Args:
            duration_days: Simulation duration in days
            **kwargs: Moon-specific parameters

        Returns:
            Dict with simulation results
        """
        config = self.load_config() if self._config is None else self._config
        result = self._simulate(duration_days, **kwargs)

        # Compute autonomy if not already present
        if "autonomy_achieved" not in result:
            autonomy = self.compute_autonomy(duration_days * 24)
            result["autonomy_achieved"] = autonomy["autonomy_achieved"]
            result["autonomy_met"] = autonomy["autonomy_met"]

        result["config"] = config

        emit_receipt(
            f"{self.MOON_NAME.lower()}_simulation",
            {
                "receipt_type": f"{self.MOON_NAME.lower()}_simulation",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "duration_days": duration_days,
                "autonomy_achieved": result.get("autonomy_achieved", 0),
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )

        return result

    def run_hybrid(
        self,
        fractal_result: Dict[str, Any],
        duration_days: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run hybrid fractal + moon simulation.

        This is the common pattern for d*_*_hybrid() functions.

        Args:
            fractal_result: Result from d*_recursive_fractal()
            duration_days: Simulation duration
            **kwargs: Moon-specific parameters

        Returns:
            Dict with hybrid results
        """
        # Run moon simulation
        moon_result = self.run_simulation(duration_days, **kwargs)

        # Compute combined SLO
        from ..utils.spec_loader import get_all_depth_constants

        depth_constants = get_all_depth_constants(self.DEPTH)

        combined_slo = compute_combined_slo(
            fractal_result,
            moon_result,
            depth_constants["alpha_floor"],
            self.autonomy_requirement,
        )

        result = {
            f"d{self.DEPTH}_result": {
                "tree_size": fractal_result.get("tree_size"),
                "base_alpha": fractal_result.get("base_alpha"),
                "depth": fractal_result.get("depth"),
                "eff_alpha": fractal_result.get("eff_alpha"),
                "floor_met": fractal_result.get("floor_met"),
                "target_met": fractal_result.get("target_met"),
                "instability": fractal_result.get("instability", 0.0),
            },
            f"{self.MOON_NAME.lower()}_result": {
                k: v
                for k, v in moon_result.items()
                if k not in ["config"] and not isinstance(v, dict)
            },
            "combined_slo": combined_slo,
            "gate": "t24h",
        }

        emit_receipt(
            f"d{self.DEPTH}_{self.MOON_NAME.lower()}_hybrid",
            {
                "receipt_type": f"d{self.DEPTH}_{self.MOON_NAME.lower()}_hybrid",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "tree_size": fractal_result.get("tree_size"),
                "eff_alpha": fractal_result.get("eff_alpha"),
                "autonomy_achieved": moon_result.get("autonomy_achieved"),
                "all_targets_met": combined_slo["all_targets_met"],
                "payload_hash": dual_hash(json.dumps(result, sort_keys=True)),
            },
        )

        return result

    def get_info(self) -> Dict[str, Any]:
        """Get moon module info.

        Returns:
            Dict with module info
        """
        config = self.load_config() if self._config is None else self._config

        from ..utils.spec_loader import get_all_depth_constants

        depth_constants = get_all_depth_constants(self.DEPTH)

        info = {
            "module": f"{self.MOON_NAME.lower()}_hybrid",
            "version": "1.0.0",
            "config": config,
            "autonomy": {
                "requirement": self.autonomy_requirement,
                "latency_min": self.latency_min,
                "earth_callback_max_pct": self.earth_callback_max_pct,
            },
            f"d{self.DEPTH}_integration": {
                "alpha_floor": depth_constants["alpha_floor"],
                "tree_min": depth_constants["tree_min"],
            },
            "description": f"{self.MOON_NAME} ISRU simulation with D{self.DEPTH} integration",
        }

        emit_receipt(
            f"{self.MOON_NAME.lower()}_info",
            {
                "receipt_type": f"{self.MOON_NAME.lower()}_info",
                "tenant_id": self.TENANT_ID,
                "ts": datetime.utcnow().isoformat() + "Z",
                "version": info["version"],
                "autonomy_requirement": self.autonomy_requirement,
                "payload_hash": dual_hash(json.dumps(info, sort_keys=True)),
            },
        )

        return info

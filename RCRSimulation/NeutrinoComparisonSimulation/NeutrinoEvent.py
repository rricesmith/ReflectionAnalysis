"""Lightweight data model for neutrino simulation events.

Stores per-triggered-event signal properties (arrival angle, polarization)
separated by trigger group (shallow LPDA vs deep phased array) for comparison
with RCR background distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


# Channel group definitions matching gen2_hybrid_2021.json
SHALLOW_CHANNELS = [0, 1, 2, 3]       # LPDAs at -3m
PA_8CH_CHANNELS = [4, 5, 6, 7, 8, 9, 10, 11]  # Phased array at -143 to -150m
PA_4CH_CHANNELS = [8, 9, 10, 11]       # 4-channel PA subset


@dataclass
class NeutrinoEvent:
    """Container for a single triggered neutrino event.

    Attributes:
        event_id: Event index (matches HDF5 array index for weight lookup)
        energy: Neutrino energy in eV
        nu_zenith: Neutrino direction zenith angle (rad)
        nu_azimuth: Neutrino direction azimuth angle (rad)
        weight: NuRadioMC event weight (Earth absorption + interaction probability)
        triggers: List of trigger names that fired for this event

        lpda_arrival_zenith: Average signal arrival zenith at LPDA channels (rad)
        lpda_polarization: Average polarization angle at LPDA channels (rad)
        pa_arrival_zenith: Average signal arrival zenith at PA channels (rad)
        pa_polarization: Average polarization angle at PA channels (rad)
    """
    event_id: int
    energy: float
    nu_zenith: float
    nu_azimuth: float
    weight: float = 1.0
    triggers: List[str] = field(default_factory=list)

    lpda_arrival_zenith: Optional[float] = None
    lpda_polarization: Optional[float] = None
    pa_arrival_zenith: Optional[float] = None
    pa_polarization: Optional[float] = None

    def has_lpda_trigger(self) -> bool:
        """Check if any LPDA trigger fired."""
        return any("LPDA" in t for t in self.triggers)

    def has_pa_trigger(self) -> bool:
        """Check if any phased array trigger fired."""
        return any("PA_" in t for t in self.triggers)

    def get_arrival_zenith(self, trigger_type: str) -> Optional[float]:
        """Get arrival zenith for a trigger group.

        Args:
            trigger_type: 'lpda' or 'pa'
        """
        if trigger_type == "lpda":
            return self.lpda_arrival_zenith
        elif trigger_type == "pa":
            return self.pa_arrival_zenith
        return None

    def get_polarization(self, trigger_type: str) -> Optional[float]:
        """Get polarization angle for a trigger group.

        Args:
            trigger_type: 'lpda' or 'pa'
        """
        if trigger_type == "lpda":
            return self.lpda_polarization
        elif trigger_type == "pa":
            return self.pa_polarization
        return None

    def get_arrival_zenith_deg(self, trigger_type: str) -> Optional[float]:
        """Get arrival zenith in degrees."""
        val = self.get_arrival_zenith(trigger_type)
        return np.rad2deg(val) if val is not None else None

    def get_polarization_deg(self, trigger_type: str) -> Optional[float]:
        """Get polarization angle in degrees."""
        val = self.get_polarization(trigger_type)
        return np.rad2deg(val) if val is not None else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "energy": self.energy,
            "nu_zenith": self.nu_zenith,
            "nu_azimuth": self.nu_azimuth,
            "weight": self.weight,
            "triggers": self.triggers,
            "lpda_arrival_zenith": self.lpda_arrival_zenith,
            "lpda_polarization": self.lpda_polarization,
            "pa_arrival_zenith": self.pa_arrival_zenith,
            "pa_polarization": self.pa_polarization,
        }

    def __repr__(self) -> str:
        trigs = ", ".join(self.triggers) if self.triggers else "none"
        return (
            f"NeutrinoEvent(id={self.event_id}, E={self.energy:.2e}eV, "
            f"zen={np.rad2deg(self.nu_zenith):.1f}deg, triggers=[{trigs}])"
        )

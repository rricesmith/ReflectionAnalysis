"""RCR Event Object for tracking multi-station triggers in reflected cosmic ray simulations.

This module provides the RCREvent class which stores event information and trigger
results for both direct and reflected stations in a single pass simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from NuRadioReco.utilities import units
from NuRadioReco.framework.parameters import eventParameters as evtp
from NuRadioReco.framework.parameters import showerParameters as shp
from NuRadioReco.framework.parameters import stationParameters as stnp


# Station ID convention: direct < 100, reflected >= 100
REFLECTED_STATION_OFFSET = 100


@dataclass
class RCREvent:
    """Container for RCR simulation event data with multi-station trigger tracking.

    Attributes:
        event_id: Unique event identifier
        coreas_x: CoREAS x position (m)
        coreas_y: CoREAS y position (m)
        energy: Primary particle energy (eV)
        zenith: Shower zenith angle (rad)
        azimuth: Shower azimuth angle (rad)
        layer_dB: Reflection layer loss in dB
        station_triggers: Dict mapping trigger_name -> list of station IDs that triggered
        station_snr: Dict mapping station_id -> SNR value
        recon_zenith: Dict mapping station_id -> reconstructed zenith
        recon_azimuth: Dict mapping station_id -> reconstructed azimuth
    """
    event_id: int
    coreas_x: float
    coreas_y: float
    energy: float
    zenith: float
    azimuth: float
    layer_dB: Optional[float] = None

    # Trigger tracking: trigger_name -> list of station IDs
    station_triggers: Dict[str, List[int]] = field(default_factory=dict)

    # Per-station measurements
    station_snr: Dict[int, float] = field(default_factory=dict)
    recon_zenith: Dict[int, Optional[float]] = field(default_factory=dict)
    recon_azimuth: Dict[int, Optional[float]] = field(default_factory=dict)

    # Event rate weights: weight_name -> weight value (for analysis)
    weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_nuradio_event(cls, event, layer_dB: Optional[float] = None) -> "RCREvent":
        """Create RCREvent from a NuRadioReco event object.

        Args:
            event: NuRadioReco event object
            layer_dB: Reflection layer loss in dB

        Returns:
            New RCREvent instance populated from the NuRadioReco event
        """
        sim_shower = event.get_sim_shower(0)

        rcr_event = cls(
            event_id=event.get_id(),
            coreas_x=float(event.get_parameter(evtp.coreas_x) / units.m),
            coreas_y=float(event.get_parameter(evtp.coreas_y) / units.m),
            energy=float(sim_shower[shp.energy] / units.eV),
            zenith=float(sim_shower[shp.zenith] / units.rad),
            azimuth=float(sim_shower[shp.azimuth] / units.rad),
            layer_dB=layer_dB,
        )

        # Extract trigger and reconstruction info from all stations
        for station in event.get_stations():
            station_id = station.get_id()

            if station.has_triggered():
                # Get all trigger names and check which ones fired
                for trigger in station.get_triggers().values():
                    trigger_name = trigger.get_name()
                    if trigger.has_triggered():
                        rcr_event.add_trigger(trigger_name, station_id)

                # Try to get reconstructed angles
                try:
                    rcr_event.recon_zenith[station_id] = float(
                        station.get_parameter(stnp.zenith) / units.rad
                    )
                    rcr_event.recon_azimuth[station_id] = float(
                        station.get_parameter(stnp.azimuth) / units.rad
                    )
                except (KeyError, TypeError):
                    rcr_event.recon_zenith[station_id] = None
                    rcr_event.recon_azimuth[station_id] = None

        return rcr_event

    def add_trigger(self, trigger_name: str, station_id: int) -> None:
        """Record that a station triggered for a given trigger type."""
        if trigger_name not in self.station_triggers:
            self.station_triggers[trigger_name] = []
        if station_id not in self.station_triggers[trigger_name]:
            self.station_triggers[trigger_name].append(station_id)

    def has_triggered(self, trigger_name: str, station_id: Optional[int] = None) -> bool:
        """Check if event triggered.

        Args:
            trigger_name: Name of the trigger to check
            station_id: Optional station ID. If None, returns True if any station triggered.
        """
        if trigger_name not in self.station_triggers:
            return False
        if station_id is None:
            return len(self.station_triggers[trigger_name]) > 0
        return station_id in self.station_triggers[trigger_name]

    def direct_triggers(self, trigger_name: str) -> List[int]:
        """Get list of direct station IDs that triggered (station_id < 100)."""
        if trigger_name not in self.station_triggers:
            return []
        return [s for s in self.station_triggers[trigger_name] if s < REFLECTED_STATION_OFFSET]

    def reflected_triggers(self, trigger_name: str) -> List[int]:
        """Get list of reflected station IDs that triggered (station_id >= 100)."""
        if trigger_name not in self.station_triggers:
            return []
        return [s for s in self.station_triggers[trigger_name] if s >= REFLECTED_STATION_OFFSET]

    def has_direct_trigger(self, trigger_name: str) -> bool:
        """Check if any direct station triggered."""
        return len(self.direct_triggers(trigger_name)) > 0

    def has_reflected_trigger(self, trigger_name: str) -> bool:
        """Check if any reflected station triggered."""
        return len(self.reflected_triggers(trigger_name)) > 0

    def get_energy(self) -> float:
        """Get primary energy in eV."""
        return self.energy

    def get_angles(self) -> tuple[float, float]:
        """Get (zenith, azimuth) in radians."""
        return self.zenith, self.azimuth

    def get_angles_deg(self) -> tuple[float, float]:
        """Get (zenith, azimuth) in degrees."""
        return np.rad2deg(self.zenith), np.rad2deg(self.azimuth)

    def get_coreas_position(self) -> tuple[float, float]:
        """Get (x, y) CoREAS position in meters."""
        return self.coreas_x, self.coreas_y

    def set_snr(self, station_id: int, snr: float) -> None:
        """Set SNR for a station."""
        self.station_snr[station_id] = snr

    def get_snr(self, station_id: int) -> Optional[float]:
        """Get SNR for a station, or None if not set."""
        return self.station_snr.get(station_id)

    def all_triggered_stations(self, trigger_name: str) -> List[int]:
        """Get all station IDs that triggered for a given trigger name."""
        return self.station_triggers.get(trigger_name, [])

    def all_trigger_names(self) -> List[str]:
        """Get list of all trigger names that have at least one station triggered."""
        return [name for name, stations in self.station_triggers.items() if stations]

    def has_any_trigger(self) -> bool:
        """Check if any trigger fired for any station."""
        return any(len(stations) > 0 for stations in self.station_triggers.values())

    def set_weight(self, weight: float, weight_name: str) -> None:
        """Set event rate weight for a specific weight category.

        Args:
            weight: The weight value (typically evts/yr contribution)
            weight_name: Name identifying the weight type (e.g., 'direct', 'reflected')
        """
        self.weights[weight_name] = weight

    def get_weight(self, weight_name: str) -> float:
        """Get event rate weight for a specific weight category.

        Args:
            weight_name: Name identifying the weight type

        Returns:
            Weight value, or 0.0 if not set
        """
        return self.weights.get(weight_name, 0.0)

    def get_radius(self) -> float:
        """Get radial distance from station (at origin) in meters."""
        return np.sqrt(self.coreas_x**2 + self.coreas_y**2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "coreas_x": self.coreas_x,
            "coreas_y": self.coreas_y,
            "energy": self.energy,
            "zenith": self.zenith,
            "azimuth": self.azimuth,
            "layer_dB": self.layer_dB,
            "station_triggers": self.station_triggers,
            "station_snr": self.station_snr,
            "recon_zenith": self.recon_zenith,
            "recon_azimuth": self.recon_azimuth,
            "weights": self.weights,
        }

    def __repr__(self) -> str:
        n_direct = sum(len(self.direct_triggers(t)) for t in self.station_triggers)
        n_reflected = sum(len(self.reflected_triggers(t)) for t in self.station_triggers)
        return (
            f"RCREvent(id={self.event_id}, E={self.energy:.2e}eV, "
            f"zen={np.rad2deg(self.zenith):.1f}°, "
            f"direct_triggers={n_direct}, reflected_triggers={n_reflected})"
        )

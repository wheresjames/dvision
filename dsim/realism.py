"""Environment and sensor realism for the simulator.

Everything here exists because a client has to behave differently when it is
switched on: GPS that can be denied, estimators that can go invalid, wind that
a position controller has to fight, telemetry that arrives late, sensors that
do not read exactly, a battery that ends a mission, and a fence the vehicle
will not cross. Physics stays in ``dsim.py``; this module owns the environment
acting on it and the difference between the truth and what is published.

The random processes are seeded and correlated in time rather than white:
a GPS fix wanders, a barometer drifts, gusts build and fade. Uncorrelated
noise on every sample is easy to filter and teaches a client nothing.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

# fix_type follows GPS_FIX_TYPE in MAVLink: 0 none, 2 2D, 3 3D, 4 RTK fixed.
GPS_MODES: dict[str, dict[str, float]] = {
    "off": {"fix_type": 0, "satellites": 0, "hdop": 99.99, "vdop": 99.99,
            "noise_m": 0.0},
    "degraded": {"fix_type": 2, "satellites": 6, "hdop": 2.50, "vdop": 3.20,
                 "noise_m": 2.5},
    "good": {"fix_type": 3, "satellites": 12, "hdop": 0.90, "vdop": 1.30,
             "noise_m": 0.8},
    "rtk": {"fix_type": 4, "satellites": 20, "hdop": 0.60, "vdop": 0.80,
            "noise_m": 0.05},
}

#: A 3-D fix or better is what makes the global position estimate usable.
GPS_USABLE_FIX = 3

# Published-state noise. ``light`` stays inside the arrival gates the committed
# tours use; ``heavy`` does not, and a tour flown under it has to widen its own
# gates rather than have the follower quietly widen them.
SENSOR_NOISE_PROFILES: dict[str, dict[str, float]] = {
    "none": {"heading_deg": 0.0, "altitude_m": 0.0, "altitude_drift_m": 0.0,
             "velocity_mps": 0.0},
    "light": {"heading_deg": 0.20, "altitude_m": 0.005,
              "altitude_drift_m": 0.010, "velocity_mps": 0.005},
    "heavy": {"heading_deg": 3.00, "altitude_m": 0.100,
              "altitude_drift_m": 0.300, "velocity_mps": 0.060},
}

GEOFENCE_ACTIONS = ("hold", "rtl")

# Time constants for the correlated processes, in seconds.
_GPS_TAU_S = 12.0
_GUST_TAU_S = 3.0
_BARO_TAU_S = 30.0

#: Every realism knob and its default. The CLI, the vehicle profile file and
#: the runtime commands all speak these names, so a report can be reproduced
#: from the values it publishes.
REALISM_DEFAULTS: dict[str, Any] = {
    "gps": "good",
    "gps_noise_m": None,          # None = the mode's own figure
    "local_estimator": "on",
    "wind_mps": 0.0,
    "wind_dir_deg": 0.0,
    "wind_gust_mps": 0.0,
    "telemetry_latency_ms": 0.0,
    "telemetry_jitter_ms": 0.0,
    "sensor_noise": "none",
    "battery_failsafe_pct": 0.0,
    "battery_drain_pct_s": 0.01,
    "geofence": "",
    "geofence_action": "hold",
    "realism_seed": 1234,
}


@dataclass(frozen=True)
class Geofence:
    """An axis-aligned box in map metres, with an optional ceiling."""

    x0: float
    y0: float
    x1: float
    y1: float
    max_alt_m: float | None = None

    @classmethod
    def parse(cls, text: str) -> "Geofence | None":
        text = (text or "").strip()
        if not text:
            return None
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) not in (4, 5):
            raise ValueError(
                "geofence must be x0,y0,x1,y1 or x0,y0,x1,y1,max_alt_m")
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"geofence must be numeric: {exc}") from exc
        return cls(min(values[0], values[2]), min(values[1], values[3]),
                   max(values[0], values[2]), max(values[1], values[3]),
                   values[4] if len(values) == 5 else None)

    def contains(self, x: float, y: float, z: float) -> bool:
        if not (self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1):
            return False
        return self.max_alt_m is None or z <= self.max_alt_m

    def describe(self) -> str:
        box = f"{self.x0:g},{self.y0:g},{self.x1:g},{self.y1:g}"
        return box if self.max_alt_m is None else f"{box},{self.max_alt_m:g}"


class _Correlated:
    """One first-order noise process: it wanders and returns, it does not jump."""

    def __init__(self, rng: random.Random, tau_s: float, sigma: float) -> None:
        self._rng = rng
        self._tau = max(tau_s, 1e-3)
        self.sigma = sigma
        self.value = 0.0

    def update(self, dt: float) -> float:
        if self.sigma <= 0.0:
            self.value = 0.0
            return 0.0
        alpha = 1.0 - math.exp(-dt / self._tau)
        # Scaled so the stationary spread stays near sigma whatever dt is.
        self.value += (-self.value * alpha
                       + self._rng.gauss(0.0, self.sigma) * math.sqrt(2.0 * alpha))
        self.value = max(-3.0 * self.sigma, min(3.0 * self.sigma, self.value))
        return self.value


class TelemetryDelay:
    """A short delay ring: what a client reads is what was true a moment ago."""

    def __init__(self, rng: random.Random, latency_ms: float,
                 jitter_ms: float) -> None:
        self._rng = rng
        self.latency_s = max(0.0, latency_ms) / 1000.0
        self.jitter_s = max(0.0, jitter_ms) / 1000.0
        self._queue: list[tuple[float, dict[str, str]]] = []
        self._last_release = -math.inf

    @property
    def enabled(self) -> bool:
        return self.latency_s > 0.0 or self.jitter_s > 0.0

    def push(self, now: float, values: dict[str, str]) -> None:
        delay = self.latency_s
        if self.jitter_s > 0.0:
            delay += self._rng.uniform(-self.jitter_s, self.jitter_s)
        # Jitter may not reorder samples: a client that reads an older state
        # after a newer one is being lied to about time, not delayed.
        release = max(now + max(0.0, delay), self._last_release)
        self._last_release = release
        self._queue.append((release, values))

    def release(self, now: float) -> dict[str, str] | None:
        latest: dict[str, str] | None = None
        while self._queue and self._queue[0][0] <= now:
            latest = self._queue.pop(0)[1]
        return latest


@dataclass
class Realism:
    """The configured environment, its live state, and what it publishes."""

    gps_mode: str = "good"
    gps_noise_m: float | None = None
    local_estimator: bool = True
    wind_mps: float = 0.0
    wind_dir_deg: float = 0.0
    wind_gust_mps: float = 0.0
    sensor_noise: str = "none"
    battery_failsafe_pct: float = 0.0
    battery_drain_pct_s: float = 0.01
    geofence: Geofence | None = None
    geofence_action: str = "hold"
    telemetry: TelemetryDelay | None = None
    seed: int = 1234
    # Runtime estimator faults, set by command; absent means healthy.
    faults: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()
        self._rng = random.Random(self.seed)
        self._gust = _Correlated(self._rng, _GUST_TAU_S, self.wind_gust_mps)
        self._gps_north = _Correlated(self._rng, _GPS_TAU_S, self._gps_sigma())
        self._gps_east = _Correlated(self._rng, _GPS_TAU_S, self._gps_sigma())
        self._gps_alt = _Correlated(self._rng, _GPS_TAU_S, self._gps_sigma() * 1.6)
        profile = SENSOR_NOISE_PROFILES[self.sensor_noise]
        self._baro_drift = _Correlated(self._rng, _BARO_TAU_S,
                                       profile["altitude_drift_m"])
        if self.telemetry is None:
            self.telemetry = TelemetryDelay(self._rng, 0.0, 0.0)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        if self.gps_mode not in GPS_MODES:
            raise ValueError(f"unknown gps mode: {self.gps_mode}")
        if self.sensor_noise not in SENSOR_NOISE_PROFILES:
            raise ValueError(f"unknown sensor noise profile: {self.sensor_noise}")
        if self.geofence_action not in GEOFENCE_ACTIONS:
            raise ValueError(f"unknown geofence action: {self.geofence_action}")
        if self.wind_mps < 0.0 or self.wind_gust_mps < 0.0:
            raise ValueError("wind speeds must not be negative")
        if self.gps_noise_m is not None and self.gps_noise_m < 0.0:
            raise ValueError("GPS noise must not be negative")
        if not 0.0 <= self.battery_failsafe_pct <= 100.0:
            raise ValueError("battery failsafe percent must be within 0..100")
        if self.battery_drain_pct_s < 0.0:
            raise ValueError("battery drain must not be negative")

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "Realism":
        merged = dict(REALISM_DEFAULTS)
        merged.update({k: v for k, v in settings.items() if k in REALISM_DEFAULTS})
        if (float(merged["telemetry_latency_ms"]) < 0.0
                or float(merged["telemetry_jitter_ms"]) < 0.0):
            raise ValueError("telemetry latency and jitter must not be negative")
        rng = random.Random(int(merged["realism_seed"]))
        return cls(
            gps_mode=str(merged["gps"]),
            gps_noise_m=(None if merged["gps_noise_m"] is None
                         else float(merged["gps_noise_m"])),
            local_estimator=str(merged["local_estimator"]).lower() in ("on", "1", "true"),
            wind_mps=float(merged["wind_mps"]),
            wind_dir_deg=float(merged["wind_dir_deg"]),
            wind_gust_mps=float(merged["wind_gust_mps"]),
            sensor_noise=str(merged["sensor_noise"]),
            battery_failsafe_pct=float(merged["battery_failsafe_pct"]),
            battery_drain_pct_s=float(merged["battery_drain_pct_s"]),
            geofence=Geofence.parse(str(merged["geofence"])),
            geofence_action=str(merged["geofence_action"]),
            telemetry=TelemetryDelay(rng, float(merged["telemetry_latency_ms"]),
                                     float(merged["telemetry_jitter_ms"])),
            seed=int(merged["realism_seed"]),
        )

    def settings(self) -> dict[str, Any]:
        """The current configuration in the CLI's own names, ready to round-trip.

        Distinct from :meth:`describe`, which resolves ``gps_noise_m`` to the
        figure actually in use for a report. This one keeps it as the override
        it is, so feeding the result back through :meth:`apply` changes
        nothing -- which is what lets a form be populated from it.
        """
        return {
            "gps": self.gps_mode,
            "gps_noise_m": self.gps_noise_m,
            "local_estimator": "on" if self.local_estimator else "off",
            "wind_mps": self.wind_mps,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_gust_mps": self.wind_gust_mps,
            "telemetry_latency_ms": self.telemetry.latency_s * 1000.0,
            "telemetry_jitter_ms": self.telemetry.jitter_s * 1000.0,
            "sensor_noise": self.sensor_noise,
            "battery_failsafe_pct": self.battery_failsafe_pct,
            "battery_drain_pct_s": self.battery_drain_pct_s,
            "geofence": "" if self.geofence is None else self.geofence.describe(),
            "geofence_action": self.geofence_action,
            "realism_seed": self.seed,
        }

    def apply(self, **settings: Any) -> None:
        """Change the environment while the simulation runs.

        Takes the same names the CLI and the profile file use, and any subset
        of them. Everything is validated first, by building a throwaway
        instance from the merged settings, so a rejected change leaves the
        environment exactly as it was rather than half applied.

        The noise processes keep their current wander across a change: turning
        the wind up should not also teleport the gust, and re-tuning a sigma is
        not the same as re-rolling the dice. Changing ``realism_seed`` is how
        you ask for the dice back.
        """
        unknown = sorted(set(settings) - set(REALISM_DEFAULTS))
        if unknown:
            raise ValueError(f"unknown realism setting: {', '.join(unknown)}")
        candidate = Realism.from_settings({**self.settings(), **settings})

        reseeded = candidate.seed != self.seed
        for name in ("gps_mode", "gps_noise_m", "local_estimator", "wind_mps",
                     "wind_dir_deg", "wind_gust_mps", "sensor_noise",
                     "battery_failsafe_pct", "battery_drain_pct_s", "geofence",
                     "geofence_action", "seed"):
            setattr(self, name, getattr(candidate, name))
        if reseeded:
            self._rng = random.Random(self.seed)
            self._gust = _Correlated(self._rng, _GUST_TAU_S, 0.0)
            self._gps_north = _Correlated(self._rng, _GPS_TAU_S, 0.0)
            self._gps_east = _Correlated(self._rng, _GPS_TAU_S, 0.0)
            self._gps_alt = _Correlated(self._rng, _GPS_TAU_S, 0.0)
            self._baro_drift = _Correlated(self._rng, _BARO_TAU_S, 0.0)
        self._retune()
        self.telemetry.latency_s = candidate.telemetry.latency_s
        self.telemetry.jitter_s = candidate.telemetry.jitter_s

    def _retune(self) -> None:
        """Point every noise process at the sigma its setting now implies."""
        sigma = self._gps_sigma()
        self._gps_north.sigma = self._gps_east.sigma = sigma
        self._gps_alt.sigma = sigma * 1.6
        self._gust.sigma = self.wind_gust_mps
        self._baro_drift.sigma = \
            SENSOR_NOISE_PROFILES[self.sensor_noise]["altitude_drift_m"]

    def _gps_sigma(self) -> float:
        if self.gps_noise_m is not None:
            return max(0.0, self.gps_noise_m)
        return float(GPS_MODES[self.gps_mode]["noise_m"])

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------

    def set_gps(self, mode: str, noise_m: float | None = None) -> None:
        """Deny or restore GPS mid-flight, as a jammer or a canyon would."""
        if mode not in GPS_MODES:
            raise ValueError(f"unknown gps mode: {mode}")
        self.gps_mode = mode
        if noise_m is not None:
            self.gps_noise_m = max(0.0, float(noise_m))
        self._retune()

    def set_estimator(self, **flags: bool) -> None:
        """Fault or restore an estimator, independently of what caused it."""
        for name, value in flags.items():
            if name not in ("attitude", "local", "global", "velocity"):
                raise ValueError(f"unknown estimator: {name}")
            self.faults[name] = bool(value)

    # ------------------------------------------------------------------
    # Per-tick environment
    # ------------------------------------------------------------------

    def update(self, dt: float) -> None:
        self._gust.update(dt)
        self._gps_north.update(dt)
        self._gps_east.update(dt)
        self._gps_alt.update(dt)
        self._baro_drift.update(dt)

    def wind_vector(self) -> tuple[float, float]:
        """Wind as a map-frame (x, y) velocity, gusts included.

        ``wind_dir_deg`` is the compass direction the wind blows *from*, the
        convention every weather report and pilot uses; the air therefore moves
        toward the reciprocal.
        """
        speed = self.wind_mps + self._gust.value
        if speed == 0.0:
            return 0.0, 0.0
        toward = math.radians((self.wind_dir_deg + 180.0) % 360.0)
        return speed * math.sin(toward), -speed * math.cos(toward)

    @property
    def wind_speed_mps(self) -> float:
        return self.wind_mps + self._gust.value

    # ------------------------------------------------------------------
    # Sensors and estimators
    # ------------------------------------------------------------------

    def gps_fix(self) -> dict[str, float]:
        return dict(GPS_MODES[self.gps_mode])

    def gps_offset_m(self) -> tuple[float, float, float]:
        """North/east/up error currently present in the published fix."""
        if GPS_MODES[self.gps_mode]["fix_type"] == 0:
            return 0.0, 0.0, 0.0
        return self._gps_north.value, self._gps_east.value, self._gps_alt.value

    def estimators(self) -> dict[str, bool]:
        """Live estimator validity. Arming does not make an estimator valid."""
        global_ok = GPS_MODES[self.gps_mode]["fix_type"] >= GPS_USABLE_FIX
        attitude = self.faults.get("attitude", True)
        # Local validity is independent of GPS on purpose: that is the
        # GPS-denied case worth testing, where VIO or flow still holds a pose.
        local = self.local_estimator and self.faults.get("local", True)
        global_valid = global_ok and self.faults.get("global", True)
        velocity = (local or global_valid) and self.faults.get("velocity", True)
        return {"attitude": attitude, "local": local, "global": global_valid,
                "velocity": velocity}

    def noisy_heading_deg(self, heading_deg: float) -> float:
        sigma = SENSOR_NOISE_PROFILES[self.sensor_noise]["heading_deg"]
        if sigma <= 0.0:
            return heading_deg
        return (heading_deg + self._rng.gauss(0.0, sigma)) % 360.0

    def noisy_altitude_m(self, altitude_m: float) -> float:
        profile = SENSOR_NOISE_PROFILES[self.sensor_noise]
        if profile["altitude_m"] <= 0.0 and profile["altitude_drift_m"] <= 0.0:
            return altitude_m
        return altitude_m + self._baro_drift.value + self._rng.gauss(
            0.0, profile["altitude_m"])

    def noisy_velocity_mps(self, velocity_mps: float) -> float:
        sigma = SENSOR_NOISE_PROFILES[self.sensor_noise]["velocity_mps"]
        return velocity_mps if sigma <= 0.0 else velocity_mps + self._rng.gauss(0.0, sigma)

    # ------------------------------------------------------------------
    # Safety
    # ------------------------------------------------------------------

    def outside_geofence(self, x: float, y: float, z: float) -> bool:
        return self.geofence is not None and not self.geofence.contains(x, y, z)

    def battery_exhausted(self, battery_pct: float) -> bool:
        return (self.battery_failsafe_pct > 0.0
                and battery_pct <= self.battery_failsafe_pct)

    def status_fields(self) -> dict[str, str]:
        fix = self.gps_fix()
        estimators = self.estimators()
        return {
            "gps.fix_type": str(int(fix["fix_type"])),
            "gps.satellites": str(int(fix["satellites"])),
            "gps.hdop": f"{fix['hdop']:.2f}",
            "gps.vdop": f"{fix['vdop']:.2f}",
            "est.attitude_valid": "1" if estimators["attitude"] else "0",
            "est.local_position_valid": "1" if estimators["local"] else "0",
            "est.global_position_valid": "1" if estimators["global"] else "0",
            "est.velocity_valid": "1" if estimators["velocity"] else "0",
            "wind.speed_mps": f"{self.wind_speed_mps:.3f}",
            "wind.dir_deg": f"{self.wind_dir_deg % 360.0:.1f}",
            "wind.gust_mps": f"{self.wind_gust_mps:.3f}",
            "geofence.box": "" if self.geofence is None else self.geofence.describe(),
            "geofence.action": "" if self.geofence is None else self.geofence_action,
            "realism.telemetry_latency_ms": f"{self.telemetry.latency_s * 1000.0:.3f}",
            "realism.telemetry_jitter_ms": f"{self.telemetry.jitter_s * 1000.0:.3f}",
            "realism.sensor_noise": self.sensor_noise,
            "realism.battery_failsafe_pct": f"{self.battery_failsafe_pct:.3f}",
            "realism.battery_drain_pct_s": f"{self.battery_drain_pct_s:.4f}",
            "realism.seed": str(self.seed),
        }

    def describe(self) -> dict[str, Any]:
        """The configuration a report repeats, in the names the CLI uses."""
        return {
            "gps": self.gps_mode,
            "gps_noise_m": round(self._gps_sigma(), 4),
            "local_estimator": "on" if self.local_estimator else "off",
            "wind_mps": self.wind_mps,
            "wind_dir_deg": self.wind_dir_deg,
            "wind_gust_mps": self.wind_gust_mps,
            "telemetry_latency_ms": round(self.telemetry.latency_s * 1000.0, 3),
            "telemetry_jitter_ms": round(self.telemetry.jitter_s * 1000.0, 3),
            "sensor_noise": self.sensor_noise,
            "battery_failsafe_pct": self.battery_failsafe_pct,
            "battery_drain_pct_s": self.battery_drain_pct_s,
            "geofence": "" if self.geofence is None else self.geofence.describe(),
            "geofence_action": self.geofence_action,
            "realism_seed": self.seed,
        }

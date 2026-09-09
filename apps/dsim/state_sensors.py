"""The inexpensive state sensors: GNSS, IMU, barometer, compass, temperature.

These read the vehicle and the environment rather than the scene, so they need
no ray casting and no renderer. They are strictly observational in v1: they
publish what an instrument would have measured, and nothing here feeds back
into the physics or the flight estimator.

Two noise sources meet in this module and they are deliberately kept apart.
Slowly-wandering environmental state -- a GNSS fix drifting, a barometer
warming up -- lives in :mod:`dsim.realism`, evolves on every physics tick, and
is *read* here so that one wander reaches vehicle status and a sensor record as
the same number. Per-sample white noise is drawn from the logical capture's own
generator, so a published record depends on its capture index and not on how
many other samples happened to be drawn first.

Where a sensor has both a hardware figure and a realism figure for the same
quantity, they are independent sources and combine in quadrature.
"""

from __future__ import annotations

import math

import numpy as np

from dsim.realism import SENSOR_NOISE_PROFILES
from dsim.transforms import resolve

#: Standard gravity, m/s^2. Positive downward in the map frame.
GRAVITY_MPS2 = 9.80665

#: ISA troposphere constants for the pressure/altitude relationship.
_ISA_LAPSE_K_PER_M = 0.0065
_ISA_SEA_LEVEL_K = 288.15
_ISA_EXPONENT = 5.25588


def combined(*sigmas: float) -> float:
    """Independent noise sources add in quadrature, not by addition."""
    return math.sqrt(sum(float(s) ** 2 for s in sigmas))


def pressure_pa(altitude_amsl_m: float, sea_level_pa: float) -> float:
    """ISA troposphere pressure at an altitude above mean sea level."""
    ratio = 1.0 - _ISA_LAPSE_K_PER_M * altitude_amsl_m / _ISA_SEA_LEVEL_K
    return sea_level_pa * max(ratio, 1e-6) ** _ISA_EXPONENT


def gnss(model, vehicle, state, rng):
    """A GNSS fix, or an invalid record when the receiver has none.

    Installation is the profile's business -- a vehicle with no
    ``position.gnss`` sensor publishes nothing at all and reports no fix in
    vehicle status. Fix quality and the position error are the environment's,
    and both come from the same wander vehicle status publishes, so a client
    cannot get a better position by reading the sensor instead.

    Velocity is over the ground, which is what a receiver measures: the
    vehicle's velocity through the air plus the wind carrying it.
    """
    del model, rng
    realism = vehicle.realism
    fix = realism.gps_fix()
    valid = fix["fix_type"] > 0
    north_err, east_err, up_err = realism.gps_offset_m()
    wind_x, wind_y = realism.wind_vector()
    payload = dict(schema="gnss.sample.v1", fix_type=int(fix["fix_type"]),
                   satellites=int(fix["satellites"]),
                   hdop=round(float(fix["hdop"]), 3),
                   vdop=round(float(fix["vdop"]), 3), valid=valid)
    if not valid:
        payload.update(lat_deg=None, lon_deg=None, alt_m=None,
                       vel_north_mps=None, vel_east_mps=None, vel_down_mps=None,
                       error_north_m=None, error_east_m=None, error_up_m=None)
        return payload, False
    lat, lon, alt = vehicle.map_to_gps(state.x + east_err, state.y - north_err,
                                       state.z + up_err)
    payload.update(lat_deg=lat, lon_deg=lon, alt_m=alt,
                   error_north_m=north_err, error_east_m=east_err,
                   error_up_m=up_err,
                   # Map Y is south-positive and map Z is up, so north and down
                   # are the negatives of those axes.
                   vel_north_mps=-(state.vy + wind_y),
                   vel_east_mps=state.vx + wind_x,
                   vel_down_mps=-state.vz)
    return payload, True


def imu(model, profile_data, sensor_id, state, previous, dt, rng):
    """Angular rate and specific force in the sensor's own axes.

    **Axes** are the sensor's, which are the body's through its mount:
    +X forward, +Y right, +Z up.

    **Angular rate** is measured, not differentiated from Euler angles: the
    sensor's rotation matrix is resolved at both snapshots and the rotation
    between them is read off directly, so the answer does not depend on an
    Euler convention and stays correct under combined roll and pitch. A
    positive Z rate is a turn to the right.

    **Specific force** is proper acceleration -- what an accelerometer
    actually reads -- so at rest and level the Z axis reads ``+g``, not zero
    and not ``-g``. It is the world-frame acceleration minus gravity, rotated
    into the sensor's axes.

    The lever arm is not modelled: a sensor mounted away from the centre of
    rotation does not see the extra centripetal or tangential acceleration
    that offset would produce.
    """
    now = resolve(profile_data, sensor_id, state)[:3, :3]
    if previous is None or dt <= 0.0:
        rate = np.zeros(3)
        world_accel = np.zeros(3)
    else:
        before = resolve(profile_data, sensor_id, previous)[:3, :3]
        rate = _angular_rate(before, now, dt)
        world_accel = (np.array([state.vx, state.vy, state.vz])
                       - np.array([previous.vx, previous.vy, previous.vz])) / dt
    specific = now.T @ (world_accel + np.array([0.0, 0.0, GRAVITY_MPS2]))
    rate = np.degrees(rate) + model["gyro_bias_dps"] + rng.standard_normal(3) * model["gyro_noise_std_dps"]
    specific = specific + model["accel_bias_mps2"] + rng.standard_normal(3) * model["accel_noise_std_mps2"]
    return dict(schema="imu.sample.v1",
                angular_rate_dps=_xyz(rate),
                specific_force_mps2=_xyz(specific),
                gravity_mps2=GRAVITY_MPS2, valid=True), True


def _angular_rate(before, now, dt):
    """Body-axis angular velocity between two rotation matrices, in rad/s.

    The skew-symmetric part of ``before^T · now`` is the rotation between them
    to first order, and the axis it encodes has to be read with the public
    frame's own handedness -- which is not uniform. ``DV-SENSORS.md`` picks
    the roll and pitch matrices so that a positive roll drops the right wing
    and a positive pitch raises the nose, which makes those two rotations
    left-handed about +X and +Y, while positive yaw turning right is the
    ordinary right-handed rotation about +Z. So the X and Y components are the
    negative of the usual skew-to-axis reading and the Z component is not.
    Each sign is asserted by the tests rather than inferred here.
    """
    delta = before.T @ now
    skew = (delta - delta.T) / 2.0
    return np.array([skew[1, 2], skew[2, 0], skew[1, 0]]) / dt


def _xyz(vector):
    return dict(x=float(vector[0]), y=float(vector[1]), z=float(vector[2]))


def barometer(model, vehicle, state, rng):
    """Pressure altitude and the pressure it was derived from.

    ``altitude_m`` is height above the map's ground, the same datum every
    other altitude in this project uses, and carries the environment's slow
    drift plus this instrument's own white noise. ``pressure_pa`` is the ISA
    troposphere pressure at that altitude above mean sea level, using the map
    origin's elevation, so a client can invert one to the other.
    """
    profile = SENSOR_NOISE_PROFILES[vehicle.realism.sensor_noise]
    sigma = combined(model["noise_std_m"], profile["altitude_m"])
    altitude = state.z + vehicle.realism.baro_drift_m + rng.standard_normal() * sigma
    if model["quantization_m"] > 0.0:
        altitude = round(altitude / model["quantization_m"]) * model["quantization_m"]
    return dict(schema="barometer.sample.v1", altitude_m=altitude,
                pressure_pa=pressure_pa(vehicle.origin_alt_m + altitude,
                                        model["sea_level_pressure_pa"]),
                sea_level_pressure_pa=model["sea_level_pressure_pa"],
                valid=True), True


def magnetometer(model, vehicle, state, rng):
    """Measured compass heading in degrees, 0 north and increasing clockwise.

    Declination, inclination, hard and soft iron effects and the field vector
    itself are all deferred; what is published is the heading a corrected
    compass would report, with the environment's heading noise and this
    instrument's own combined.

    The renderer's private yaw is converted by the backend rather than here:
    this record and the ``body.heading_deg`` beside it in the same capture have
    to be the same number, and a provider with its own yaw convention gets to
    say what that number is exactly once.
    """
    profile = SENSOR_NOISE_PROFILES[vehicle.realism.sensor_noise]
    sigma = combined(model["noise_std_deg"], profile["heading_deg"])
    heading = vehicle.compass_heading(state.yaw_deg) + rng.standard_normal() * sigma
    return dict(schema="magnetometer.sample.v1",
                heading_deg=heading % 360.0, valid=True), True


def temperature(model, vehicle, state, rng):
    """Ambient air temperature at the vehicle, in Celsius.

    The ground-level value is a realism setting because it is a condition, and
    the sensor cools with height at the configured lapse rate. This is the
    cheapest sensor in the set and it is here to prove the shared compact ring
    carries a plain scalar without a channel or a status key of its own.
    """
    value = (vehicle.realism.ambient_temp_c - model["lapse_rate_c_per_m"] * state.z
             + rng.standard_normal() * model["noise_std_c"])
    if model["quantization_c"] > 0.0:
        value = round(value / model["quantization_c"]) * model["quantization_c"]
    return dict(schema="temperature.sample.v1", temperature_c=value,
                valid=True), True

"""Measurement models for the ray-based sensors, and their deterministic noise.

Truth comes from the shared ray service in :mod:`dsim.range`; everything here
turns truth into a reading. The sensors differ only in which rays they cast
and how a set of returns is reduced, so a fix to the noise, gating or
confidence rules applies to all of them at once.

Randomness is derived per logical capture rather than carried in a stream, so
a run that skips publishing a sample still produces the same numbers for every
sample it does publish.
"""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

from dsim.profiles import scan_angles_deg
from dsim.transforms import pinhole_rays, spherical_rays

#: Versioned seed material. Changing any part of the encoding, the digest or
#: the generator is a contract change and invalidates the frozen test vectors.
SEED_SCHEME = b'dvision2.sensor-noise.v1'

#: The golden angle, in radians. Successive cone samples are turned by it, so
#: any prefix of the pattern is still spread evenly around the beam axis.
_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))


def capture_rng(seed: int, sensor_id: str, reset_epoch: int, index: int) -> np.random.Generator:
    """The generator for one logical capture of one sensor.

    The material is unambiguous -- a fixed scheme tag, then the run seed, the
    NUL-terminated sensor id, the reset epoch and the logical capture index --
    so two sensors cannot collide and a replayed capture cannot drift. Python's
    process-randomised ``hash()`` is deliberately not involved.
    """
    material = (SEED_SCHEME + b'\0' + struct.pack('<q', int(seed))
                + sensor_id.encode('ascii') + b'\0'
                + struct.pack('<QQ', int(reset_epoch), int(index)))
    digest = hashlib.sha256(material).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype='<u4').tolist())


def cone_directions(model) -> np.ndarray:
    """Rays filling a beam cone, in the sensor's forward/right/up axes.

    Sample 0 is the beam axis. The rest spiral outwards on a sunflower
    pattern: polar angle ``half * sqrt(i / (n - 1))`` gives equal area per
    sample, and each is turned a further golden angle about the axis.
    """
    count = model['beam_samples']
    half = math.radians(model['beam_fov_deg']) / 2.0
    index = np.arange(count, dtype=np.float64)
    polar = half * np.sqrt(index / max(1.0, count - 1.0))
    around = index * _GOLDEN_ANGLE
    return np.stack((np.cos(polar), np.sin(polar) * np.cos(around),
                     np.sin(polar) * np.sin(around)), axis=-1)


def scan_directions(model) -> np.ndarray:
    """Rays of one 2D scan, in ray order, in the sensor's own axes."""
    angle_min, increment = scan_angles_deg(model)
    return spherical_rays(angle_min + increment * np.arange(model['samples']),
                          model['elevation_deg'])


def image_directions(model) -> np.ndarray:
    """Rays of one range image, row-major, in the sensor's own axes."""
    return pinhole_rays(model)


def directions(kind: str, model) -> np.ndarray:
    if kind == 'lidar.scan2d': return scan_directions(model)
    if kind == 'lidar.range_image': return image_directions(model)
    return cone_directions(model)


def measure(truth: np.ndarray, model, rng) -> tuple[np.ndarray, np.ndarray]:
    """Apply the sensor error model to true ranges.

    Returns metres as float32 with ``NaN`` for no valid return, and confidence
    as uint8 from 0 (nothing) to 255. Draws are taken for every ray in a fixed
    order -- one normal per ray, then one uniform per ray -- whether or not the
    ray returned anything, so a miss cannot shift a later ray's error.

    ``limit_degradation`` scales both the noise and the dropout probability
    with the fraction of full scale being measured, which is how the infrared
    and ultrasonic presets get worse as they approach their limits.
    """
    truth = np.asarray(truth, np.float64)
    normal = rng.standard_normal(truth.shape)
    uniform = rng.random(truth.shape)
    returned = np.isfinite(truth)
    fraction = np.where(returned, np.clip(truth / model['max_range_m'], 0.0, 1.0), 0.0)
    scale = 1.0 + model['limit_degradation'] * fraction
    ranges = np.where(returned, truth, np.nan) + normal * model['noise_std_m'] * scale
    if model['quantization_m'] > 0:
        ranges = np.round(ranges / model['quantization_m']) * model['quantization_m']
    kept = (returned & (uniform >= np.clip(model['dropout_probability'] * scale, 0.0, 1.0))
            & (ranges >= model['min_range_m']) & (ranges <= model['max_range_m']))
    ranges = np.where(kept, ranges, np.nan)
    if model['confidence_model'] == 'range_linear':
        level = 255.0 * (1.0 - np.clip(np.nan_to_num(ranges) / model['max_range_m'], 0.0, 1.0))
    else:
        level = np.full(truth.shape, 255.0)
    return ranges.astype(np.float32), np.where(kept, level, 0.0).astype(np.uint8)


def reduce_beam(ranges: np.ndarray, confidence: np.ndarray, reducer: str):
    """Collapse a cone of returns into the one reading the sensor reports.

    ``nearest`` is the default and the physical answer for a time-of-flight
    beam: the first echo wins. No valid return is not the same as maximum
    range, and is reported as an invalid reading rather than a distance.
    """
    valid = np.isfinite(ranges)
    if not valid.any():
        return None, 0.0, 0
    kept = ranges[valid]
    if reducer == 'nearest': pick = int(np.argmin(kept))
    elif reducer == 'farthest': pick = int(np.argmax(kept))
    else: pick = int(np.argsort(kept, kind='stable')[(len(kept) - 1) // 2])
    order = np.flatnonzero(valid)[pick]
    return float(ranges[order]), float(confidence[order]) / 255.0, int(valid.sum())


def pack_array(ranges: np.ndarray, confidence: np.ndarray) -> bytes:
    """The published array payload: float32 metres, then uint8 confidence.

    Both are little-endian and row-major, and ``NaN`` with zero confidence is
    the invalid sample. JSON's ban on NaN does not apply here: this is a packed
    binary array whose dtype and shape the manifest declares.
    """
    return (np.ascontiguousarray(ranges, '<f4').tobytes()
            + np.ascontiguousarray(confidence, '|u1').tobytes())


def array_layout(kind: str, model) -> list[dict]:
    """The manifest's description of a packed array payload."""
    shape = ([model['samples']] if kind == 'lidar.scan2d'
             else [model['height_px'], model['width_px']])
    return [dict(name='range_m', dtype='<f4', shape=shape, invalid='NaN'),
            dict(name='confidence', dtype='|u1', shape=shape, invalid='0')]


def calibration(kind: str, model) -> dict:
    """The angular calibration a consumer needs to turn an array into points."""
    if kind == 'lidar.scan2d':
        angle_min, increment = scan_angles_deg(model)
        return dict(angle_min_deg=angle_min, angle_increment_deg=increment,
                    samples=model['samples'], elevation_deg=model['elevation_deg'])
    return {k: model[k] for k in ('width_px', 'height_px', 'fx_px', 'fy_px', 'cx_px', 'cy_px')}

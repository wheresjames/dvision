# Mount and rig transform nodes

`mount.fixed`, `rig.fixed` and `mount.ptz` are not sensors. They are named
nodes in the profile's parent-linked transform graph: they carry a pose and,
for a PTZ, a settable state, and every sensor beneath one inherits their
motion. They produce no samples and own no channels.

## Profile fields

Mounts live in the profile's flat `mounts` array. `id`, `parent` and
`pose_parent` follow the same rules as a sensor
([`camera-rgb.md`](camera-rgb.md) documents them), and the same graph rules
apply: unique ids across mounts and sensors, an existing parent, no cycles,
exactly one path to `body`, and a sensor may never be another component's
parent.

| Field | Default | Validation |
|---|---|---|
| `type` | required | `mount.fixed`, `rig.fixed`, or `mount.ptz` |
| `state` | zeros | `mount.ptz` only: `pan_deg`, `tilt_deg`, `roll_deg` |
| `limits` | `[-180, 180]` each | `mount.ptz` only: two ordered bounds per axis; the initial `state` must lie inside them |

`mount.fixed` and `rig.fixed` are the same node with different names: use
`rig.fixed` for a structure carrying several sensors, `mount.fixed` for a
single bracket. Neither accepts `state` or `limits`.

The Sensors tab edits each PTZ axis state and its comma-separated minimum and
maximum limits. Invalid state/limit combinations block Apply and leave the active
profile unchanged. Components can be reparented using the parent field or by
dragging onto body or a mount; descendants are excluded as destinations.

## Composition

A component's world transform is its parent's composed with its own:

```text
T_world_child = T_world_parent · T_parent_child
R = Rz(yaw) · Ry(pitch) · Rx(roll)
```

A PTZ inserts its dynamic state between its fixed placement and its children:

```text
T_parent_child = T_parent_mount · Rz(pan) · Ry(tilt) · Rx(roll)
```

so the mechanism's mounting bracket does not move when the head pans, and a
stereo pair beneath it shares the platform motion while keeping its own
left/right calibration error. Angles are stored as Euler triples for
readability and converted to matrices at load; nothing at runtime works in
Euler angles.

Example, matching `assets/drone_profiles/stereo-nav-and-proximity.json`:

```json
{
  "id": "nav_ptz", "type": "mount.ptz", "parent": "body",
  "pose_parent": {"x_m": 0.08, "y_m": 0.0, "z_m": 0.10,
                   "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0},
  "state": {"pan_deg": 0.0, "tilt_deg": -5.0, "roll_deg": 0.0},
  "limits": {"pan_deg": [-170.0, 170.0], "tilt_deg": [-90.0, 30.0]}
}
```

## Limitations

- PTZ state is profile configuration applied while disarmed. There is no
  command to move a mount at run time; when one arrives it must go through the
  pymembus control facilities, not a new transport.
- The state is a pose, not a mechanism: no slew rate, backlash, or limit
  behaviour is modelled, so a mount reaches its commanded angle instantly.
- Single parenting only. A shared or cyclic node is a validation error.

## Correctness tests

`tests/test_sensor_contract.py` covers nested `body → PTZ → rig → camera`
composition in the documented order, translation rotating with body heading,
PTZ state outside its limits, missing parents, and cycles.

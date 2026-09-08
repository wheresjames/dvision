# dctl device intake

The **Devices** tab browses the enabled devices in the vehicle's published
`sensors.manifest`. Drag a device row from the tree onto a grid cell to open
it in exactly that cell; the check mark is an indicator, not a control.
Mounts follow the same component tree as the simulator's editor; manifests
without an embedded profile fall back to a flat list. A pane leaves through
its **×** button, which unchecks the device. A plain click opens nothing —
except on an absent device's grey row, which forgets its reservation so
the cell is not held for a device that is gone. An initial selection from
`--devices` takes the near-square automatic layout.

```sh
python apps/dctl/dctl.py --id area1 --devices nav_left,scan,flash,imu
python apps/dctl/dctl.py --id area1 --camera nav_right --sensor-cache-mb 32
python apps/dctl/dctl.py --id area1 --layout inspection
python apps/dctl/dctl.py --id area1 --no-sensors
```

`--camera` chooses Flight video (the manifest's primary by default). The
Flight source dropdown can switch to any published video device, plus one
`stereo:<group>` entry per two-camera sync group. The primary
camera stays a required input; selecting another camera does not release it. `--devices` is a comma-separated initial selection. `--no-sensors`
disables sensor discovery, video and the Devices tab; controls and telemetry
still work. `--width` and `--height` continue to cap Flight video dimensions.
Camera-free manifests still support the device browser and controls.

## Arrange and remember the bench

Drag a device row from the tree onto a cell to open it there; a pane already
standing in that cell is closed and replaced, so a drop never rearranges the
grid behind your back — structural edits stay in the operator's hands. Drag a pane title
onto another pane to swap their occupied rectangles. Drag its bottom-right
resize handle to extend or shrink its row and column span, or right-click it
for **Span right**, **Span down** and **Shrink to one cell**. A pane that
would collide with an expanded pane moves to the first free cell; the grid
adds a row if needed. Drag the narrow sashes between rows or columns to
adjust their proportions. These operations keep panes tiled and on screen.

Tick **Arrange** to see and edit the grid itself. Panes collapse to their
title strips, painting pauses while intake keeps draining exactly as under
freeze, the sashes fatten, and every empty cell outlines itself as a drop
target. **+ Row** and **+ Col** append an empty row or column at the edge;
**− Row** and **− Col** remove the edge row or column, and are disabled
unless it is empty — a structural edit never moves a pane, so drag the pane
out first. Untick **Arrange** to restore the data at its current captures.

To build a bench around one large video pane: tick **Arrange**, add rows and
columns with the + buttons, drag the camera from the tree into the centre
cell, span it with the menu commands or the resize handle, drop the sensors
into the surrounding cells, then drag the sashes until the video dominates.

Layouts save on close and when switching profiles, in
`~/.config/dvision2/device_layouts.json`, through the same locked state store
used for window geometry. The default key is the profile **name**, with an
instance-id fallback. `--layout NAME` chooses an explicit shared layout name.
Changing a profile digest keeps the arrangement; the footer reports a changed
revision. Missing devices retain their cells and options and appear grey in the
tree, then return to their saved cells if restored. New devices stay unchecked.
`--devices` overrides the initial selection when explicitly provided.

The saved record includes row/column counts, spans, row/column weights and each
renderer’s options. Invalid spans and overlapping cells are repaired individually
without discarding the rest of the record. Concurrent windows preserve other
layout keys; the last writer wins for the same key.

## Device views

- Cameras use the same `ImageRenderer` in Flight and Devices. Fit can contain,
  cover, or use native pixels; images resize smoothly. Camera panes report
  resolution, calibrated fields of view and capture pose.
- Single-beam range sensors show metres, a gauge and a selectable 10/30/60-second
  strip chart. Invalid samples say **no return** and disable the gauge.
- Barometer and temperature show values with units and strip charts. The
  magnetometer shows its heading above a full compass rose and has no strip
  chart or window option: a heading wraps around, so its line plot is noise.
- Scanning LiDAR shows calibrated rays and range rings, coloured by confidence.
  Choose forward-up or north-up and automatic or manually entered range scale.
  The readout gives valid returns and the nearest sensor-relative bearing.
- Range images use a fixed viridis ramp over the sensor’s configured metre
  range. Switch to confidence for the diverging confidence ramp. Invalid cells
  are grey; nearest-neighbour resizing preserves individual measurements.
  Hover reports the original cell’s range to three decimal places, including
  in confidence mode.

Renderer settings survive hiding a tab, profile Apply and restarting the window.

- IMU shows angular rate and specific force as three-axis envelope strip charts
  sharing a time axis; the 10/30/60-second window and the envelope compression
  keep a 100 Hz sensor affordable at the 4 Hz numeric paint rate.
- GNSS shows the fix state as one of three distinct failures — **no receiver**,
  **no fix**, **fix rejected** — rather than blank fields, with quality
  (satellites, HDOP/VDOP), position, NED velocities, and a recent
  north/east error scatter whose shape makes drift visible.

A pane's header shows its id, type and configured rate, followed by three
small discs. They are fixed-width and packed before the title, so a pane
squeezed narrow clips its own name rather than losing its controls: **i**
swaps the graphic for the pane's text readout and back, **❄** freezes that
pane alone while intake continues, and **×** closes it.

The readout replaces the graphic rather than sitting under it — a two-line
strip beneath a live picture was too cramped to read, and it took height from
the graphic to be so. It carries a health dot, achieved rate, simulated sample
time, capture id, sequence attachment baseline, gaps, association drops and
late drops, plus the renderer's own rows, and it wraps to the cell it owns.
The choice persists with the pane.

While the text body is showing, the pane draws no graphic: every renderer
sizes itself from its widget, so drawing into an unpacked one would shrink
every plot to its minimum and leave that behind. The pane therefore drops out
of the video budget entirely, and **Snapshot PNG** is refused with `the pane
is showing its readout, not a graphic` rather than rasterizing a canvas that
was drawn at some other size. Swapping back redraws immediately, frozen panes
included.

## Stereo pairs

A `sync_group` of exactly two cameras appears both as a `stereo:<group>` pane
in the tree and as a Flight video source. Groups with any other membership
stay browsable as individual cameras. A pair composes only frames that share
a `capture_id` (and generation and epochs); a pair that is not synchronized
says so in its readout instead of blending neighbouring frames. The readout
also gives the baseline from the two sensors' resolved poses.

Modes are **side-by-side**, **anaglyph** (red channel from the left eye) and
**difference**. Cameras that differ in resolution render both eyes at natural
scale side by side with the mismatch stated in the readout — a stereo view
that quietly resamples one eye is worse than no stereo view. Flying a stereo
source opens both member cameras; the primary camera stays the required input.

## Freeze and synchronized captures

**Freeze all** holds every visible pane at one shared `capture_id` — the
newest capture for which every visible device has a record — so all panes
show the same simulated instant. A pane with no record at that capture says
**no sample at capture N** rather than showing a neighbouring one. When no
capture is shared by every visible device, the footer says so and freeze
holds the newest capture that exists. Untick **Sync captures** to freeze each
pane at its own newest record instead. Intake keeps draining while frozen;
the held capture is invalidated by a reset or profile change and the panes
say so. A single pane's freeze toggle holds that pane alone.

## Pop-out windows

Right-click a pane for **Pop out / Return to grid** and its export commands;
the toolbar buttons export the selected tree row's open pane. A popped-out
pane keeps its stream, renderer and displayed
state, floats as its own window positioned by the window manager, and returns
to its saved cell when closed. Its geometry persists through the same store
as window positions, so it reopens where it was left. A popped-out pane
carries none of the flying window's key bindings: flight keys cannot reach
it, and key events delivered inside it never send a command.

## Snapshot and sample export

**Snapshot PNG** rasterizes the pane's displayed canvas with its readout, and
**Dump JSON/CSV** writes the pane's retained samples, newest first under
8 MiB and 1000 samples; the footer reports what was omitted. Both write into
`sim.report_dir/dctl` — the run root the simulator publishes — and the
controls are disabled while that key is absent rather than inventing a path.
Snapshots embed the `capture_id` and sensor id in the PNG metadata. A dumped
file reloads losslessly: the sample schema is `dvision2.device-samples.v1`,
with array fields and images base64-encoded with dtype and shape.
The close button unchecks the device. Device rates are records per simulated
second; the footer's paint rate is per wall second. The footer also shows
cache usage and ceiling, cache evictions, ring overruns, sequence gaps and
paints skipped when a round exhausts its 8 ms budget.

Profile Apply closes the old generation's channels and reattaches by id.
Removed panes disappear; the selection is retained so they return if their
ids reappear. Recreated panes show “profile changed”. Reset invalidates
unmatched pairs and counts their discarded records. Browsing devices never
makes them required Flight inputs.

## Intake and accounting

`dcmn.sensors.SensorSession` owns one compact ring per vehicle and one bulk
channel per opened device. `open(id, required=False, accounting=...)` returns
a refcounted `DeviceStream`; matching `release(id)` calls close it on the last
reference. Every compact record is decoded once and dispatched by sensor id.
`poll()` runs once on the control window's tick, before painting, and does
no image copying or array unpacking. Call a stream's `refresh()` on its paint
cadence to read bulk data and match it with already-drained metadata.

- `from_generation` is the API default and counts sequence gaps from 1,
  including records published before attachment. Required Flight input uses it.
- `from_attach` starts at the first observed record and reports “attached at
  sequence N”. Optional browser panes explicitly choose it. Opening a device
  ten minutes into a flight therefore does not count ten minutes of loss.

Array and metadata sequence gaps are tracked separately and summed. Delayed
pairs match by `array_sequence`, cameras by `video_sequence`; both invalidate
pending data on reset and generation changes. Late pairs never replace a newer
sample. `SensorVideo` remains the compatibility adapter for existing algorithm
consumers, including calibration and `capture_status()`.

Compact histories hold up to 60 simulated seconds, bounded by configured
rate. A bulk stream retains its latest matched sample and bounded unmatched
pairs. All retained records, images and fields participate in global oldest-first
cache eviction. The default ceiling is 64 MiB of encoded record and array/image
bytes, excluding Python object overhead and Tk's displayed images. An individual
sample larger than the ceiling is evicted too; it is never exempted silently.

Hidden Devices panes release optional subscriptions and reopen with a fresh
attachment baseline when visible. The unchecked/hidden interval is deliberately
not subscribed, rather than counted as transport loss. The compact ring still
drains and its per-device health continues to update. Flight's required stream
remains open. Freeze does not release the subscription.

## Renderer contract

`dcmn.device_view.RENDERERS` is an explicit dictionary mapping type strings to
renderer classes. Resolution tries an exact type, then the longest registered
prefix ending in `.`, then `GenericRenderer`. A renderer implements:

```python
class Renderer:
    def __init__(self, parent, entry): ...
    def draw(self, sample): ...
    def describe(self, sample): ...  # list of (label, value) readout rows
    def destroy(self): ...
```

`Sample` supplies `sensor_id`, `type`, `capture_id`, `sequence`, `sim_time_s`,
`status`, decoded `payload`, optional named NumPy `fields`, optional RGB
`image`, and the manifest `entry`. Renderers do not open transports or import
the controller. `DevicePane` owns the common title, health/readout, freeze and
close controls. `DeviceGrid` owns visibility, subscription references, auto
layout, drag controls, persisted options and painting budgets.

Unknown types use the generic renderer: formatted JSON, RGB images and
layout-driven heat maps. Every array field is shown in manifest order, with
its name, dtype and shape in the readout. Higher dimensions flatten into rows.
Fields scale independently from their finite minimum to maximum (blue through
green to red), constant values use the midpoint, and nonfinite cells are black.
Nearest-neighbour resizing preserves measured cells. No matplotlib dependency
or background drain thread is introduced.

Numeric panes paint at most 4 Hz. Bulk panes share the 30 Hz image allowance,
with a 5 Hz floor per pane. Painting stops starting additional panes after the
round budget is exhausted; earlier selections have priority. Control and lease
maintenance keep their normal tick cadence.

## Speed ceiling

The reference profile publishes 306 compact records per simulated second and
its default compact ring holds 308 slots: approximately one simulated second.
At a 30 Hz wall-clock poll rate, it covers about 30 polls at real time, three
at 10×, and wraps between polls past roughly **30× real time**. Scheduling
stalls or expensive paints can lower that ceiling. Overruns and gaps report
the loss. Raising the profile's `transport.retention_s` buys headroom at the
cost of publisher shared memory; increasing the viewer cache cannot recover
records already overwritten in the transport.

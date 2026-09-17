"""What an operator sees of the detectors: the Live overlay and the Detections tab.

Two halves. The pure half builds pictures from the detector host
(:class:`dalg.detect.host.DetectorSources`) with no Tk: map shapes for projected
detections, estimates and the camera footprint (:func:`detection_shapes`), the
report's ``detections.png`` (:func:`detections_image`), and the frame a
detector actually processed with its detections, rejected candidates and a
corner stamp (:func:`frame_overlay`). The map shapes are ordinary
:class:`dcmn.mapview.Shape` primitives, so the Tk pane and the PIL report draw
the same thing.

The Tk half is :class:`DetectionsTab`: a map, an estimates table (selecting a
row highlights that estimate), per-detector health rows and a sim-time strip
per detector -- ticks for frames with detections, gaps where frames were
skipped or failed. ``dalg.py`` only wires these in.
"""
from __future__ import annotations

from dcmn import theme
from dcmn.mapview import Shape

#: Projected detections older than this (sim seconds) are not drawn.
FADE_S = 30.0
#: Rows of the camera footprint sampled from the bottom of the frame up.
FOOTPRINT_ROWS = 16
STRIP_COLOURS = {'detect': theme.OK, 'empty': theme.DIM, 'skip': theme.CAUTION, 'fail': theme.DANGER}


def _fade(age_s: float) -> str:
    if age_s < 2.0: return theme.HIGHLIGHT
    if age_s < 10.0: return theme.WARN
    return theme.DIM


def footprint(state) -> list[tuple[float, float]]:
    """The ground a detector's last frame could see, as a polygon (empty when unknown)."""
    from dalg.detect.projection import project
    last = getattr(state, 'last', None)
    if not last or last.get('camera') is None: return []
    intrinsics, camera, settings = state.intrinsics, last['camera'], state.settings
    edges = {0.0: [], float(intrinsics.width_px): []}
    for column in edges:
        for index in range(FOOTPRINT_ROWS + 1):
            row = intrinsics.height_px * (1.0 - index / FOOTPRINT_ROWS)
            projection = project(column, row, 0.0, camera, intrinsics, max_range_m=settings.max_range_m)
            if not projection.projectable: break
            edges[column].append(projection.xy_m)
    left, right = edges[0.0], edges[float(intrinsics.width_px)]
    if len(left) < 2 or len(right) < 2: return []
    return [left[0], *left[1:], *reversed(right[1:]), right[0]]


def detection_shapes(host, now_s: float, *, selected: str | None = None) -> list[Shape]:
    """Everything the Detections map draws, in map metres."""
    shapes: list[Shape] = []
    if host is None: return shapes
    for state in host.states.values():
        ring = footprint(state)
        if len(ring) >= 3:
            shapes.append(Shape.polygon(ring, outline=theme.GRID, width=1.))
    for detection in host.recent:
        age = now_s - detection['time_s']
        if age > FADE_S: continue
        x, y = detection['xy_m']
        r = 0.15
        shapes.append(Shape('oval', x - r, y - r, x + r, y + r, fill=_fade(age), outline=_fade(age), width=1.))
    for estimate in host.estimates():
        x, y = estimate['xy_m']
        radius = max(estimate['spread_m'], 0.3)
        colour = theme.DIM if estimate['stale'] else theme.TARGET_EDGE
        fill = colour if estimate['confirmed'] and not estimate['stale'] else None
        if estimate['stale']: fill = None
        width = 3. if estimate['id'] == selected else 2.
        outline = theme.HIGHLIGHT if estimate['id'] == selected else colour
        shapes.append(Shape('oval', x - radius, y - radius, x + radius, y + radius, fill=fill, outline=outline,
                            width=width))
        shapes.append(Shape.label(x, y - radius - 0.6, f"{estimate['id']} x{estimate['count']}", outline=outline))
    return shapes


def _extent(host, grid):
    """Bounds for a report image with no evidence: the estimates, detections and footprints, padded."""
    points = [tuple(e['xy_m']) for e in host.estimates()] + [tuple(d['xy_m']) for d in host.recent]
    for state in host.states.values(): points.extend(footprint(state))
    if not points: return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    pad = 5.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def detections_image(host, *, grid=None, background=None, now_s: float = 0.0):
    """The Detections tab's map as a PIL image, for the report."""
    from dcmn.map_pane import Overlay, fallback_grid, snapshot_image
    if grid is None:
        bounds = _extent(host, grid)
        if bounds is None: return None
        grid = fallback_grid(bounds)
    return snapshot_image(grid, overlay=Overlay(shapes=tuple(detection_shapes(host, now_s))), sim_now_s=now_s,
                          background=background)


def frame_overlay(state, now_s: float):
    """The frame a detector last processed, drawn over: ``(PIL image, candidate reason lines)`` or ``(None, [])``."""
    from PIL import Image, ImageDraw
    last = getattr(state, 'last', None)
    if not last or last.get('image') is None: return None, []
    image = Image.fromarray(last['image']).convert('RGB')
    draw = ImageDraw.Draw(image)
    for pixel, row in zip(last['result'].detections, last['detections']):
        r = max(pixel.extent_px, 3.0)
        if pixel.box_px is not None:
            draw.rectangle(pixel.box_px, outline=theme.OK, width=1)
        draw.ellipse((pixel.cx_px - r, pixel.cy_px - r, pixel.cx_px + r, pixel.cy_px + r), outline=theme.HIGHLIGHT)
        tail = '' if row.get('projectable') else ' (not projectable)'
        draw.text((pixel.cx_px + r + 2, pixel.cy_px - r), f'{pixel.label} {pixel.confidence:.2f}{tail}',
                  fill=theme.HIGHLIGHT)
    reasons = []
    for index, candidate in enumerate(last['candidates'], 1):
        x, y = candidate.cx_px, candidate.cy_px
        draw.line((x - 3, y - 3, x + 3, y + 3), fill=theme.DIM)
        draw.line((x - 3, y + 3, x + 3, y - 3), fill=theme.DIM)
        draw.text((x + 4, y), str(index), fill=theme.DIM)
        reasons.append(f'{index}. {candidate.reason}')
    stamp = (f"t={last['capture_time_s']:.2f}s age={max(0.0, now_s - last['capture_time_s']):.2f}s "
             f"detect={last['latency_ms']:.1f}ms")
    draw.rectangle((0, 0, 7 * len(stamp) // 1.3, 12), fill=theme.CANVAS)
    draw.text((2, 1), stamp, fill=theme.TEXT)
    return image, reasons


def overlay_on_frame(image, states, sensor: str):
    """Every detector's last detections for ``sensor`` drawn over a camera image (the Live toggle)."""
    from PIL import ImageDraw
    if image is None: return image
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)
    for state in states:
        if state.config.sensor != sensor or not state.last: continue
        for pixel in state.last['result'].detections:
            r = max(pixel.extent_px, 3.0)
            draw.ellipse((pixel.cx_px - r, pixel.cy_px - r, pixel.cx_px + r, pixel.cy_px + r), outline=theme.HIGHLIGHT)
            draw.text((pixel.cx_px + r + 2, pixel.cy_px - r), f'{state.config.id}: {pixel.label}',
                      fill=theme.HIGHLIGHT)
    return image


def health_rows(host) -> list[tuple[str, ...]]:
    """One table row per detector."""
    rows = []
    for state in ([] if host is None else host.states.values()):
        h = state.health()
        latency = ('--' if h['latency_ms_p50'] is None else f"{h['latency_ms_p50']:.1f}/{h['latency_ms_p95']:.1f}")
        hz = '--' if h['hz'] is None else f"{h['hz']:.1f}"
        not_projectable = ', '.join(f'{k} {v}' for k, v in sorted(h['not_projectable'].items())) or '--'
        rows.append((h['id'], h['state'], f"{h['processed']}/{h['skipped']}/{h['failed']}",
                     f"{hz} / {h['configured_hz']:g}", latency, f"{h['detections_per_frame']:.2f}",
                     not_projectable, h['last_error'] or h['reason'] or '--'))
    return rows


def estimate_rows(host, now_s: float) -> list[tuple[str, ...]]:
    return [(e['id'], e['label'], f"{e['xy_m'][0]:.2f}, {e['xy_m'][1]:.2f}", f"{e['spread_m']:.2f}", str(e['count']),
             'yes' if e['confirmed'] else 'no', f"{max(0.0, now_s - e['last_s']):.1f}" + (' (stale)' if e['stale'] else ''))
            for e in ([] if host is None else host.estimates())]


def status_text(host, error: str = '') -> str:
    """``detectors=<n> estimates=<confirmed>/<total>``, naming a failed detector."""
    if host is None: return f'detectors=0 ({error})' if error else ''
    estimates = host.estimates()
    text = (f'detectors={len(host.states)} estimates={sum(1 for e in estimates if e["confirmed"])}'
            f'/{len(estimates)}')
    failed = [s.config.id for s in host.states.values() if s.state == 'failed']
    if failed: text += f' FAILED: {", ".join(failed)}'
    return text


class DetectionsTab:
    """The Detections tab: map, estimates, health and a sim-time strip per detector."""

    HEALTH = (('detector', 170), ('state', 70), ('proc/skip/fail', 100), ('Hz eff/cfg', 90),
              ('latency p50/p95 ms', 120), ('det/frame', 70), ('not projectable', 160), ('last error', 220))
    ESTIMATES = (('id', 150), ('label', 100), ('xy m', 120), ('spread m', 70), ('count', 60),
                 ('confirmed', 80), ('last seen s ago', 110))

    def __init__(self, notebook, run, tk, ttk, *, reference=None) -> None:
        from dcmn.map_pane import MapPane
        self.run, self.tk, self.ttk = run, tk, ttk
        self.page = ttk.Frame(notebook)
        notebook.add(self.page, text='Detections')
        self.reference = reference
        self.selected: str | None = None
        self.pane = MapPane(self.page, title='detections', width=420, height=360)
        self.pane.widget.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=(0, 6))
        self.estimates = self._table(self.page, self.ESTIMATES, 8)
        self.estimates.master.grid(row=0, column=1, sticky='nsew')
        self.estimates.bind('<<TreeviewSelect>>', self._select)
        self.health = self._table(self.page, self.HEALTH, 4)
        self.health.master.grid(row=1, column=1, sticky='nsew', pady=(6, 0))
        self.strip = tk.Canvas(self.page, height=70, bg=theme.CANVAS, highlightthickness=0)
        self.strip.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(6, 0))
        self.notice = tk.StringVar(value='')
        ttk.Label(self.page, textvariable=self.notice, style='Dim.TLabel').grid(row=3, column=0, columnspan=2,
                                                                               sticky='w')
        self.page.columnconfigure(0, weight=1); self.page.columnconfigure(1, weight=1)
        self.page.rowconfigure(0, weight=1); self.page.rowconfigure(1, weight=1)

    def _table(self, parent, columns, height):
        frame = self.ttk.Frame(parent)
        table = self.ttk.Treeview(frame, columns=[c for c, _ in columns], show='headings', height=height)
        for name, width in columns:
            table.heading(name, text=name)
            table.column(name, width=width, stretch=True, anchor='w')
        table.pack(fill='both', expand=True)
        return table

    def _select(self, _event=None) -> None:
        selection = self.estimates.selection()
        self.selected = selection[0] if selection else None
        if self.selected:
            self.notice.set(f'{self.selected} highlighted on the map')

    def _fill(self, table, rows) -> None:
        wanted = [str(row[0]) for row in rows]
        for item in table.get_children():
            if item not in wanted: table.delete(item)
        for index, row in enumerate(rows):
            item = str(row[0])
            if table.exists(item): table.item(item, values=row)
            else: table.insert('', index, iid=item, values=row)

    def paint(self) -> None:
        run = self.run
        host, now = run.detectors, run.sim_time_s()
        grids = run.evidence_grids()
        grid = grids[sorted(grids)[0]] if grids else None
        self.pane.set_grid(grid, sim_now_s=now)
        if grid is None and host is not None:
            bounds = _extent(host, None)
            self.pane.set_fallback_extent(bounds, 'detections' if bounds else '')
        else:
            self.pane.set_fallback_extent(None)
        if self.reference is not None:
            self.pane.set_background(self.reference.background(), refresh=False)
        self.pane.set_overlay(shapes=tuple(detection_shapes(host, now, selected=self.selected)))
        self.pane.set_header(status_text(host, run.detector_error) or 'no detectors in this profile')
        self.pane.refresh()
        self._fill(self.estimates, estimate_rows(host, now))
        self._fill(self.health, health_rows(host))
        self._paint_strip(host, now)

    def _paint_strip(self, host, now) -> None:
        from dalg.detect.host import STRIP_S
        canvas = self.strip
        canvas.delete('all')
        if host is None: return
        width = max(1, canvas.winfo_width())
        states = list(host.states.values())
        lane = max(12, 70 // max(len(states), 1))
        start = now - STRIP_S
        for index, state in enumerate(states):
            y0 = index * lane
            canvas.create_text(4, y0 + lane / 2, text=state.config.id, anchor='w', fill=theme.DIM,
                               font=('TkDefaultFont', 8))
            for time_s, kind in state.strip:
                if time_s < start: continue
                x = 150 + (width - 154) * (time_s - start) / STRIP_S
                top = y0 + 2 if kind != 'detect' else y0 + 1
                canvas.create_line(x, top, x, y0 + lane - 2, fill=STRIP_COLOURS.get(kind, theme.DIM))


def reasons_text(reasons: list[str]) -> str:
    return '\n'.join(reasons) if reasons else 'no rejected candidates'


def selector_values(run) -> list[str]:
    """Evidence sources and detectors, for the Live selector."""
    sources = [] if run.sources is None else list(run.sources.states)
    detectors = [] if run.detectors is None else list(run.detectors.states)
    return sources + detectors

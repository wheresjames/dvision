"""Manifest-driven, read-only device panes. Renderers receive only Sample values."""
import json
import math
import time
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageTk

from dcmn.pacing import TEXT_HZ, VIDEO_HZ
from dcmn import theme, layout
from dcmn.window import load_state, save_state, save_window_geometry, restore_window_geometry
from dcmn.series import EnvelopeSeries
from dcmn.sensors import Sample
from dcmn.device_export import export_directory, dump_samples, snapshot_png

DOTS = {'ok': '●', 'warn': '◐', 'bad': '✕', 'unknown': '○'}


def heatmap(fields, layout):
    """Render every manifest field without assuming its name, dtype or shape.

    Higher-dimensional fields are flattened into rows; nonfinite values are
    black. Nearest-neighbour resizing in the widget preserves measured cells.
    """
    images = []
    for field in layout:
        data = np.asarray(fields[field['name']], dtype=float)
        if not data.size: continue
        data = data.reshape((-1, data.shape[-1] if data.ndim else 1))
        valid = np.isfinite(data)
        normalized = np.zeros(data.shape)
        if valid.any():
            low, high = data[valid].min(), data[valid].max()
            normalized[valid] = (data[valid]-low) / (high-low) if high > low else .5
        rgb = np.stack((normalized, 1-abs(2*normalized-1), 1-normalized), axis=-1)
        rgb[~valid] = 0
        images.append(Image.fromarray((rgb*255).astype(np.uint8)))
    if not images: return None
    width = max(im.width for im in images)
    result = Image.new('RGB', (width, sum(max(16, im.height) for im in images)))
    y = 0
    for im in images:
        height = max(16, im.height)
        result.paste(im.resize((width, height), Image.Resampling.NEAREST), (0, y))
        y += height
    return result


class Renderer:
    DEFAULTS = {}
    def __init__(self, parent, entry):
        self.entry = entry
        self.options = dict(self.DEFAULTS); self.variables = {}; self.choices = {}; self.history = []
        self.widget = ttk.Frame(parent)
        self.widget.pack(fill='both', expand=True)

    def option(self, key, choices, editable=False):
        self.choices[key] = choices
        row = ttk.Frame(self.widget); row.pack(fill='x')
        ttk.Label(row, text=key.replace('_', ' ')).pack(side='left')
        variable = tk.StringVar(value=str(self.options[key])); self.variables[key] = variable
        ttk.Combobox(row, textvariable=variable, values=choices, state='normal' if editable else 'readonly', width=12).pack(side='right')
        def changed(*args):
            value = variable.get()
            if self.valid_option(key, value): self.options[key] = value
        variable.trace_add('write', changed)

    def valid_option(self, key, value):
        if key == 'range_m' and value != 'auto':
            try: return math.isfinite(float(value)) and float(value) > 0
            except (ValueError, TypeError): return False
        return key not in self.choices or str(value) in self.choices[key]

    def set_options(self, options):
        self.options.update({k: v for k, v in options.items() if self.valid_option(k, v)})
        for key, variable in self.variables.items(): variable.set(str(self.options[key]))

    def draw(self, sample): raise NotImplementedError
    def describe(self, sample): return []
    def destroy(self):
        for variable in self.variables.values():
            for modes, callback in variable.trace_info(): variable.trace_remove(modes, callback)
        self.widget.destroy()


class GenericRenderer(Renderer):
    """JSON plus an image or a layout-driven heat map, with no type assumptions."""
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.image = ttk.Label(self.widget, anchor='center')
        self.image.pack(fill='both', expand=True)
        self.text = tk.Text(self.widget, height=7, width=28, wrap='word', state='disabled')
        self.text.pack(fill='both', expand=True)
        self.photo = None

    def draw(self, sample):
        self.text.configure(state='normal')
        self.text.delete('1.0', 'end')
        self.text.insert('1.0', json.dumps(sample.payload, indent=2, sort_keys=True))
        self.text.configure(state='disabled')
        picture = (Image.fromarray(sample.image) if sample.image is not None else
                   heatmap(sample.fields, sample.entry.get('layout', [])) if sample.fields else None)
        if picture is not None:
            picture.thumbnail((max(1, self.widget.winfo_width()),
                               max(1, self.widget.winfo_height()//2)), Image.Resampling.NEAREST)
            self.photo = ImageTk.PhotoImage(picture)
            self.image.configure(image=self.photo)

    def describe(self, sample):
        return [(field['name'], f"{field['dtype']} {field['shape']}")
                for field in sample.entry.get('layout', [])]

    def clear(self):
        # No sample means no heat map either: a stale picture under a "no
        # sample" readout is exactly the quiet lie this renderer exists to
        # avoid.
        self.text.configure(state='normal'); self.text.delete('1.0', 'end'); self.text.configure(state='disabled')
        self.image.configure(image=''); self.photo = None


def colour_field(values, low, high, lut):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    normalized = np.zeros(values.shape)
    normalized[valid] = np.clip((values[valid]-low)/max(1e-12, high-low), 0, 1)
    rgb = lut[(normalized*255).astype(np.uint8)].copy()
    rgb[~valid] = (32, 32, 32)
    return rgb


def polar_points(sample, up='forward'):
    calibration = sample.entry['calibration']
    ranges = np.asarray(sample.fields['range_m']).ravel()
    bearings = calibration['angle_min_deg'] + np.arange(ranges.size)*calibration['angle_increment_deg']
    elevation = math.radians(calibration.get('elevation_deg', 0.))
    angles = np.radians(bearings)
    rays = np.column_stack((np.cos(angles)*math.cos(elevation), np.sin(angles)*math.cos(elevation),
                            np.full(len(angles), math.sin(elevation))))
    if 'pose_world' in sample.payload:
        vectors = rays @ np.asarray(sample.payload['pose_world'])[:3, :3].T
        east, north = vectors[:, 0]*ranges, -vectors[:, 1]*ranges
        if up == 'forward':
            heading = math.radians(sample.payload.get('body', {}).get('heading_deg', 0.))
            east, north = east*math.cos(heading)-north*math.sin(heading), east*math.sin(heading)+north*math.cos(heading)
        return np.column_stack((east, north)), bearings
    return np.column_stack((rays[:, 1]*ranges, rays[:, 0]*ranges)), bearings



class ImageRenderer(Renderer):
    DEFAULTS = {'fit': 'contain'}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.option('fit', ('contain', 'cover', 'native'))
        self.canvas = tk.Canvas(self.widget, background=theme.CANVAS, highlightthickness=0, width=240, height=180)
        self.canvas.pack(fill='both', expand=True)
        self.item = self.canvas.create_image(0, 0, anchor='nw')
        self.photo = None; self.bounds = (0, 0, 1, 1); self.cap = None

    def clear(self):
        self.canvas.itemconfigure(self.item, image=''); self.photo = None

    def show_image(self, image, measured=False):
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        if self.cap:
            width, height = min(width, self.cap[0]), min(height, self.cap[1])
        fit = self.options['fit']
        scale = (max(width/image.width, height/image.height) if fit == 'cover' else
                 min(width/image.width, height/image.height)) if fit != 'native' else 1.
        size = (max(1, round(image.width*scale)), max(1, round(image.height*scale)))
        image = image.resize(size, Image.Resampling.NEAREST if measured else Image.Resampling.BILINEAR)
        x, y = (self.canvas.winfo_width()-size[0])//2, (self.canvas.winfo_height()-size[1])//2
        self.bounds = (x, y, size[0], size[1])
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.coords(self.item, x, y); self.canvas.itemconfigure(self.item, image=self.photo)

    def draw(self, sample):
        if sample.image is not None: self.show_image(Image.fromarray(sample.image))

    def describe(self, sample):
        m = sample.entry.get('model', {})
        rows = [('resolution', f"{sample.image.shape[1]} × {sample.image.shape[0]}")] if sample.image is not None else []
        for axis, size, focal in (('horizontal', 'width_px', 'fx_px'), ('vertical', 'height_px', 'fy_px')):
            if m.get(focal): rows.append((axis+' FOV', f"{math.degrees(2*math.atan(m[size]/(2*m[focal]))):.1f}°"))
        rows.append(('pose', json.dumps(sample.payload.get('pose', {}), sort_keys=True)))
        return rows


class RangeImageRenderer(ImageRenderer):
    DEFAULTS = {'fit': 'contain', 'field': 'range_m'}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.option('field', ('range_m', 'confidence'))
        self.hover = ttk.Label(self.widget, text='Hover for metres'); self.hover.pack(fill='x')
        self.ranges = None
        self.canvas.bind('<Motion>', self._hover)

    def draw(self, sample):
        self.ranges = sample.fields['range_m']
        field = self.options['field']; values = sample.fields[field]
        model = sample.entry['model']
        low, high = (0, 255) if field == 'confidence' else (model['min_range_m'], model['max_range_m'])
        rgb = colour_field(values, low, high, theme.CONFIDENCE_LUT if field == 'confidence' else theme.RANGE_LUT)
        rgb[~np.isfinite(self.ranges)] = (32, 32, 32)
        self.show_image(Image.fromarray(rgb), measured=True)

    def value_at(self, x, y):
        if self.ranges is None: return None
        left, top, width, height = self.bounds
        if not left <= x < left+width or not top <= y < top+height: return None
        row = min(self.ranges.shape[0]-1, int((y-top)*self.ranges.shape[0]/height))
        col = min(self.ranges.shape[1]-1, int((x-left)*self.ranges.shape[1]/width))
        value = float(self.ranges[row, col])
        return value if math.isfinite(value) else None

    def clear(self):
        # Hover reads self.ranges, so a cleared pane must stop reporting
        # metres from the sample it no longer shows.
        super().clear(); self.ranges = None
        self.hover.configure(text='Hover for metres')

    def _hover(self, event):
        if self.ranges is None:
            self.hover.configure(text='Hover for metres'); return
        value = self.value_at(event.x, event.y)
        self.hover.configure(text='no return' if value is None else f'{value:.3f} m')

    def describe(self, sample):
        values = sample.fields['range_m']; valid = np.isfinite(values)
        return [('returns', f'{valid.sum()}/{values.size}'), ('field', self.options['field'])]


class ScalarRenderer(Renderer):
    DEFAULTS = {'window_s': '60'}
    VALUES = {'altimeter.barometric': ('altitude_m', 'm'),
              'heading.magnetometer': ('heading_deg', '°'),
              'environment.temperature': ('temperature_c', '°C')}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        # A heading wraps around: its line plot is noise, so the compass rose
        # is the whole graphic and the plot window option does not apply.
        self.rose = entry['type'] == 'heading.magnetometer'
        if not self.rose: self.option('window_s', ('10', '30', '60'))
        self.value = ttk.Label(self.widget, text='waiting', font=('', 22)); self.value.pack(fill='x')
        self.canvas = tk.Canvas(self.widget, width=240, height=150, background=theme.CANVAS, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

    def measurement(self, sample):
        key, unit = self.VALUES[sample.type]
        value = sample.payload.get(key)
        return (float(value) if sample.status and value is not None and math.isfinite(value) else None), unit

    def draw(self, sample):
        value, unit = self.measurement(sample)
        self.value.configure(text='no return' if value is None else f'{value:.3f} {unit}')
        self.canvas.delete('all')
        width, height = max(20, self.canvas.winfo_width()), max(20, self.canvas.winfo_height())
        if self.rose:
            self._rose(value, width, height)
            return
        window = float(self.options['window_s'])
        samples = [s for s in self.history if sample.sim_time_s-window <= s.sim_time_s <= sample.sim_time_s]
        # Preserve invalid gaps; bound Canvas work by screen width.
        samples = samples[::max(1, len(samples)//max(1, width))]
        points = [(s.sim_time_s, self.measurement(s)[0]) for s in samples]
        finite = [v for _, v in points if v is not None]
        if finite:
            low, high = min(finite), max(finite)
            segment = []
            for stamp, number in points + [(sample.sim_time_s, None)]:
                if number is None:
                    if len(segment) >= 4: self.canvas.create_line(*segment, fill=theme.ACCENT)
                    segment = []; continue
                segment.extend((5+(stamp-sample.sim_time_s+window)/window*(width-10),
                                height-10-(number-low)/max(1e-6, high-low)*(height-20)))
        if sample.type == 'heading.magnetometer' and value is not None:
            x, y, radius = width-38, 38, 25
            self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline=theme.DIM)
            self.canvas.create_text(x, y-radius-5, text='N', fill=theme.TEXT)
            angle = math.radians(value)
            self.canvas.create_line(x, y, x+radius*math.sin(angle), y-radius*math.cos(angle), fill=theme.OK, arrow='last')

    def _rose(self, value, width, height):
        """A full-size compass rose centred in the pane, ticks and all."""
        x, y = width/2, height/2
        radius = max(20., min(width, height)/2-12)
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline=theme.DIM)
        for step in range(72):
            angle = math.radians(step*5)
            inner = radius*(.80 if step % 18 else .65)  # long ticks at the cardinals
            self.canvas.create_line(x+inner*math.sin(angle), y-inner*math.cos(angle),
                x+radius*math.sin(angle), y-radius*math.cos(angle), fill=theme.DIM)
        for label, angle in (('N', 0), ('E', 90), ('S', 180), ('W', 270)):
            self.canvas.create_text(x+radius*.45*math.sin(math.radians(angle)),
                                     y-radius*.45*math.cos(math.radians(angle)), text=label, fill=theme.TEXT)
        if value is None: return
        angle = math.radians(value)
        self.canvas.create_line(x, y, x+radius*.9*math.sin(angle), y-radius*.9*math.cos(angle),
                                fill=theme.OK, arrow='last')

    def clear(self):
        self.value.configure(text='waiting'); self.canvas.delete('all')

    def describe(self, sample):
        if self.rose: return []  # the value line above the rose already carries the heading
        return [('window', f"{self.options['window_s']} simulated seconds")]


class BeamRenderer(ScalarRenderer):
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.gauge = ttk.Progressbar(self.widget, maximum=1.); self.gauge.pack(fill='x')

    def measurement(self, sample):
        value = sample.payload.get('range_m')
        return (float(value) if sample.status and value is not None and math.isfinite(value) else None), 'm'

    def draw(self, sample):
        super().draw(sample)
        value, _ = self.measurement(sample); model = sample.entry['model']
        self.gauge['value'] = 0 if value is None else np.clip((value-model['min_range_m'])/(model['max_range_m']-model['min_range_m']), 0, 1)
        self.gauge.state(['disabled'] if value is None else ['!disabled'])

    def clear(self):
        super().clear(); self.gauge['value'] = 0; self.gauge.state(['disabled'])

    def describe(self, sample):
        return [(k, str(sample.payload.get(k, '—'))) for k in ('returns', 'samples', 'reducer', 'confidence')]


class PolarRenderer(Renderer):
    DEFAULTS = {'up': 'forward', 'range_m': 'auto'}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.option('up', ('forward', 'north'))
        self.option('range_m', ('auto', '5', '10', '25', '50', '100'), editable=True)
        self.canvas = tk.Canvas(self.widget, width=240, height=200, background=theme.CANVAS, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)

    def draw(self, sample):
        points, _ = polar_points(sample, self.options['up'])
        radius_m = sample.entry['model']['max_range_m'] if self.options['range_m'] == 'auto' else float(self.options['range_m'])
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        x, y = width/2, height/2; radius = max(1, min(width, height)/2-16)
        self.canvas.delete('all')
        for fraction in (.25, .5, .75, 1.):
            r = radius*fraction
            self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=theme.GRID)
        self.canvas.create_text(x, 8, text=('N' if self.options['up'] == 'north' else 'Forward')+f' · {radius_m:g} m', fill=theme.TEXT)
        confidence = sample.fields.get('confidence', np.full(len(points), 255)).ravel()
        for (east, north), conf in zip(points, confidence):
            if not np.isfinite(east+north) or math.hypot(east, north) > radius_m: continue
            px, py = x+east/radius_m*radius, y-north/radius_m*radius
            colour = '#%02x%02x%02x' % tuple(theme.CONFIDENCE_LUT[int(np.clip(conf, 0, 255))])
            self.canvas.create_oval(px-2, py-2, px+2, py+2, fill=colour, outline='')

    def describe(self, sample):
        ranges = sample.fields['range_m'].ravel(); valid = np.isfinite(ranges)
        rows = [('returns', f'{valid.sum()}/{ranges.size}')]
        if valid.any():
            index = int(np.nanargmin(ranges)); _, bearings = polar_points(sample)
            rows.append(('nearest', f'{ranges[index]:.3f} m at {bearings[index]:.1f}° sensor bearing'))
        return rows


def stereo_groups(manifest):
    groups = {}
    for sid, entry in manifest.get('sensors', {}).items():
        if entry['type'] == 'camera.rgb' and entry.get('sync_group'):
            groups.setdefault(entry['sync_group'], []).append(sid)
    return {'stereo:'+group: tuple(sorted(ids)) for group, ids in groups.items() if len(ids) == 2}


def device_entries(session):
    entries = dict(session.devices)
    for sid, members in stereo_groups(session.manifest).items():
        entries[sid] = dict(type='camera.stereo', transport='video',
                           rate_hz=min(entries[m]['rate_hz'] for m in members), members=members)
    return entries


def sample_identity(sample):
    return sample.provider_session_id, sample.generation, sample.reset_epoch, sample.clock_epoch


def stereo_sample(sid, left, right, entry):
    if left.capture_id != right.capture_id or sample_identity(left) != sample_identity(right): return None
    if left.image is None or right.image is None: return None
    poses = [s.payload.get('pose_world') for s in (left, right)]
    baseline = (float(np.linalg.norm(np.asarray(poses[0])[:3, 3]-np.asarray(poses[1])[:3, 3]))
                if all(p is not None for p in poses) else None)
    return Sample(sid, 'camera.stereo', left.capture_id, left.sequence, left.sim_time_s,
        min(left.status, right.status), dict(members=[left.sensor_id, right.sensor_id], baseline_m=baseline),
        fields={'left': left.image, 'right': right.image}, entry=entry,
        provider_session_id=left.provider_session_id, generation=left.generation,
        reset_epoch=left.reset_epoch, clock_epoch=left.clock_epoch)


class StereoStream:
    """A view of two refcounted streams; never opens another compact ring."""
    def __init__(self, session, sid, *, accounting='from_attach'):
        self.session, self.selected = session, sid
        self.members = stereo_groups(session.manifest)[sid]
        self.streams = []
        try:
            for member in self.members: self.streams.append(session.open(member, accounting=accounting))
        except Exception:
            self.close(); raise
        self.closed = False

    @property
    def entry(self): return device_entries(self.session)[self.selected]
    @property
    def revision(self): return tuple(stream.revision for stream in self.streams)
    @property
    def required(self): return False
    @property
    def attached_at(self): return tuple(stream.attached_at for stream in self.streams)
    @property
    def skipped(self): return sum(stream.skipped for stream in self.streams)
    @property
    def association_drops(self): return sum(stream.association_drops for stream in self.streams)
    @property
    def late_drops(self): return sum(stream.late_drops for stream in self.streams)
    def getFps(self): return self.entry['rate_hz']
    @property
    def history(self):
        if self.selected not in device_entries(self.session): return {}
        left, right = [{s.capture_id: s for s, _ in stream.history.values()} for stream in self.streams]
        out = {}
        for capture in sorted(left.keys() & right.keys()):
            sample = stereo_sample(self.selected, left[capture], right[capture], self.entry)
            if sample is not None: out[capture] = (sample, None)
        return out

    @property
    def latest(self):
        history = self.history
        if not history: return None
        sample = next(reversed(history.values()))[0]
        captures = [s.latest.capture_id if s.latest is not None else None for s in self.streams]
        sample.payload['latest_captures'] = captures
        return sample

    def refresh(self):
        for stream in self.streams: stream.refresh()
    def getSeq(self):
        self.refresh(); sample = self.latest
        return sample.sequence if sample else 0
    def close(self):
        for stream in self.streams: stream.close()
        self.streams = []; self.closed = True


def open_device(session, sid, **kwargs):
    # A pair view cannot itself be a required input: its members carry their
    # own flags, and the primary camera is held required separately.
    if sid in stereo_groups(session.manifest):
        return StereoStream(session, sid, accounting=kwargs.get('accounting', 'from_attach'))
    return session.open(sid, **kwargs)


def stream_stats(report, stream):
    """A pair reports its worst member; an absent device reports nothing."""
    members = getattr(stream, 'members', (stream.selected,))
    stats = [report['devices'][member] for member in members if member in report['devices']]
    if not stats: return dict(grade='unknown', achieved_hz=None)
    return min(stats, key=lambda s: s['achieved_hz'] if s['achieved_hz'] is not None else -1)


def stream_at_capture(stream, capture):
    return next((s for s, _ in reversed(stream.history.values()) if s.capture_id == capture), None)


def synchronized_capture(streams):
    """Newest shared capture, or a labelled incomplete newest capture if none exists."""
    sets = [{s.capture_id for s, _ in stream.history.values()} for stream in streams]
    if not sets: return None, False
    common = set.intersection(*sets)
    union = set.union(*sets)
    return (max(common), True) if common else (max(union) if union else None, False)


def envelope_arrays(history, field, axis, now, window, rate, pixels=200):
    """Build only the selected time window and compress it to a screen envelope.

    Input capacity follows the configured rate; the extra pair avoids a forced
    half-compression right at a full window's boundary.
    """
    series = EnvelopeSeries(capacity=max(2, 2*math.ceil(rate*window/2)+2))
    for sample in history:
        if now-window <= sample.sim_time_s <= now and sample.status:
            value = sample.payload.get(field)
            if axis is not None: value = value.get(axis) if isinstance(value, dict) else None
            if isinstance(value, (int, float)) and math.isfinite(value): series.add(sample.sim_time_s, value)
    while len(series) > max(2, pixels): series.compress()
    return series


class VectorRenderer(Renderer):
    DEFAULTS = {'window_s': '60'}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.option('window_s', ('10', '30', '60'))
        self.canvas = tk.Canvas(self.widget, width=280, height=220, background=theme.CANVAS, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.envelopes = {}

    def draw(self, sample):
        self.canvas.delete('all'); self.envelopes = {}
        width, height = max(80, self.canvas.winfo_width()), max(100, self.canvas.winfo_height())
        window = float(self.options['window_s'])
        for row, (field, label) in enumerate((('angular_rate_dps', 'Angular rate · °/s'), ('specific_force_mps2', 'Specific force · m/s²'))):
            arrays = []
            for axis in 'xyz':
                series = envelope_arrays(self.history or [sample], field, axis, sample.sim_time_s, window,
                                         self.entry['rate_hz'], min(160, width//2))
                self.envelopes[(field, axis)] = series
                arrays.append(series.plot_arrays())
            values = [v for _, _, low, high in arrays for v in low+high]
            low, high = (min(values), max(values)) if values else (0., 1.)
            top, bottom = row*height/2+24, (row+1)*height/2-14
            self.canvas.create_text(5, top-12, anchor='w', text=label+'  X red · Y green · Z blue', fill=theme.TEXT)
            for (x, mean, minimum, maximum), colour in zip(arrays, (theme.DANGER, theme.OK, theme.ACCENT)):
                if not x: continue
                px = [5+(stamp-sample.sim_time_s+window)/window*(width-10) for stamp in x]
                scale = lambda data: [bottom-(v-low)/max(1e-6, high-low)*(bottom-top) for v in data]
                ys, lo, hi = scale(mean), scale(minimum), scale(maximum)
                if len(px) > 1:
                    polygon = [v for point in zip(px, lo) for v in point]+[v for point in reversed(list(zip(px, hi))) for v in point]
                    self.canvas.create_polygon(*polygon, fill=colour, stipple='gray25', outline='')
                    self.canvas.create_line(*[v for point in zip(px, ys) for v in point], fill=colour)
            self.canvas.create_text(5, bottom+8, anchor='w', text=f'{sample.sim_time_s-window:.1f} … {sample.sim_time_s:.1f} s (sim)', fill=theme.DIM)

    def describe(self, sample):
        if not sample.status: return [('state', 'invalid IMU sample')]
        rows = []
        for field in ('angular_rate_dps', 'specific_force_mps2'):
            values = sample.payload.get(field) or {}  # a null field is invalid, not a crash
            axes = []
            for axis in 'xyz':
                value = values.get(axis)
                axes.append(f'{axis.upper()} {value:.3f}'
                            if isinstance(value, (int, float)) and math.isfinite(value) else f'{axis.upper()} —')
            rows.append((field, ' · '.join(axes)))
        return rows


def fix_state(sample):
    if sample is None: return 'no receiver'
    payload = sample.payload
    if not payload.get('fix_type'): return 'no fix'
    if not sample.status or payload.get('valid') is False: return 'fix rejected'
    return 'fix accepted'


class FixRenderer(Renderer):
    DEFAULTS = {'window_s': '60'}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.option('window_s', ('10', '30', '60'))
        self.quality = ttk.Label(self.widget, text='no receiver', wraplength=360); self.quality.pack(fill='x')
        self.canvas = tk.Canvas(self.widget, width=240, height=160, background=theme.CANVAS, highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.envelopes = {}

    def draw(self, sample):
        self.quality.configure(text=fix_state(sample))
        self.canvas.delete('all')
        if fix_state(sample) != 'fix accepted': return
        window = float(self.options['window_s'])
        self.envelopes = {axis: envelope_arrays(self.history or [sample], f'error_{axis}_m', None,
                          sample.sim_time_s, window, self.entry['rate_hz'], 120) for axis in ('north', 'east')}
        north, east = [self.envelopes[axis].plot_arrays() for axis in ('north', 'east')]
        if not north[0] or not east[0]: return
        extent = max(.1, *(abs(v) for a in (north, east) for column in a[1:] for v in column))
        width, height = max(20, self.canvas.winfo_width()), max(20, self.canvas.winfo_height())
        scale = min(width, height)/2/extent*.8
        self.canvas.create_text(5, 10, anchor='w', text=f'N ↑ · E → · ±{extent:.2f} m', fill=theme.DIM)
        # Time keys prevent separately missing axes from manufacturing a position.
        n = {t: values for t, *values in zip(*north)}
        e = {t: values for t, *values in zip(*east)}
        for stamp in n.keys() & e.keys():
            nm, nl, nh = n[stamp]; em, el, eh = e[stamp]
            x, y = width/2+em*scale, height/2-nm*scale
            self.canvas.create_rectangle(width/2+el*scale, height/2-nh*scale,
                                         width/2+eh*scale, height/2-nl*scale, outline=theme.GRID)
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=theme.ACCENT, outline='')

    def describe(self, sample):
        return [('state', fix_state(sample))]+[(key, str(sample.payload.get(key, '—'))) for key in
            ('fix_type', 'satellites', 'hdop', 'vdop', 'lat_deg', 'lon_deg', 'alt_m',
             'vel_north_mps', 'vel_east_mps', 'vel_down_mps')]


class StereoRenderer(ImageRenderer):
    DEFAULTS = {'fit': 'contain', 'mode': 'side-by-side'}
    def __init__(self, parent, entry):
        super().__init__(parent, entry)
        self.option('mode', ('side-by-side', 'anaglyph', 'difference'))

    def draw(self, sample):
        left, right = sample.fields['left'], sample.fields['right']
        mismatch = left.shape != right.shape
        mode = 'side-by-side' if mismatch else self.options['mode']
        if mode == 'side-by-side':
            rgb = np.zeros((max(left.shape[0], right.shape[0]), left.shape[1]+right.shape[1], 3), dtype=np.uint8)
            rgb[:left.shape[0], :left.shape[1]] = left; rgb[:right.shape[0], left.shape[1]:] = right
        elif mode == 'anaglyph':
            rgb = right.copy(); rgb[:, :, 0] = left[:, :, 0]
        else: rgb = np.abs(left.astype(np.int16)-right.astype(np.int16)).astype(np.uint8)
        if mismatch:
            fit = self.options['fit']; self.options['fit'] = 'native'
            try: self.show_image(Image.fromarray(rgb))
            finally: self.options['fit'] = fit
        else: self.show_image(Image.fromarray(rgb))

    def describe(self, sample):
        mismatch = sample.fields['left'].shape != sample.fields['right'].shape
        latest = sample.payload.get('latest_captures', [sample.capture_id]*2)
        state = 'matched' if latest == [sample.capture_id]*2 else f'waiting for matching capture (latest {latest})'
        return [('sync', state), ('baseline', f"{sample.payload['baseline_m']:.4f} m" if sample.payload.get('baseline_m') is not None else 'unknown'),
                ('resolution', 'mismatch: native side-by-side' if mismatch else str(sample.fields['left'].shape[:2]))]


RENDERERS = {'camera.rgb': ImageRenderer, 'camera.stereo': StereoRenderer,
             'motion.imu': VectorRenderer, 'position.gnss': FixRenderer, 'range.': BeamRenderer,
             'lidar.scan2d': PolarRenderer, 'lidar.range_image': RangeImageRenderer,
             'altimeter.barometric': ScalarRenderer, 'heading.magnetometer': ScalarRenderer,
             'environment.temperature': ScalarRenderer}



def resolve_renderer(kind):
    if kind in RENDERERS: return RENDERERS[kind]
    for prefix in sorted(RENDERERS, key=len, reverse=True):
        if prefix.endswith('.') and kind.startswith(prefix): return RENDERERS[prefix]
    return GenericRenderer


class CircleButton(tk.Canvas):
    """A small translucent-seeming disc, always visible in a shrunken pane.

    Tk has no real translucency, so a disc is a faint tint of the panel
    colour that brightens on hover and lights when its state is on. The
    other half of "always visible" is size and pack order: three fixed
    eighteen-pixel discs claim less header width than the title does, and
    being packed before it they win the space, so a narrow pane clips the
    name, never the controls.
    """
    RADIUS = 9

    def __init__(self, parent, glyph, command):
        super().__init__(parent, width=2*self.RADIUS, height=2*self.RADIUS,
                         background=theme.PANEL, highlightthickness=0, bd=0, cursor='hand2')
        self.command = command; self.active = False; self.hovering = False
        edge = 2*self.RADIUS-2
        self.disc = self.create_oval(1, 1, edge, edge, width=1)
        self.mark = self.create_text(self.RADIUS, self.RADIUS, text=glyph, font=('', 9, 'bold'))
        self.bind('<Button-1>', lambda event: self.invoke())
        self.bind('<Enter>', lambda event: self._hovered(True))
        self.bind('<Leave>', lambda event: self._hovered(False))
        self._paint()

    def invoke(self):
        if callable(self.command): self.command()

    def set_active(self, active):
        """Light the disc when its pane state is on, dim it when off."""
        self.active = bool(active); self._paint()

    def _hovered(self, hovering):
        self.hovering = hovering; self._paint()

    def _paint(self):
        if self.active:
            fill, edge, glyph = theme.blend(theme.PANEL, theme.ACCENT, .3), theme.ACCENT, theme.TEXT
        elif self.hovering:
            fill, edge, glyph = theme.blend(theme.PANEL, theme.TEXT, .12), theme.DIM, theme.TEXT
        else:
            fill, edge, glyph = theme.blend(theme.PANEL, theme.TEXT, .05), theme.GRID, theme.DIM
        self.itemconfigure(self.disc, fill=fill, outline=edge)
        self.itemconfigure(self.mark, fill=glyph)


class DevicePane(tk.Frame, tk.Wm):
    def __init__(self, parent, stream, close, options=None, show_readout=False):
        super().__init__(parent, padx=5, pady=5, relief='groove', background=theme.PANEL)
        self.stream = stream
        self.revision = stream.revision
        # The generation this pane was born under: a revision bump alone is
        # ambiguous between a profile change and a simulator reset, and only
        # the profile change is allowed to say "profile changed".
        self.generation = (stream.session.identity[1] if stream.session.identity else None)
        self.frozen = tk.BooleanVar(value=False)
        # The graphic owns the whole cell by default; the info disc reveals
        # the text strip, and the choice persists with the pane's record.
        self.show_readout = bool(show_readout)
        header = ttk.Frame(self, style='Panel.TFrame'); self.header = header; header.pack(fill='x')
        self.title = ttk.Label(header, style='Panel.TLabel',
                               text=f"{stream.selected} · {stream.entry['type']} · {stream.getFps():g} Hz")
        # Pack order is priority order under `pack`, so the three discs claim
        # their fixed width before the title is offered what is left: a pane
        # squeezed narrow clips its own name and never its controls. That is
        # the whole reason the title is packed last despite sitting leftmost.
        self.close_button = CircleButton(header, '×', close)
        self.close_button.pack(side='right', padx=1)
        self.freeze_button = CircleButton(header, '❄', lambda: self.frozen.set(not self.frozen.get()))
        self.freeze_button.pack(side='right', padx=1)
        self.text_button = CircleButton(header, 'i', self._toggle_readout)
        self.text_button.pack(side='right', padx=1)
        self.text_button.set_active(self.show_readout)
        self.title.pack(side='left')
        self.readout = ttk.Label(self, style='Panel.TLabel', text='not subscribed',
                                 anchor='nw', justify='left', wraplength=400)
        self.readout.bind('<Configure>', self._wrap_readout)
        self._collapsed = False
        self.renderer = resolve_renderer(stream.entry['type'])(self, stream.entry)
        self.renderer.set_options(options or {})
        self.resize_grip = ttk.Label(self, style='Panel.TLabel', text="↘", cursor="sizing")
        self.resize_grip.pack(side="bottom", anchor="e")
        self._show_body()
        self.last_paint = -1e9
        self.changed = False
        self.redraw_pending = False
        self.display_capture = None; self.held_capture = None
        self.frozen.trace_add('write', self._freeze_changed)

    def _freeze_changed(self, *args):
        self.held_capture = self.display_capture if self.frozen.get() else None
        self.freeze_button.set_active(self.frozen.get())
        self.last_paint = -1e9

    def _toggle_readout(self):
        self.show_readout = not self.show_readout
        self.text_button.set_active(self.show_readout)
        # Whichever body is coming back has not been drawn since it left, and
        # a frozen pane would otherwise short-circuit on its held capture and
        # never redraw. Ask for one paint outright.
        self.redraw_pending = True; self.last_paint = -1e9
        self._show_body()

    def _show_body(self):
        """Give the cell to one body or the other -- graphic or text.

        The readout used to be a strip pinned under a live graphic, where it
        was too cramped to read, so the info disc swaps the two instead of
        stacking them. The loser is unpacked rather than covered: nothing
        draws into it, and it cannot steal height from the winner.
        """
        if self._collapsed: return  # both return with the body at arrange exit
        shown, hidden = ((self.readout, self.renderer.widget) if self.show_readout
                         else (self.renderer.widget, self.readout))
        hidden.pack_forget()
        # Packed after the grip, so the grip keeps its slab and the body takes
        # what is left: on a pane too short for both, the sizing corner stays.
        shown.pack(fill='both', expand=True)

    def _wrap_readout(self, event):
        # The readout owns the whole cell now, so it wraps to the cell instead
        # of to a fixed width that a narrow pane would run straight past.
        wrap = max(80, event.width-8)
        if wrap != int(self.readout['wraplength']): self.readout.configure(wraplength=wrap)

    def collapse(self, collapsed):
        """Show only the title strip while the bench is being arranged.

        The renderer, stream and options all stay alive; painting is paused
        at the grid, so nothing draws into a hidden body.
        """
        self._collapsed = collapsed
        if collapsed:
            for widget in (self.renderer.widget, self.readout, self.resize_grip):
                widget.pack_forget()
        else:
            # The same packing sequence as construction, so the restored
            # layout is pixel-identical to the one the operator left.
            self.resize_grip.pack(side='bottom', anchor='e')
            self.redraw_pending = True
            self._show_body()

    def _missing(self, capture):
        self.display_capture = None
        if hasattr(self.renderer, 'clear'): self.renderer.clear()
        elif hasattr(self.renderer, 'canvas'): self.renderer.canvas.delete('all')
        elif hasattr(self.renderer, 'text'):
            self.renderer.text.configure(state='normal'); self.renderer.text.delete('1.0', 'end'); self.renderer.text.configure(state='disabled')
        if hasattr(self.renderer, 'quality'): self.renderer.quality.configure(text='no receiver')
        self.readout.configure(text=f'no sample at capture {capture}' if capture is not None else 'waiting for sample')

    def destroy(self):
        for modes, callback in self.frozen.trace_info(): self.frozen.trace_remove(modes, callback)
        self.renderer.destroy()
        super().destroy()

    def paint(self, stats, rate, *, hold=False, capture=None):
        now = time.monotonic()
        if now-self.last_paint+1e-9 < 1/rate: return False
        self.last_paint = now
        if self.revision != self.stream.revision:
            self.revision = self.stream.revision
            # A revision bump is ambiguous: only a generation change says
            # "profile changed"; a reset epoch change says so through its
            # counted drops instead.
            identity = self.stream.session.identity
            if identity and self.generation != identity[1]:
                self.generation = identity[1]; self.changed = True
            self.display_capture = None
            self.frozen.set(False)
            options = dict(self.renderer.options)
            self.renderer.destroy()
            self.renderer = resolve_renderer(self.stream.entry['type'])(self, self.stream.entry)
            self.renderer.set_options(options)
            self._show_body()  # a fresh renderer packs itself; the text body still wins the cell
        self.stream.refresh()
        # A pane's own freeze holds that pane alone, even under a global one.
        if self.frozen.get():
            hold, capture = True, self.held_capture
        if hold and capture is not None and self.display_capture == capture and not self.redraw_pending: return False
        sample = stream_at_capture(self.stream, capture) if hold else self.stream.latest
        if sample is None:
            self._missing(capture)
            return False
        self.display_capture = sample.capture_id
        if self.show_readout:
            # The graphic is not on screen. Drawing into an unpacked widget
            # would size every plot to its minimum and leave that behind for
            # the snapshot, so the text body simply does not draw one.
            self.redraw_pending = True
        else:
            self.renderer.history = [s for s, _ in self.stream.history.values() if s.capture_id <= sample.capture_id]
            self.renderer.draw(sample)
            self.renderer.history = []
            self.redraw_pending = False
        hz = stats.get('achieved_hz')
        rate_text = '?' if hz is None else f'{hz:.1f}'
        rows = ' · '.join(f'{k}: {v}' for k, v in self.renderer.describe(sample))
        self.readout.configure(text=(f"{DOTS[stats['grade']]} {rate_text}/{self.stream.getFps():g} Hz (sim) · "
            f"{sample.sim_time_s:.3f} s · capture {sample.capture_id}\n"
            f"attached at sequence {self.stream.attached_at} · gaps {self.stream.skipped} · "
            f"drops {self.stream.association_drops} · late {self.stream.late_drops}"
            + (' · profile changed' if self.changed else '') + '\n' + rows))
        return True


def _wants_video(pane):
    """Whether a pane is actually painting frames rather than a text readout."""
    return pane.stream.entry['transport'] != 'compact' and not pane.show_readout


class DeviceGrid(ttk.Frame):
    """Auto layout and visibility own the lifetime of optional subscriptions."""
    def __init__(self, parent, session, changed=lambda: None):
        super().__init__(parent)
        self.session = session; self.changed = changed
        self.wanted = []; self.panes = {}; self.visible = False
        self.token = None; self.skipped_paints = 0; self.paints = 0
        self.started = time.monotonic()
        self.state = layout.auto_layout([]); self.sashes = {}; self._dimensions = (0, 0)
        self.arranging = False; self.targets = {}
        self.popped = {}; self.freeze_enabled = False; self.freeze_targets = {}; self.freeze_message = 'live'
        self.freeze_token = None; self.freeze_capture = None; self.freeze_synchronized = True
        self.pending_change = False
        self.export_action = lambda sid, kind: None
        self.export_enabled = lambda: False

    def toggle(self, sid):
        self.state = self.serialize()
        if sid in self.wanted:
            self.wanted.remove(sid)
            self.state['panes'] = [p for p in self.state['panes'] if p['id'] != sid]
            self.state['popped'].pop(sid, None)
        else:
            self.wanted.append(sid)
            self.state['panes'].append(dict(id=sid, row=-1, col=-1))
        self.state = layout.reconcile(self.state)
        self.reconcile(); self.changed()

    def reconcile(self):
        token = self.session.generation_token
        # The first generation seen is not a change; a generation change
        # observed while the tab is hidden must survive until the panes are
        # recreated visible, so it is remembered rather than consumed here.
        if self.token is not None and token != self.token: self.pending_change = True
        self.token = token
        changed = self.pending_change
        if not self.state['panes'] and self.wanted: self.state = layout.auto_layout(self.wanted)
        self.serialize()
        entries = device_entries(self.session)
        for sid in list(self.panes):
            if changed or (not self.visible and sid not in self.popped) or sid not in self.wanted or sid not in entries:
                self._discard_pane(sid)
                if sid not in self.wanted: self.state['popped'].pop(sid, None)
        created = False
        if self.visible or self.state['popped']:
            for sid in self.wanted:
                if sid in self.panes or sid not in entries or (not self.visible and sid not in self.state['popped']): continue
                stream = open_device(self.session, sid, accounting='from_attach')
                record = next((p for p in self.state['panes'] if p['id'] == sid), None)
                if record is None:
                    # A wanted id without a record (a hand-mutated wanted
                    # list) still gets a pane, seated by the unplaced marker.
                    record = dict(id=sid, row=-1, col=-1, options={})
                    self.state['panes'].append(record)
                    self.state = layout.reconcile(self.state)
                pane = DevicePane(self, stream, lambda sid=sid: self.toggle(sid), record['options'],
                                  record.get('readout', False))
                for target in (pane.header, pane.title):
                    target.configure(cursor='fleur')
                    target.bind('<ButtonRelease-1>', lambda event, sid=sid: self._drop(sid, event))
                pane.resize_grip.bind('<ButtonRelease-1>', lambda event, sid=sid: self._drop(sid, event, True))
                pane.changed = changed
                if self.arranging: pane.collapse(True)
                self.panes[sid] = pane; created = True
                menu = tk.Menu(pane, tearoff=False)
                menu.add_command(label='Pop out / Return to grid', command=lambda sid=sid: self.return_to_grid(sid) if sid in self.popped else self.pop_out(sid))
                menu.add_command(label='Span right', command=lambda sid=sid: self.grow_span(sid, 0, 1))
                menu.add_command(label='Span down', command=lambda sid=sid: self.grow_span(sid, 1, 0))
                menu.add_command(label='Shrink to one cell', command=lambda sid=sid: self.shrink_span(sid))
                menu.add_command(label='Snapshot PNG', command=lambda sid=sid: self.export_action(sid, 'png'))
                menu.add_command(label='Dump JSON', command=lambda sid=sid: self.export_action(sid, 'json'))
                menu.add_command(label='Dump CSV', command=lambda sid=sid: self.export_action(sid, 'csv'))
                def popup(event, menu=menu, pane=pane):
                    record = next(p for p in self.state['panes'] if p['id'] == pane.stream.selected)
                    menu.entryconfigure(1, state='normal' if record['col']+record['colspan'] < self.state['columns'] else 'disabled')
                    menu.entryconfigure(2, state='normal' if record['row']+record['rowspan'] < self.state['rows'] else 'disabled')
                    menu.entryconfigure(3, state='normal' if record['rowspan'] > 1 or record['colspan'] > 1 else 'disabled')
                    exportable = self.export_enabled() and pane.display_capture is not None
                    # Only the PNG needs a drawn graphic; the dumps come from
                    # the sample history and stay available in text mode.
                    menu.entryconfigure(4, state='normal' if exportable and not pane.show_readout else 'disabled')
                    for i in (5, 6): menu.entryconfigure(i, state='normal' if exportable else 'disabled')
                    try: menu.tk_popup(event.x_root, event.y_root)
                    finally: menu.grab_release()
                pane.bind('<Button-3>', popup); pane.header.bind('<Button-3>', popup)
                for child in pane.header.winfo_children(): child.bind('<Button-3>', popup)
                if sid in self.state['popped']: self.pop_out(sid)
        # The remembered change has reached the panes it affected; panes born
        # later were not affected by it.
        if created or self.visible: self.pending_change = False
        self._arrange()

    def serialize(self):
        for sid, pane in self.popped.items(): self.state['popped'][sid] = {'geometry': pane.wm_geometry()}
        for pane in self.state['panes']:
            if pane['id'] in self.panes:
                pane['options'] = dict(self.panes[pane['id']].renderer.options)
                pane['readout'] = self.panes[pane['id']].show_readout
        return layout.reconcile(self.state)

    def restore(self, record):
        for sid in list(self.panes): self._discard_pane(sid)
        self.panes.clear()
        self.state = layout.reconcile(record)
        self.wanted = [p['id'] for p in self.state['panes']]
        self.reconcile()

    def swap(self, first, second):
        self.state = layout.swap(self.serialize(), first, second)
        self._arrange()

    def resize(self, sid, row, col):
        self.state = layout.span(self.serialize(), sid, row, col)
        self._arrange()

    def grow_span(self, sid, rows, columns):
        """Extend a pane's span one cell right or down, grid bounds permitting."""
        record = next(p for p in self.state['panes'] if p['id'] == sid)
        self.state = layout.span(self.serialize(), sid,
            record['row']+record['rowspan']-1+rows, record['col']+record['colspan']-1+columns)
        self._arrange(); self.changed()

    def shrink_span(self, sid):
        self.state = layout.unspan(self.serialize(), sid)
        self._arrange(); self.changed()

    def drop_device(self, sid, row, col):
        """Seat a tree row at an explicit cell, replacing any occupant outright.

        A drop never rearranges the grid on its own: a device dropped on an
        occupied cell closes the pane standing there and takes its place, so
        structural edits stay entirely in the operator's hands.
        """
        if sid not in device_entries(self.session) and sid not in self.wanted: return
        occupant = next((p['id'] for p in self.state['panes']
                         if p['id'] != sid and (row, col) in layout.cells(p)), None)
        if occupant is not None: self.toggle(occupant)
        if sid not in self.wanted: self.wanted.append(sid)
        self.state = layout.place(self.serialize(), sid, row, col)
        self.reconcile(); self.changed()

    def grow(self, axis):
        """Append one empty row at the bottom or column at the right edge."""
        record = self.serialize()
        self.state = (layout.insert_row if axis == 'rows' else layout.insert_column)(record, record[axis])
        self._arrange(); self.changed()

    def shrink(self, axis):
        """Remove the edge row or column; a structural edit never moves a pane."""
        record = self.serialize()
        if not self.edge_is_empty(axis): return
        self.state = (layout.remove_row if axis == 'rows' else layout.remove_column)(record, record[axis]-1)
        self._arrange(); self.changed()

    def edge_is_empty(self, axis):
        """Whether the bottom edge row or right edge column holds no pane."""
        index = self.state[axis]-1
        span = self.state['columns' if axis == 'rows' else 'rows']
        occupied = layout.occupied(self.state)
        return index > 0 and not any((index, i) in occupied if axis == 'rows'
                                     else (i, index) in occupied for i in range(span))

    def set_arranging(self, enabled):
        """Collapse the bench to its grid skeleton; intake keeps draining."""
        if enabled == self.arranging: return
        self.arranging = enabled
        for sid, pane in self.panes.items():
            # A floating window is never part of the grid skeleton, but every
            # pane -- popped ones included -- gets its body back on exit: a
            # pane popped mid-arrange must not stay a bare title strip.
            if sid not in self.popped or not enabled: pane.collapse(enabled)
            if not enabled: pane.last_paint = -1e9
        self._arrange()

    def _cell(self, event):
        col, row = self.grid_location(event.x_root-self.winfo_rootx(), event.y_root-self.winfo_rooty())
        return (max(0, min(row, self.state['rows']-1)), max(0, min(col, self.state['columns']-1)))

    def _drop(self, sid, event, resize=False):
        if sid in self.popped: return
        row, col = self._cell(event)
        if resize: self.resize(sid, row, col); return
        target = next((p['id'] for p in self.state['panes'] if (row, col) in layout.cells(p)), None)
        if target and target != sid: self.swap(sid, target)

    def _weight(self, axis, index, event):
        key, extent, offset = (('column_weights', self.winfo_width(), event.x_root-self.winfo_rootx())
                              if axis == 'col' else ('row_weights', self.winfo_height(), event.y_root-self.winfo_rooty()))
        weights = self.state[key]; total = sum(weights)
        before = sum(weights[:index]); pair = weights[index]+weights[index+1]
        weight = max(pair*.05, min(pair*.95, offset/max(1, extent)*total-before))
        weights[index], weights[index+1] = weight, pair-weight
        self._arrange()

    def _arrange(self):
        for i in range(max(self._dimensions[0], self.state['columns'])):
            self.columnconfigure(i, weight=round(self.state['column_weights'][i]*100) if i < self.state['columns'] else 0, minsize=0, uniform='bench-cols')
        for i in range(max(self._dimensions[1], self.state['rows'])):
            self.rowconfigure(i, weight=round(self.state['row_weights'][i]*100) if i < self.state['rows'] else 0, minsize=0, uniform='bench-rows')
        self._dimensions = self.state['columns'], self.state['rows']
        for record in self.state['panes']:
            pane = self.panes.get(record['id'])
            if pane and record['id'] not in self.popped:
                pane.grid(row=record['row'], column=record['col'], rowspan=record['rowspan'],
                          columnspan=record['colspan'], sticky='nsew', padx=3, pady=3)
        wanted = set()
        thickness = 8 if self.arranging else 4
        for axis, weights in (('col', self.state['column_weights']), ('row', self.state['row_weights'])):
            for i in range(len(weights)-1):
                key = (axis, i); wanted.add(key)
                if key not in self.sashes:
                    sash = tk.Frame(self, background=theme.GRID, cursor='sb_h_double_arrow' if axis == 'col' else 'sb_v_double_arrow')
                    sash.bind('<B1-Motion>', lambda event, axis=axis, i=i: self._weight(axis, i, event))
                    self.sashes[key] = sash
                # The sash is the only sizing gesture, so arranging makes it fat
                # and impossible to miss; live mode keeps it out of the way.
                self.sashes[key].configure(background=theme.ACCENT if self.arranging else theme.GRID)
                fraction = sum(weights[:i+1])/sum(weights)
                if axis == 'col': self.sashes[key].place(relx=fraction, rely=0, width=thickness, relheight=1, anchor='n')
                else: self.sashes[key].place(relx=0, rely=fraction, height=thickness, relwidth=1, anchor='w')
                self.sashes[key].lift()
        for key in set(self.sashes)-wanted: self.sashes.pop(key).destroy()
        self._outline_cells()
        for sash in self.sashes.values(): sash.lift()

    def _outline_cells(self):
        """In arrange mode, draw a drop target over every empty cell."""
        wanted = set()
        if self.arranging:
            occupied = layout.occupied(self.state)
            widths = self.state['column_weights']; heights = self.state['row_weights']
            x0 = [sum(widths[:i])/sum(widths) for i in range(len(widths)+1)]
            y0 = [sum(heights[:i])/sum(heights) for i in range(len(heights)+1)]
            for row in range(self.state['rows']):
                for col in range(self.state['columns']):
                    if (row, col) in occupied: continue
                    key = (row, col); wanted.add(key)
                    if key not in self.targets:
                        self.targets[key] = tk.Label(self, text='＋', relief='groove', borderwidth=2,
                            background=theme.PANEL, foreground=theme.DIM, anchor='center')
                    self.targets[key].place(relx=x0[col], rely=y0[row],
                                            relwidth=x0[col+1]-x0[col], relheight=y0[row+1]-y0[row])
        for key in set(self.targets)-wanted: self.targets.pop(key).destroy()

    def _geometry_key(self, sid):
        return f'dctl.{self.session.instance}.device.{sid}'

    def pop_out(self, sid):
        pane = self.panes.get(sid)
        if pane is None or sid in self.popped: return
        # A floating window shows its body even while the grid is arranged.
        if self.arranging: pane.collapse(False)
        pane.grid_forget()
        root_tag = str(self.winfo_toplevel())
        def isolate(widget):
            widget.bindtags(tuple(tag for tag in widget.bindtags() if tag != root_tag))
            for child in widget.winfo_children(): isolate(child)
        isolate(pane)
        # Tk promotes this same Frame into a managed toplevel, preserving every
        # canvas item, renderer, PhotoImage, and transport reference.
        pane.tk.call('wm', 'manage', str(pane))
        pane.wm_title(sid)
        pane.wm_geometry(self.state['popped'].get(sid, {}).get('geometry', '640x520'))
        restore_window_geometry(pane, self._geometry_key(sid))
        if not hasattr(pane, '_return_command'):
            pane._return_command = pane.register(lambda: self.return_to_grid(sid))
        pane.tk.call('wm', 'protocol', str(pane), 'WM_DELETE_WINDOW', pane._return_command)
        self.popped[sid] = pane
        self.state['popped'][sid] = {'geometry': pane.wm_geometry()}

    def return_to_grid(self, sid):
        pane = self.popped.pop(sid, None)
        if pane is None: return
        save_window_geometry(pane, self._geometry_key(sid))
        pane.tk.call('wm', 'forget', str(pane))
        root_tag = str(self.winfo_toplevel())
        def restore_tags(widget):
            tags = [tag for tag in widget.bindtags() if tag not in (str(pane), root_tag) or tag == str(widget)]
            if root_tag not in tags: tags.insert(max(1, len(tags)-1), root_tag)
            widget.bindtags(tuple(tags))
            for child in widget.winfo_children(): restore_tags(child)
        restore_tags(pane)
        self.state['popped'].pop(sid, None)
        self.reconcile()

    def _discard_pane(self, sid):
        pane = self.panes.pop(sid)
        if sid in self.popped:
            self.state['popped'][sid] = {'geometry': pane.wm_geometry()}
            save_window_geometry(pane, self._geometry_key(sid))
            self.popped.pop(sid)
        pane.destroy(); pane.stream.close()

    def freeze_all(self, enabled, synchronized=True):
        self.freeze_enabled = enabled
        self.freeze_synchronized = synchronized
        self.freeze_token = (self.session.generation_token, self.session.reset_epoch)
        self.freeze_targets = {}
        if enabled:
            panes = [pane for pane in self.panes.values() if pane.winfo_ismapped()]
            capture, common = synchronized_capture([pane.stream for pane in panes])
            self.freeze_capture = capture
            self.freeze_message = ('shared capture ' if common else 'no shared capture; holding ')+str(capture)
            for pane in panes: self.freeze_targets[pane.stream.selected] = self._freeze_target(pane)
        else: self.freeze_message = 'live'
        for pane in self.panes.values(): pane.last_paint = -1e9

    def _freeze_target(self, pane):
        """The capture a pane holds under freeze, including one opened later."""
        if self.freeze_synchronized: return self.freeze_capture
        latest = pane.stream.latest
        return latest.capture_id if latest else None

    def set_visible(self, visible):
        if visible != self.visible:
            self.visible = visible; self.reconcile()

    def paint(self, report, budget_s=.008):
        # Arranging shows structure, not data: panes stay collapsed at their
        # last frame while the drain keeps running, exactly as under freeze.
        if self.arranging: return
        if not self.visible and not self.popped: return
        hold, invalidated = self.freeze_enabled, False
        if hold:
            generation, epoch = self.session.generation_token, self.session.reset_epoch
            token_generation, token_epoch = self.freeze_token
            invalidated = generation != token_generation or (token_epoch is not None and epoch != token_epoch)
            if invalidated:
                self.freeze_targets = {}; self.freeze_message = 'capture invalidated by reset/profile change'
            elif token_epoch is None and epoch is not None:
                # The freeze was armed before any record existed; the first
                # epoch arriving is not a reset. Remember it so a real reset
                # still invalidates the held capture.
                self.freeze_token = (generation, epoch)
        start = time.perf_counter()
        # A pane showing its text body decodes nothing, so it neither takes a
        # share of the video budget nor needs the video rate.
        images = sum(_wants_video(p) for p in self.panes.values())
        # Only mapped panes contend for the budget, so the skipped count is
        # panes that would otherwise have painted, not hidden ones.
        panes = [(sid, pane) for sid, pane in self.panes.items() if pane.winfo_ismapped()]
        for i, (sid, pane) in enumerate(panes):
            if time.perf_counter()-start >= budget_s:
                self.skipped_paints += len(panes)-i; break
            capture = self.freeze_targets.get(sid)
            if hold and not invalidated and sid not in self.freeze_targets:
                capture = self.freeze_targets[sid] = self._freeze_target(pane)
            rate = max(5., VIDEO_HZ/max(1, images)) if _wants_video(pane) else TEXT_HZ
            stat = stream_stats(report, pane.stream)
            self.paints += pane.paint(stat, rate, hold=hold, capture=capture)


def device_rows(manifest):
    """Use the editor's component tree, degrading to a flat manifest list."""
    devices = manifest.get('sensors', {})
    try:
        from dsim.sensors_panel import component_tree
        profile = manifest['profile']
        rendered = component_tree(SimpleNamespace(data=profile))
        nodes = {n['id']: n for n in profile['mounts'] + profile['sensors']}
        rows = [('body', '', 'body')]
        for line in rendered.splitlines()[1:]:
            label = line.split('-- ', 1)[1]
            sid = label.split(' (', 1)[0]
            node = nodes[sid]
            if sid in devices or sid in {n['id'] for n in profile['mounts']}:
                rows.append((sid, node['parent'], label))
        if not set(devices).issubset({r[0] for r in rows}): raise ValueError('incomplete tree')
        return rows
    except (KeyError, TypeError, ValueError, RecursionError):
        return [(sid, '', f"{sid} ({entry['type']})") for sid, entry in devices.items()]


class DevicesView:
    def __init__(self, notebook, session, initial=(), layout_name=None, status_values=lambda: {}):
        self.page = ttk.Frame(notebook)
        self.page.columnconfigure(1, weight=1); self.page.rowconfigure(0, weight=1)
        self.session = session; self.token = None
        self.layout_name = layout_name; self.layout_key = None; self.initial = list(initial)
        self.tree = ttk.Treeview(self.page, columns=('checked', 'health'), displaycolumns=('checked', 'health'))
        self.tree.heading('#0', text='Device'); self.tree.heading('checked', text='Open')
        self.tree.heading('health', text='Health')
        for col in ('checked', 'health'): self.tree.column(col, width=55, stretch=False)
        self.tree.grid(row=0, column=0, sticky='nsew')
        self.tree.tag_configure('absent', foreground=theme.DIM)
        self.grid = DeviceGrid(self.page, session, self._grid_changed)
        self.grid.wanted = list(dict.fromkeys(initial))
        self.status_values = status_values
        self.grid.export_enabled = lambda: export_directory(self.status_values()) is not None
        self.grid.export_action = self.export
        controls = ttk.Frame(self.page); controls.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.freeze = tk.BooleanVar(value=False); self.sync = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text='Freeze all', variable=self.freeze, command=self._freeze).pack(side='left')
        ttk.Checkbutton(controls, text='Sync captures', variable=self.sync, command=self._freeze).pack(side='left')
        self.arrange = tk.BooleanVar(value=False)
        self.arrange_button = ttk.Checkbutton(controls, text='Arrange', variable=self.arrange, command=self._arrange_toggled)
        self.arrange_button.pack(side='left')
        # Structural edits: + appends at the edge; − removes the edge row or
        # column and only when it is empty -- a structural edit never moves a
        # pane, so the operator drags it out first.
        self.structure = []
        for text, method, axis, growable in (('+ Row', 'grow', 'rows', True), ('− Row', 'shrink', 'rows', False),
                                             ('+ Col', 'grow', 'columns', True), ('− Col', 'shrink', 'columns', False)):
            button = ttk.Button(controls, text=text, command=lambda method=method, axis=axis:
                                getattr(self.grid, method)(axis))
            self.structure.append((button, axis, growable))
        self.export_buttons = []
        for kind in ('png', 'json', 'csv'):
            button = ttk.Button(controls, text='Snapshot PNG' if kind == 'png' else 'Dump '+kind.upper(),
                                command=lambda kind=kind: self.export_selected(kind))
            button.pack(side='left'); self.export_buttons.append(button)
        self.export_message = ''
        self.grid.grid(row=0, column=1, sticky='nsew')
        self.footer = ttk.Label(self.page); self.footer.grid(row=2, column=0, columnspan=2, sticky='ew')
        # The tree's check mark is an indicator, not a control: a device joins
        # the bench by being dragged onto a cell and leaves through its pane's ×.
        self._press_state = None; self._preview = None
        self.tree.bind('<ButtonPress-1>', self._press)
        self.tree.bind('<B1-Motion>', self._drag)
        self.tree.bind('<ButtonRelease-1>', self._release)

    def _press(self, event):
        """Only device rows drag: mounts and headings are not dragged anywhere."""
        sid = self.tree.identify_row(event.y)
        self._press_state = (sid, event.x_root, event.y_root) \
            if sid and (sid in device_entries(self.session) or sid in self.grid.wanted) else None
        self._preview = None

    def _drag(self, event):
        if not self._press_state: return
        sid, x0, y0 = self._press_state
        if abs(event.x_root-x0)+abs(event.y_root-y0) < 6: return
        if self._preview is None:
            self._preview = tk.Toplevel(self.page.winfo_toplevel())
            self._preview.overrideredirect(True)
            tk.Label(self._preview, text=sid, background=theme.PANEL, foreground=theme.TEXT,
                     relief='groove', borderwidth=1).pack()
        self._preview.wm_geometry(f'+{event.x_root+12}+{event.y_root+12}')

    def _release(self, event):
        state, self._press_state = self._press_state, None
        preview, self._preview = self._preview, None
        if preview is not None: preview.destroy()
        if state is None: return
        if preview is None:
            # A plain click is not a control -- except on an absent device,
            # whose grey row is the only handle left for dropping its
            # reservation so the device is forgotten instead of returning.
            if state[0] in self.grid.wanted and state[0] not in device_entries(self.session):
                self.grid.toggle(state[0])
            return
        grid = self.grid
        x, y = event.x_root-grid.winfo_rootx(), event.y_root-grid.winfo_rooty()
        if 0 <= x < grid.winfo_width() and 0 <= y < grid.winfo_height():
            col, row = grid.grid_location(x, y)
            if row >= 0 and col >= 0: grid.drop_device(state[0], row, col)

    def _arrange_toggled(self):
        arranging = self.arrange.get()
        self.grid.set_arranging(arranging)
        anchor = self.arrange_button
        for button, axis, growable in self.structure:
            if arranging:
                button.pack(side='left', after=anchor)
            else: button.pack_forget()
            anchor = button
        self._refresh_structure()

    def _refresh_structure(self):
        if not self.structure: return  # the buttons do not exist yet
        for button, axis, growable in self.structure:
            button.configure(state='normal' if growable or self.grid.edge_is_empty(axis) else 'disabled')

    def _grid_changed(self):
        self.update_tree()
        self._refresh_structure()

    def update_tree(self):
        report = self.session.report()
        for sid in self.tree.get_children():
            if self.tree.item(sid, 'tags') == ('absent',) and sid not in self.grid.wanted: self.tree.delete(sid)
        entries = device_entries(self.session)
        for sid in entries:
            if self.tree.exists(sid):
                self.tree.set(sid, 'checked', '☑' if sid in self.grid.wanted else '☐')
                self.tree.set(sid, 'health', DOTS[self._tree_grade(report, entries, sid)])

    @staticmethod
    def _tree_grade(report, entries, sid):
        """A pair reports its worst member; the session report has no stereo rows."""
        members = entries.get(sid, {}).get('members')
        if not members: return report['devices'].get(sid, {'grade': 'unknown'})['grade']
        rank = {'bad': 0, 'warn': 1, 'ok': 2}
        known = [g for g in (report['devices'].get(m, {'grade': 'unknown'})['grade'] for m in members) if g in rank]
        return min(known, key=rank.get) if known else 'unknown'

    def _freeze(self):
        self.grid.freeze_all(self.freeze.get(), self.sync.get())

    def export_selected(self, kind):
        selection = self.tree.selection()
        if selection: self.export(selection[0], kind)
        else: self.export_message = 'Select a device to export'

    def export(self, sid, kind):
        pane = self.grid.panes.get(sid)
        if pane is None: self.export_message = 'Open a device pane first'; return
        try:
            directory = export_directory(self.status_values())
            if kind == 'png':
                path = snapshot_png(directory, pane); self.export_message = f'Saved {path.name}'
            else:
                samples = [s for s, _ in pane.stream.history.values() if pane.display_capture is not None and s.capture_id <= pane.display_capture]
                path, count, omitted = dump_samples(directory, sid, samples, format=kind)
                self.export_message = f'Saved {path.name}: {count} samples, {omitted} omitted'
        except (OSError, ValueError) as exc: self.export_message = str(exc)

    def save(self):
        for sid, pane in self.grid.popped.items(): save_window_geometry(pane, self.grid._geometry_key(sid))
        if self.layout_key is not None:
            save_state('device_layouts', self.layout_key, self.grid.serialize())

    def update(self, visible):
        manifest = self.session.manifest
        key = self.layout_name or layout.layout_key(manifest, self.session.instance)
        if manifest and self.layout_key != key:
            self.save()
            _, record = layout.resolve_layout(load_state, manifest, self.session.instance, self.layout_name)
            if self.layout_key is None and self.initial:
                record = layout.auto_layout(self.initial)
            self.layout_key = key
            record.setdefault('profile_digest', manifest.get('profile_digest'))
            self.grid.restore(record)
            # A restored record can empty or fill the edge row or column while
            # the structure buttons' enablement is showing.
            self._refresh_structure()
        self.grid.set_visible(visible)
        if self.token != self.session.generation_token:
            self.token = self.session.generation_token
            self.tree.delete(*self.tree.get_children())
            for sid, parent, label in device_rows(self.session.manifest):
                self.tree.insert(parent, 'end', iid=sid, text=label, open=True)
            for sid in stereo_groups(self.session.manifest):
                self.tree.insert('', 'end', iid=sid, text=sid+' (stereo pair)')
            self.grid.reconcile()
        # A wanted device that the manifest no longer lists keeps its grey row
        # for as long as its reservation stands, whenever that started.
        for sid in self.grid.wanted:
            if sid not in device_entries(self.session) and not self.tree.exists(sid):
                self.tree.insert('', 'end', iid=sid, text=sid+' (absent)', values=('☑', '✕'), tags=('absent',))
        for button in self.export_buttons: button.configure(state='normal' if self.grid.export_enabled() else 'disabled')
        if not visible and not self.grid.popped: return
        report = self.session.report()
        self.grid.paint(report)
        self.update_tree()
        gaps = sum(d['gaps'] for d in report['devices'].values())
        revision = ' · profile revision changed' if self.grid.state.get('profile_digest') != manifest.get('profile_digest') else ''
        self.footer.configure(text=f"{self.grid.freeze_message} · {self.export_message} · layout {self.layout_key}{revision} · cache {report['cache_bytes']/1048576:.2f}/{report['cache_limit']/1048576:g} MiB · "
            f"overruns {report['overruns']} · gaps {gaps} · cache drops {report['cache_drops']} · "
            f"skipped paints {self.grid.skipped_paints} · paint {self.grid.paints/max(.001, time.monotonic()-self.grid.started):.1f} Hz (wall)")

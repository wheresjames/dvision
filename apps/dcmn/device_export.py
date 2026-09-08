"""Bounded, lossless device sample exports under the simulator's report root."""
import base64
import csv
from dataclasses import fields
import io
import json
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageTk, PngImagePlugin
from dcmn.sensors import Sample

MAX_EXPORT_BYTES = 8*1024*1024
MAX_EXPORT_SAMPLES = 1000
SCHEMA = 'dvision2.device-samples.v1'


def export_directory(values):
    value = values.get('sim.report_dir')
    return Path(value)/'dctl' if isinstance(value, str) and value.strip() else None


def _array(value):
    if value is None: return None
    contiguous = np.ascontiguousarray(value)
    return dict(dtype=contiguous.dtype.str, shape=list(value.shape), data=base64.b64encode(contiguous.tobytes()).decode('ascii'))


def _decode_array(value):
    if value is None: return None
    dtype = np.dtype(value['dtype'])
    if dtype.hasobject: raise ValueError('object arrays are not supported')
    return np.frombuffer(base64.b64decode(value['data'], validate=True), dtype).reshape(value['shape']).copy()


def encode_sample(sample):
    record = {field.name: getattr(sample, field.name) for field in fields(Sample) if field.name not in ('fields', 'image')}
    record['fields'] = None if sample.fields is None else {key: _array(value) for key, value in sample.fields.items()}
    record['image'] = _array(sample.image)
    return record


def decode_sample(record):
    record = dict(record)
    record['fields'] = None if record['fields'] is None else {key: _decode_array(value) for key, value in record['fields'].items()}
    record['image'] = _decode_array(record['image'])
    return Sample(**record)


def _csv_row(record):
    stream = io.StringIO(newline='')
    csv.writer(stream).writerow([json.dumps(record[key], separators=(',', ':'), allow_nan=False) for key in record])
    return stream.getvalue()


def dump_samples(directory, sid, samples, *, format='json', max_bytes=MAX_EXPORT_BYTES, max_samples=MAX_EXPORT_SAMPLES):
    """Newest samples that fit both ceilings; result includes omitted count."""
    if directory is None: raise ValueError('sim.report_dir is unavailable')
    if format not in ('json', 'csv'): raise ValueError('unsupported sample format')
    selected, count, size = [], 0, 256
    samples = list(samples)
    for sample in reversed(samples):
        if count >= max_samples: break
        record = encode_sample(sample)
        encoded = json.dumps(record, separators=(',', ':'), allow_nan=False) if format == 'json' else _csv_row(record)
        if size+len(encoded.encode('utf-8'))+2 > max_bytes: break
        selected.append((record, encoded)); size += len(encoded.encode('utf-8'))+2; count += 1
    if not selected: raise ValueError('no samples fit the export limit')
    selected.reverse()
    omitted = len(samples)-count
    if format == 'json':
        text = json.dumps(dict(schema=SCHEMA, omitted=omitted), separators=(',', ':'))[:-1]+',"samples":['+','.join(text for _, text in selected)+']}'
    else:
        output = io.StringIO(newline=''); csv.writer(output).writerow(selected[0][0].keys())
        text = output.getvalue()+''.join(text for _, text in selected)
    if len(text.encode('utf-8')) > max_bytes: raise ValueError('export limit is too small')
    path = _new_path(directory, sid, format)
    with path.open('x', encoding='utf-8', newline='') as stream: stream.write(text)
    return path, count, omitted


def load_samples(path):
    path = Path(path)
    if path.suffix == '.json':
        data = json.loads(path.read_text(encoding='utf-8'))
        if data['schema'] != SCHEMA: raise ValueError('unsupported sample schema')
        records = data['samples']
    else:
        with path.open(encoding='utf-8', newline='') as stream:
            records = [{key: json.loads(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    return [decode_sample(record) for record in records]


def _new_path(directory, sid, suffix):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    name = re.sub('[^a-zA-Z0-9_-]', '_', sid)[:80]
    return directory/f'{name}-{uuid.uuid4().hex[:12]}.{suffix}'


def snapshot_png(directory, pane):
    """Rasterize the displayed canvas without a screen grab or Ghostscript.

    The pane's displayed readout is included, so a frozen capture is labelled
    with its actual time. Occluding another window cannot contaminate a snapshot.
    A pane switched to its text body has no graphic to rasterize and is refused.
    """
    if directory is None: raise ValueError('sim.report_dir is unavailable')
    if pane.display_capture is None: raise ValueError('no displayed sample')
    # The pane is showing its readout instead of its graphic, so there is no
    # drawn canvas to rasterize -- only whatever it last drew at some other
    # size, which is not what the operator is looking at.
    if pane.show_readout: raise ValueError('the pane is showing its readout, not a graphic')
    renderer = pane.renderer
    canvas = getattr(renderer, 'canvas', None)
    width = max(320, canvas.winfo_width() if canvas is not None else pane.winfo_width())
    height = max(120, canvas.winfo_height() if canvas is not None else 320)
    readout = str(pane.readout['text'])
    lines = [readout[i:i+max(30, width//7)] for i in range(0, len(readout), max(30, width//7))]
    image = Image.new('RGB', (width, height+30+16*len(lines)), '#010409')
    draw = ImageDraw.Draw(image)
    if canvas is not None:
        for item in canvas.find_all():
            kind, coords = canvas.type(item), canvas.coords(item)
            fill = canvas.itemcget(item, 'fill') if kind != 'image' else ''
            if kind == 'image' and getattr(renderer, 'photo', None) is not None:
                image.paste(ImageTk.getimage(renderer.photo).convert('RGB'), (round(coords[0]), round(coords[1])))
            elif kind == 'line': draw.line(list(zip(coords[::2], coords[1::2])), fill=fill or 'white', width=1)
            elif kind in ('oval', 'rectangle'):
                outline = canvas.itemcget(item, 'outline')
                method = draw.ellipse if kind == 'oval' else draw.rectangle
                method(coords, fill=fill or None, outline=outline or None)
            elif kind == 'polygon': draw.polygon(list(zip(coords[::2], coords[1::2])), fill=fill or None)
            elif kind == 'text':
                draw.text(tuple(coords[:2]), canvas.itemcget(item, 'text'), fill=fill or 'white')
    else:
        text = renderer.text.get('1.0', 'end') if hasattr(renderer, 'text') else readout
        draw.multiline_text((5, 5), text, fill='white')
    draw.text((5, height+2), pane.stream.selected, fill='white')
    draw.multiline_text((5, height+20), '\n'.join(lines), fill='white')
    metadata = PngImagePlugin.PngInfo(); metadata.add_text('capture_id', str(pane.display_capture))
    metadata.add_text('sensor_id', pane.stream.selected)
    path = _new_path(directory, pane.stream.selected, 'png')
    image.save(path, pnginfo=metadata)
    return path

"""The Sensors tab: the vehicle's hardware, edited as the tree its links imply.

The profile is stored flat -- one ``mounts`` array, one ``sensors`` array, a
``parent`` on every component -- because that is what is easy to validate,
canonicalise and diff. It is *edited* as a tree, because that is what the
thing actually is. Reparenting changes one field.

Nothing here touches the running simulation until Apply. Apply validates a
complete immutable profile first and hands it to the simulator whole, so a
draft that does not resolve, or a generation that cannot be constructed,
leaves the vehicle flying on exactly the profile it already had.
"""

import copy
import math
import tkinter as tk
from tkinter import filedialog, ttk

from dcmn import theme
from dsim.add_menu import AddMenu
from dsim.scroll import Scrollable
from dsim.profiles import (ARRAY_TYPES, CHOICE_FIELDS, INTEGER_FIELDS,
                           MOUNT_TYPES, PTZ_AXES, POSE_KEYS, SENSOR_TYPES,
                           DroneProfile, model_fields, new_component,
                           stereo_pair, unique_id)
from dsim.transforms import euler_angles, mount_chain

#: One line under each entry of the Add menu: what the type is, briefly. A
#: type without an entry here just shows its name.
ADD_DESCRIPTIONS = {
    'camera.rgb': "a rendered RGB pinhole camera",
    'lidar.scan2d': "a polar sweep of ranges around the vehicle",
    'lidar.range_image': "a rectangular grid of ranges",
    'range.infrared': "a narrow proximity beam",
    'range.ultrasonic': "a wide, short-range sonar cone",
    'range.laser': "a narrow long-range beam",
    'position.gnss': "satellite position, velocity and fix quality",
    'motion.imu': "angular rate and specific force",
    'altimeter.barometric': "pressure altitude from the air",
    'heading.magnetometer': "a measured compass heading",
    'environment.temperature': "air temperature at the vehicle",
    'mount.fixed': "a named transform node",
    'mount.ptz': "a transform node that pans and tilts",
    'rig.fixed': "a rigid rig, like a stereo bar",
}

class SensorsPanel:
    def __init__(self, parent, sim):
        self.sim = sim
        self.path = None
        self.draft = sim.profile.data
        self.selected = None
        self.resolved = None
        self._editor_key = None
        self._vars = {}
        self._errors = {}
        self._loading = False

        self.page = ttk.Frame(parent, padding=10)
        self.page.columnconfigure(1, weight=1)
        self.page.rowconfigure(2, weight=1)
        self._build_toolbar()

        self.info = tk.StringVar()
        ttk.Label(self.page, textvariable=self.info, wraplength=760,
                  style="Dim.TLabel").grid(row=1, column=0, columnspan=2,
                                           sticky="ew", pady=(6, 6))

        self.tree = ttk.Treeview(self.page, columns=("badges",), show="tree headings",
                                 height=14, selectmode="browse")
        self.tree.heading("#0", text="component")
        self.tree.heading("badges", text="type / rate")
        self.tree.column("#0", width=210, stretch=False)
        self.tree.column("badges", width=250, stretch=True)
        self.tree.grid(row=2, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._selection_changed)
        self.tree.bind("<ButtonPress-1>", self._drag_start, add=True)
        self.tree.bind("<ButtonRelease-1>", self._drag_end, add=True)
        self._dragged = None
        for tag, colour in (("invalid", theme.DANGER), ("primary", theme.OK),
                            ("disabled", theme.DIM)):
            self.tree.tag_configure(tag, foreground=colour)

        right = ttk.Frame(self.page, padding=(12, 0, 0, 0))
        right.grid(row=2, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        # A component's settings run well past a window's height -- pose,
        # PTZ state and a model section one field each -- so the editor lives
        # in a scrolling viewport and the resolved readout stays pinned below.
        self._scroll = Scrollable(right)
        self._scroll.outer.grid(row=0, column=0, sticky="nsew")
        self.editor = self._scroll.inner
        self.editor.columnconfigure(1, weight=1)
        self.resolved_var = tk.StringVar()
        ttk.Label(right, textvariable=self.resolved_var, justify="left",
                  font=("TkFixedFont", 8), style="Dim.TLabel").grid(
                      row=1, column=0, sticky="ew", pady=(8, 0))
        self.revert()

    # -- toolbar ----------------------------------------------------------

    def _build_toolbar(self):
        bar = ttk.Frame(self.page)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        add = ttk.Menubutton(bar, text="Add ▾")
        self.add_button = add
        self.add_menu = AddMenu(add, [
            (kind, ADD_DESCRIPTIONS.get(kind, ""), "sensor")
            for kind in SENSOR_TYPES] + [
            (kind, ADD_DESCRIPTIONS.get(kind, ""), "mount")
            for kind in MOUNT_TYPES], self.add)
        add.bind("<Button-1>", lambda _event: self.add_menu.toggle())
        add.pack(side="left")
        for label, command in (("Duplicate", self.duplicate), ("Remove", self.remove),
                               ("Add stereo pair", self.add_stereo),
                               ("Make primary", self.make_primary),
                               ("Load", self.load), ("Save As", self.save),
                               ("Revert", self.revert)):
            ttk.Button(bar, text=label, command=command).pack(side="left")
        self.apply_button = ttk.Button(bar, text="Apply", command=self.apply,
                                       style="Accent.TButton")
        self.apply_button.pack(side="left", padx=(8, 0))

    # -- draft lifecycle --------------------------------------------------

    def components(self, draft=None):
        draft = self.draft if draft is None else draft
        return [*draft.get("mounts", []), *draft.get("sensors", [])]

    def component(self, component_id):
        return next((n for n in self.components() if n.get("id") == component_id), None)

    def revert(self):
        """Throw the draft away and start again from the running profile."""
        self.draft = self.sim.profile.data
        self.path = None
        self.selected = self.draft["primary_camera"]
        self._editor_key = None
        self.refresh_all()

    def refresh_all(self):
        """Validate the draft, then redraw everything that depends on it."""
        self.error = None
        try:
            self.resolved = DroneProfile.parse(copy.deepcopy(self.draft))
        except (ValueError, TypeError, KeyError) as exc:
            self.resolved, self.error = None, str(exc)
        self._refresh_tree()
        self._refresh_editor()
        self._refresh_status()
        self.refresh()

    def _refresh_status(self):
        active = self.sim.profile.digest
        state = ("invalid" if self.resolved is None else
                 "active" if self.resolved.digest == active else "modified")
        parts = [self.path or "(unsaved)", self.draft.get("name", "?"), state]
        if self.resolved is not None:
            plan = self.resolved.plan
            parts += [f"{len(plan['cameras'])} camera(s)",
                      f"{len(plan['arrays'])} array ring(s)",
                      f"{plan['render_hz']:.0f} render Hz",
                      f"{plan['compact_hz']:.0f} compact rec/s",
                      f"{self.resolved.memory_bytes / 1048576:.1f} MiB total"]
        else:
            parts.append(self.error)
        parts.append(f"generation {getattr(self.sim.sensors, 'generation', 0)}")
        self.info.set("  |  ".join(str(p) for p in parts))
        for key, label in self._errors.items():
            label.configure(text=self._field_error(key))

    def _field_error(self, key):
        """The validation message, shown beside the field it names."""
        if self.error is None:
            return ""
        head = self.error.split(":", 1)[0]
        return self.error if head.endswith(key) or head == key else ""

    # -- tree -------------------------------------------------------------

    def _refresh_tree(self):
        opened = {i for i in self._walk_tree() if self.tree.item(i, "open")}
        self.tree.delete(*self.tree.get_children())
        self.tree.insert("", "end", iid="body", text="body", open=True,
                         values=("vehicle body",))
        children = {}
        for node in self.components():
            children.setdefault(node.get("parent"), []).append(node)
        placed = self._place(children, "body", opened)
        # Anything the parent links orphan is still shown, so a draft with a
        # broken parent can be repaired rather than disappearing.
        for node in self.components():
            if node.get("id") not in placed:
                self.tree.insert("body", "end", iid=node.get("id"),
                                 text=str(node.get("id")), tags=("invalid",),
                                 values=(f"unreachable parent {node.get('parent')!r}",))
        if self.selected and self.tree.exists(self.selected):
            self.tree.selection_set(self.selected)

    def _place(self, children, parent, opened, seen=None):
        seen = set() if seen is None else seen
        for node in children.get(parent, []):
            node_id = node.get("id")
            if not isinstance(node_id, str) or node_id in seen:
                continue
            seen.add(node_id)
            self.tree.insert(parent, "end", iid=node_id, text=node_id,
                             open=node_id in opened or True,
                             tags=self._tags(node), values=(self._badges(node),))
            self._place(children, node_id, opened, seen)
        return seen

    def _walk_tree(self, parent=""):
        for item in self.tree.get_children(parent):
            yield item
            yield from self._walk_tree(item)

    def _tags(self, node):
        if node.get("id") == self.draft.get("primary_camera"):
            return ("primary",)
        return () if node.get("enabled", True) else ("disabled",)

    def _badges(self, node):
        badges = [node.get("type", "?")]
        if node.get("id") == self.draft.get("primary_camera"):
            badges.append("primary")
        if node.get("sync_group"):
            badges.append(f"sync: {node['sync_group']}")
        if "rate_hz" in node:
            try:
                badges.append(f"{float(node['rate_hz']):g} Hz")
            except (TypeError, ValueError):
                badges.append(f"{node['rate_hz']} Hz (invalid)")
        if node.get("enabled") is False:
            badges.append("disabled")
        if node.get("type") in ARRAY_TYPES:
            badges.append("array ring")
        return ", ".join(badges)

    def _selection_changed(self, _event=None):
        selection = self.tree.selection()
        self.selected = selection[0] if selection else None
        self._refresh_editor()
        self._refresh_status()

    def _drag_start(self, event):
        self._dragged = self.tree.identify_row(event.y)

    def _drag_end(self, event):
        source, self._dragged = self._dragged, None
        target = self.tree.identify_row(event.y)
        if source and target and source != target:
            self.reparent(source, target)

    def reparent(self, component_id, parent):
        node = self.component(component_id)
        if node is not None and parent in self._parent_choices(component_id):
            node['parent'] = parent
            self.selected = component_id
            self._editor_key = None
            self.refresh_all()

    # -- editor -----------------------------------------------------------

    def _refresh_editor(self):
        node = self.component(self.selected)
        key = (self.selected, node.get("type") if node else None,
               tuple(sorted((node or {}).get("model", {}))))
        if key != self._editor_key:
            self._editor_key = key
            self._build_editor(node)
        self._refresh_resolved(node)

    def _build_editor(self, node):
        for child in self.editor.winfo_children():
            child.destroy()
        self._vars, self._errors = {}, {}
        row = self._section(0, "profile")
        row = self._field(row, "name", self.draft.get("name", ""), ("profile", "name"))
        row = self._field(row, "physics_hz", self.draft.get("physics_hz", 300.),
                          ("profile", "physics_hz"))
        for key, default in (('retention_s', 1.), ('memory_limit_mib', 256.)):
            row = self._field(row, key, self.draft.get('transport', {}).get(key, default),
                              ('transport', key))
        if node is None:
            ttk.Label(self.editor, text="Select a component.",
                      style="Dim.TLabel").grid(row=row, column=0, sticky="w")
            return
        row = self._section(row, f"{node['id']} ({node['type']})")
        row = self._field(row, "id", node["id"], ("id",))
        row = self._field(row, "parent", node["parent"], ("parent",),
                          choices=self._parent_choices(node["id"]))
        if "enabled" in node or node["type"] not in MOUNT_TYPES:
            row = self._field(row, "enabled", bool(node.get("enabled", True)),
                              ("enabled",), choices=("true", "false"))
            row = self._field(row, "rate_hz", node.get("rate_hz", 30.), ("rate_hz",))
            row = self._field(row, "sync_group", node.get("sync_group") or "",
                              ("sync_group",))
        row = self._section(row, "pose (parent-relative)")
        for key in POSE_KEYS:
            row = self._field(row, key, node.get("pose_parent", {}).get(key, 0.),
                              ("pose_parent", key))
        if node["type"] == "mount.ptz":
            row = self._section(row, "ptz state")
            for axis in PTZ_AXES:
                row = self._field(row, axis, node.get("state", {}).get(axis, 0.),
                                  ("state", axis))
            row = self._section(row, "ptz limits (minimum, maximum)")
            for axis in PTZ_AXES:
                bounds = node.get("limits", {}).get(axis, [-180., 180.])
                row = self._field(row, axis, ", ".join(map(str, bounds)),
                                  ("limits", axis))
        fields = model_fields(node["type"], node.get("model"))
        if fields:
            row = self._section(row, "model")
            model = node.setdefault("model", {})
            if node['type'] in ('camera.rgb', 'lidar.range_image') and 'fx_px' in model:
                try:
                    fov = _fov(float(model['width_px']), float(model['fx_px']))
                except (KeyError, TypeError, ValueError, ZeroDivisionError):
                    fov = ''
                row = self._field(row, 'fov_h_deg', fov,
                                  ('model', 'fov_h_deg'))
            for key in fields:
                row = self._field(row, key, model.get(key, ""), ("model", key),
                                  choices=CHOICE_FIELDS.get(key))

    def _section(self, row, title):
        ttk.Label(self.editor, text=title, style="Brand.TLabel").grid(
            row=row, column=0, columnspan=3, sticky="w", pady=(8, 2))
        return row + 1

    def _field(self, row, label, value, path, choices=None):
        ttk.Label(self.editor, text=label).grid(row=row, column=0, sticky="w")
        variable = tk.StringVar(value=_as_text(value))
        if choices:
            widget = ttk.Combobox(self.editor, textvariable=variable, width=16,
                                  state="readonly", values=list(choices))
        else:
            widget = ttk.Entry(self.editor, textvariable=variable, width=18)
        widget.grid(row=row, column=1, sticky="ew", padx=(6, 6))
        # The wheel over a field scrolls the settings, and never spins a
        # combobox -- over this form that would silently edit the draft.
        self._scroll.claim_wheel(widget)
        error = ttk.Label(self.editor, text="", foreground=theme.DANGER)
        error.grid(row=row, column=2, sticky="w")
        self._errors[path[-1]] = error
        self._vars[path] = variable
        variable.trace_add("write", lambda *_: self._commit(path, variable))
        return row + 1

    def _parent_choices(self, component_id):
        """Everything a component may hang from: body, or a mount that is not
        itself underneath it. Offering a descendant would offer a cycle."""
        banned, changed = {component_id}, True
        while changed:
            changed = False
            for node in self.components():
                if node.get("parent") in banned and node.get("id") not in banned:
                    banned.add(node.get("id"))
                    changed = True
        return ["body", *(n["id"] for n in self.draft.get("mounts", [])
                          if n.get("id") not in banned)]

    def _commit(self, path, variable):
        """Write one edited field back into the draft and revalidate.

        A blank entry removes the key rather than storing an empty string, so
        clearing a field means "use the default" -- which is how the loader
        already reads a missing one.
        """
        if self._loading:
            return
        node = self.component(self.selected)
        if path[0] in ('profile', 'transport'):
            text = variable.get().strip()
            target = self.draft if path[0] == 'profile' else self.draft.setdefault('transport', {})
            try:
                target[path[1]] = text if path[1] == 'name' else float(text)
            except ValueError:
                target[path[1]] = text
            self.refresh_all()
            return
        if node is None:
            return
        target, key = node, path[0]
        for step in path[:-1]:
            target = target.setdefault(step, {})
        key = path[-1]
        text = variable.get().strip()
        if path == ('model', 'fov_h_deg') and 'fx_px' in target:
            if target.get('fy_px') == target['fx_px']:
                target.pop('fy_px', None)
            target.pop('fx_px')
        if key == "id":
            if text == 'body' or any(n is not node and n['id'] == text for n in self.components()):
                self.error = 'id: duplicate or reserved ID'
                self.resolved = None
                self._refresh_status()
                self.refresh()
                return
            self._rename(node, text)
        elif not text:
            target.pop(key, None)
        elif key in ("parent", "sync_group"):
            target[key] = text
        elif key == "enabled":
            target[key] = text == "true"
        elif path[0] == 'limits':
            try:
                target[key] = [float(v.strip()) for v in text.split(',')]
            except ValueError:
                target[key] = text
        else:
            try:
                target[key] = int(text) if key in INTEGER_FIELDS else float(text)
            except ValueError:
                target[key] = text
        self.refresh_all()

    def _rename(self, node, name):
        previous = node["id"]
        if not name or name == previous:
            return
        node["id"] = name
        for other in self.components():
            if other.get("parent") == previous:
                other["parent"] = name
        if self.draft.get("primary_camera") == previous:
            self.draft["primary_camera"] = name
        self.selected = name

    # -- resolved readout -------------------------------------------------

    def _refresh_resolved(self, node):
        if node is None or self.resolved is None:
            self.resolved_var.set("" if node else "select a component")
            return
        data = self.resolved.data
        entry = next((s for s in self.components(data) if s["id"] == node.get("id")), None)
        if entry is None:
            self.resolved_var.set("")
            return
        lines = []
        # Body axes, not map: where a sensor sits on the airframe does not
        # depend on where the airframe is, and showing it against a heading
        # would make a mount look different from every direction.
        chain = mount_chain(data, entry["id"])
        angles = euler_angles(chain[:3, :3])
        lines.append("from body: x %+.3f  y %+.3f  z %+.3f m" % tuple(chain[:3, 3]))
        lines.append("           yaw %+.2f  pitch %+.2f  roll %+.2f deg" %
                     (angles["yaw_deg"], angles["pitch_deg"], angles["roll_deg"]))
        model = entry.get("model", {})
        if entry.get("type") == "camera.rgb":
            lines.append("fx %.2f  fy %.2f  cx %.1f  cy %.1f px" %
                         (model["fx_px"], model["fy_px"], model["cx_px"], model["cy_px"]))
            lines.append("fov %.2f x %.2f deg" % (
                _fov(model["width_px"], model["fx_px"]),
                _fov(model["height_px"], model["fy_px"])))
            lines.append(self._baseline(data, entry))
        elif entry.get("type") in ARRAY_TYPES:
            ring = self.resolved.plan["arrays"].get(entry["id"], {})
            lines.append("record %d B  ring %.0f KiB" %
                         (ring.get("record_bytes", 0), ring.get("capacity", 0) / 1024))
        plan = self.resolved.plan
        channel = plan["cameras"].get(entry["id"], plan["arrays"].get(entry["id"]))
        if channel:
            lines.append("channel %.1f MiB" %
                         (channel.get("bytes", channel.get("capacity", 0)) / 1048576))
        self.resolved_var.set("\n".join(line for line in lines if line))

    def _baseline(self, data, entry):
        """The stereo geometry a pair actually resolves to, if it is in one."""
        group = entry.get("sync_group")
        if not group:
            return ""
        members = [s for s in data["sensors"]
                   if s.get("sync_group") == group and s["enabled"]]
        if len(members) != 2:
            return f"sync {group}: {len(members)} members"
        left, right = (mount_chain(data, m["id"]) for m in members)
        offset = right[:3, 3] - left[:3, 3]
        separation = float(sum(v * v for v in offset) ** 0.5)
        relative = euler_angles(left[:3, :3].T @ right[:3, :3])
        return ("sync %s: baseline %.4f m  disparity %.1f px at 10 m\n"
                "relative yaw %+.2f  pitch %+.2f  roll %+.2f deg" % (
                    group, separation, entry["model"]["fx_px"] * separation / 10.0,
                    relative["yaw_deg"], relative["pitch_deg"],
                    relative["roll_deg"]))

    # -- actions ----------------------------------------------------------

    def add(self, kind):
        taken = {n.get("id") for n in self.components()}
        node = new_component(kind, taken)
        node["parent"] = self.selected if self.selected in {
            m.get("id") for m in self.draft.get("mounts", [])} else "body"
        key = "mounts" if kind in MOUNT_TYPES else "sensors"
        self.draft.setdefault(key, []).append(node)
        self.selected = node["id"]
        self.refresh_all()

    def duplicate(self):
        node = self.component(self.selected)
        if node is None:
            return
        clone = copy.deepcopy(node)
        clone["id"] = unique_id(f"{node['id']}_copy",
                                {n.get("id") for n in self.components()})
        key = "mounts" if node["type"] in MOUNT_TYPES else "sensors"
        self.draft[key].append(clone)
        self.selected = clone["id"]
        self.refresh_all()

    def remove(self):
        """Delete a component; its children move up to its parent.

        Reparenting rather than cascading: removing a mount should not
        silently take the cameras on it with it, and leaving them orphaned
        would make the draft unresolvable for a reason nobody asked for.
        """
        node = self.component(self.selected)
        if node is None:
            return
        for other in self.components():
            if other.get("parent") == node["id"]:
                other["parent"] = node["parent"]
        for key in ("mounts", "sensors"):
            self.draft[key] = [n for n in self.draft.get(key, []) if n is not node]
        self.selected = node["parent"]
        self.refresh_all()

    def make_primary(self):
        node = self.component(self.selected)
        if node is not None and node.get("type") == "camera.rgb":
            self.draft["primary_camera"] = node["id"]
            self.refresh_all()

    def add_stereo(self):
        """Insert a synchronized pair with an explicit left/right baseline."""
        taken = {n.get("id") for n in self.components()}
        base = next(f"stereo{n}" for n in range(1, 100)
                    if not {f"stereo{n}_left", f"stereo{n}_right"} & taken)
        parent = self.selected if self.selected in {
            m.get("id") for m in self.draft.get("mounts", [])} else "body"
        pair = stereo_pair(base, parent=parent, pose=dict(z_m=.1, pitch_deg=-5.))
        self.draft.setdefault("sensors", []).extend(pair)
        self.selected = pair[0]["id"]
        self.refresh_all()

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("Drone profile", "*.json")])
        if not path:
            return
        try:
            self.draft = DroneProfile.load(path).data
        except (ValueError, OSError) as exc:
            self.info.set(str(exc))
            return
        self.path = path
        self.selected = self.draft["primary_camera"]
        self._editor_key = None
        self.refresh_all()

    def save(self):
        if self.resolved is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json")
        if path:
            try:
                self.resolved.save(path)
                self.path = path
                self._refresh_status()
            except OSError as exc:
                self.info.set(str(exc))

    def apply(self):
        """Hand the simulator a complete, validated, immutable profile."""
        if self.resolved is None:
            return
        try:
            self.sim.apply_profile(self.resolved)
        except Exception as exc:
            self.info.set(f"Apply failed: {exc}")
            return
        self.refresh_all()

    def refresh(self):
        armed = getattr(self.sim.state, "armed", False)
        self.apply_button.configure(
            state="disabled" if armed or self.resolved is None else "normal")


def _fov(pixels, focal):
    return math.degrees(2.0 * math.atan(pixels / (2.0 * focal)))


def _as_text(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return "" if value is None else str(value)


def component_tree(profile):
    """The flat parent-linked profile drawn as the tree its links describe.

    The same view the tab renders, as text, for a report or a test that does
    not want a window.
    """
    data = profile.data
    children = {}
    for node in (*data["mounts"], *data["sensors"]):
        children.setdefault(node["parent"], []).append(node)
    lines = ["body"]

    def walk(parent, prefix):
        kids = children.get(parent, [])
        for index, node in enumerate(kids):
            last = index == len(kids) - 1
            lines.append(prefix + ("`-- " if last else "|-- ") + _label(data, node))
            walk(node["id"], prefix + ("    " if last else "|   "))

    walk("body", "")
    return "\n".join(lines)


def _label(data, node):
    badges = [node["type"]]
    if node["id"] == data["primary_camera"]:
        badges.append("primary")
    if node.get("sync_group"):
        badges.append(f"sync: {node['sync_group']}")
    if "rate_hz" in node:
        badges.append(f"{node['rate_hz']:g} Hz")
    if node.get("enabled") is False:
        badges.append("disabled")
    if node["type"] in ARRAY_TYPES:
        badges.append("array ring")
    return f"{node['id']} ({', '.join(badges)})"

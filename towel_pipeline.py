"""
towel_pipeline.py  -  one-click synthetic dataset generator

FOR THE CLIENT
--------------
1. Open the .blend file.
2. Go to the Scripting tab, click Run (the play arrow) once.
   (If the "Towel Data" panel is already there, skip this.)
3. Go back to Layout, press N to open the sidebar, click the "Towel Data" tab.
4. Pick an output folder, set how many images you want, press
   "Render Dataset".

No terminal, no Python install, no pip. Everything this needs is already
inside Blender.

You get:
    <output folder>/
        images/           the rendered pictures
        ids/              internal - used to work out where each towel is
        annotations.json  COCO format, ready for training
        counts.csv        how many towels are visible in each image
        seeds.csv         which seed made which image
        pipeline_log.txt  what happened, including any errors

IF SOMETHING GOES WRONG
-----------------------
Read pipeline_log.txt in the output folder. Every step and every error is
written there, so you do not need a terminal to see what failed.

If the images rendered but annotations.json is missing, press
"Annotate Existing IDs" - it redoes just the annotation step from the ids/
folder already on disk, and takes seconds instead of re-rendering.

WHY THERE ARE NO DEPENDENCIES
-----------------------------
Blender ships its own Python with numpy included, so the annotation maths runs
natively. Reading the ID images uses Blender's own image loader rather than
Pillow, and segmentation is emitted as COCO RLE, which needs no contour
library. RLE is accepted directly by detectron2, mmdetection and pycocotools.

Requires build_towel_stacks.py to have been run first (it creates the
TowelStacks object and its Geometry Nodes setup).
"""

import bpy
import csv
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
from mathutils import Euler, Matrix, Vector
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.object_utils import world_to_camera_view

GEN_PREFIX = "TowelStackGenerator"
ID_MATERIAL = "InstanceIDPass"

# Must match ID_LEVELS in build_towel_stacks.py
ID_LEVELS = 16

MIN_AREA = 40            # px - drop slivers
MIN_STACK_AREA = 150     # px - a whole pile is much larger than one towel
EROSION_SURVIVAL = 0.25  # geometric fringe rejection


# ---------------------------------------------------------------------------
# logging - written to a file so no terminal is needed
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, out_dir=None):
        self.path = None
        self.lines = []
        if out_dir:
            try:
                os.makedirs(out_dir, exist_ok=True)
                self.path = os.path.join(out_dir, "pipeline_log.txt")
                with open(self.path, "w") as fh:
                    fh.write(f"Towel pipeline log - {time.asctime()}\n"
                             f"Blender {bpy.app.version_string}\n"
                             + "=" * 60 + "\n")
            except Exception:
                self.path = None

    def __call__(self, msg):
        line = f"{msg}"
        self.lines.append(line)
        print(f"[towel] {line}")
        if self.path:
            try:
                with open(self.path, "a") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass

    def error(self, exc):
        self("")
        self("ERROR: " + str(exc))
        for ln in traceback.format_exc().splitlines():
            self("    " + ln)


# ---------------------------------------------------------------------------
# generator plumbing
# ---------------------------------------------------------------------------

def get_generators(scene):
    found = []
    for obj in scene.objects:
        for mod in obj.modifiers:
            if mod.type == 'NODES' and mod.node_group \
                    and mod.node_group.name.startswith(GEN_PREFIX):
                found.append((obj, mod))
    return found


def socket_identifier(node_group, name):
    for item in node_group.interface.items_tree:
        if getattr(item, "item_type", "") == 'SOCKET' \
                and item.in_out == 'INPUT' and item.name == name:
            return item.identifier
    return None


def set_input(mod, name, value):
    ident = socket_identifier(mod.node_group, name)
    if ident is not None:
        mod[ident] = value
        return True
    return False


def get_input(mod, name, default=None):
    ident = socket_identifier(mod.node_group, name)
    return mod[ident] if ident is not None else default


def assign_id_offsets(generators):
    """Each generator gets its own slice of the ID space so masks don't collide."""
    n = len(generators)
    if n == 1:
        set_input(generators[0][1], "ID Offset", 0)
        return
    slice_size = 4095 // n
    for i, (_, mod) in enumerate(generators):
        set_input(mod, "ID Offset", i * slice_size)


# ---------------------------------------------------------------------------
# engine / lighting / preflight
# ---------------------------------------------------------------------------

def eevee_engine_id():
    try:
        items = bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items
        names = {i.identifier for i in items}
    except Exception:
        names = set()
    for c in ('BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT'):
        if c in names:
            return c
    return 'BLENDER_EEVEE'


def configure_engine(scene, engine, device, samples, log):
    if engine == 'CYCLES':
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        if device == 'GPU':
            try:
                prefs = bpy.context.preferences.addons["cycles"].preferences
                prefs.get_devices()
                usable = [d.name for d in prefs.devices if d.type != 'CPU']
                if not usable:
                    raise RuntimeError("no GPU devices found")
                for d in prefs.devices:
                    d.use = True
                scene.cycles.device = 'GPU'
                log(f"Cycles on GPU: {', '.join(usable)}")
            except Exception as exc:
                scene.cycles.device = 'CPU'
                log(f"GPU unavailable ({exc}) - using CPU")
        else:
            scene.cycles.device = 'CPU'
            log(f"Cycles on CPU, {samples} samples")
    else:
        scene.render.engine = eevee_engine_id()
        scene.eevee.taa_render_samples = max(16, min(samples, 128))
        log(f"{scene.render.engine}, {scene.eevee.taa_render_samples} samples")


def aim_at(obj, target):
    d = target - obj.location
    if d.length > 1e-6:
        obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def ensure_lighting(scene, generators, log):
    """Build a 3-point rig only if the scene has no lights.

    Solid and Material Preview shading use Blender's built-in studio HDRI,
    which does not exist at render time. A scene with no lamps looks fine in
    the viewport and renders nearly black.
    """
    if any(o.type == 'LIGHT' for o in scene.objects):
        return
    target = generators[0][0].location.copy()
    tiers = get_input(generators[0][1], "Shelf Count", 3)
    gap = get_input(generators[0][1], "Shelf Spacing Z", 0.62)
    target.z += (tiers - 1) * gap * 0.5

    for name, energy, loc, size in (("AutoKey", 400.0, (2.6, -3.2, 2.8), 3.0),
                                    ("AutoFill", 120.0, (-3.0, -2.4, 1.6), 4.0),
                                    ("AutoRim", 200.0, (0.5, 3.0, 2.6), 2.5)):
        data = bpy.data.lights.new(name, 'AREA')
        data.energy, data.size = energy, size
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        obj.location = (target.x + loc[0], target.y + loc[1], target.z + loc[2])
        aim_at(obj, target)

    if scene.world is None:
        scene.world = bpy.data.worlds.new("AutoWorld")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0.35, 0.36, 0.38, 1.0)
        bg.inputs["Strength"].default_value = 0.6
    log("No lights found - built a 3-point rig")


# ---------------------------------------------------------------------------
# per-frame camera shake and light variation
# ---------------------------------------------------------------------------

# Hard floor on the light multiplier. Whatever the user asks for, a frame will
# never drop below this fraction of their own lighting - dark frames teach a
# counting model nothing except how to miss towels.
LIGHT_FLOOR = 0.65


class Baseline:
    """Snapshot of the camera pose and light levels.

    Jitter is always applied relative to this and reverted afterwards, so the
    client's scene is exactly as they left it when the run finishes - even if
    the run fails halfway.

    The camera is stored as a full matrix rather than an euler so it works
    whatever rotation mode they used, and so jitter can be composed in the
    camera's LOCAL frame (pitch/yaw/roll about its own axes, which is what
    handheld shake actually looks like).
    """

    def __init__(self, scene):
        self.scene = scene
        cam = scene.camera
        self.cam = cam
        self.cam_matrix = cam.matrix_world.copy() if cam else None
        self.lights = [(o, o.data.energy)
                       for o in scene.objects if o.type == 'LIGHT']
        self.world_bg = None
        self.world_strength = None
        if scene.world and scene.world.use_nodes:
            bg = scene.world.node_tree.nodes.get("Background")
            if bg:
                self.world_bg = bg
                self.world_strength = bg.inputs["Strength"].default_value

    def restore(self):
        if self.cam and self.cam_matrix:
            self.cam.matrix_world = self.cam_matrix
        for obj, energy in self.lights:
            obj.data.energy = energy
        if self.world_bg is not None:
            self.world_bg.inputs["Strength"].default_value = self.world_strength


def apply_variation(base, seed, cfg):
    """Deterministic per-frame camera shake and light variation.

    Keyed purely on the frame's seed, so the ID pass reproduces the exact
    same camera pose as the beauty pass. If this were not reproducible the
    masks would be offset from the images and every label would be wrong.
    """
    rng = random.Random(seed * 7919 + 104729)

    if cfg.jitter_camera and base.cam and base.cam_matrix:
        a = math.radians(cfg.cam_angle)
        shake = Euler((rng.uniform(-a, a),      # pitch
                       rng.uniform(-a, a),      # yaw
                       rng.uniform(-a, a)),     # roll about the view axis
                      'XYZ').to_matrix().to_4x4()
        s = cfg.cam_shift
        shift = Matrix.Translation(Vector((rng.uniform(-s, s),
                                           rng.uniform(-s, s),
                                           rng.uniform(-s, s))))
        # Right-multiplied, so both are in the camera's own frame.
        base.cam.matrix_world = base.cam_matrix @ shift @ shake

    if cfg.jitter_light:
        lo = max(1.0 - cfg.light_var, LIGHT_FLOOR)
        hi = 1.0 + cfg.light_var
        for obj, energy in base.lights:
            obj.data.energy = energy * rng.uniform(lo, hi)
        if base.world_bg is not None:
            # Ambient moves less than the key lights, and never toward black.
            wlo = max(1.0 - cfg.light_var * 0.5, LIGHT_FLOOR)
            base.world_bg.inputs["Strength"].default_value = (
                base.world_strength * rng.uniform(wlo, 1.0 + cfg.light_var * 0.5))


def preflight(scene, generators):
    """Return (problems, instance_count)."""
    fatal = []
    deps = bpy.context.evaluated_depsgraph_get()
    gen_objs = [o for o, _ in generators]

    for obj, _ in generators:
        if obj.hide_render:
            fatal.append(f"'{obj.name}' is hidden from renders (camera icon "
                         f"in the Outliner)")
        for coll in obj.users_collection:
            if coll.hide_render:
                fatal.append(f"Collection '{coll.name}' is hidden from renders")

    n = sum(1 for i in deps.object_instances
            if i.is_instance and i.parent and i.parent.original in gen_objs)
    if n == 0:
        fatal.append("The generator produced 0 towels - check Towel Variants "
                     "and Fill Probability on the modifier")

    if scene.camera is None:
        fatal.append("No camera in the scene")
    elif n:
        inside = 0
        for i in deps.object_instances:
            if not (i.is_instance and i.parent and i.parent.original in gen_objs):
                continue
            co = world_to_camera_view(scene, scene.camera,
                                      i.matrix_world.translation)
            if co.z > 0 and 0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0:
                inside += 1
        if inside == 0:
            fatal.append("No towels are inside the camera frame - the camera "
                         "is pointing somewhere else")

    if scene.render.use_border:
        fatal.append("Render Region is on (Output Properties > Format)")
    return fatal, n


# ---------------------------------------------------------------------------
# instance-ID pass
# ---------------------------------------------------------------------------

def build_id_material(attribute="inst_color", name=None):
    """Flat emission driven by one of the generator's ID colour attributes.

    'inst_color'  - unique per towel, gives per-object masks
    'stack_color' - shared by every towel in a pile, gives per-stack masks
    """
    name = name or f"{ID_MATERIAL}_{attribute}"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    attr = nt.nodes.new("ShaderNodeAttribute")
    attr.attribute_type = 'INSTANCER'
    attr.attribute_name = attribute
    attr.location = (-400, 0)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = 1.0
    emit.location = (-200, 0)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(attr.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    return mat


# The ID pass is forced to Cycles even when the beauty pass is EEVEE:
# view_layer.material_override is Cycles-only in Blender 4.0/4.1. EEVEE
# accepts it silently and ignores it, producing a normal lit render instead of
# flat ID colours. All bounces are zeroed, so an emission-only scene at one
# sample is effectively a rasterisation - about a second a frame on CPU.
_ID_PASS = [
    ("render.engine", 'CYCLES'),
    ("view_settings.view_transform", 'Standard'),
    ("view_settings.look", 'None'),
    ("view_settings.exposure", 0.0),
    ("view_settings.gamma", 1.0),
    ("render.filter_size", 0.01),
    ("render.film_transparent", False),
    ("render.use_compositing", False),
    ("render.use_sequencer", False),
    ("render.use_motion_blur", False),
    ("cycles.samples", 1),
    ("cycles.use_denoising", False),
    ("cycles.use_adaptive_sampling", False),
    ("cycles.max_bounces", 0),
    ("cycles.diffuse_bounces", 0),
    ("cycles.glossy_bounces", 0),
    ("cycles.transmission_bounces", 0),
    ("cycles.volume_bounces", 0),
    ("cycles.transparent_max_bounces", 0),
]


def _get_path(root, path):
    o = root
    for p in path.split("."):
        o = getattr(o, p)
    return o


def _set_path(root, path, value):
    parts = path.split(".")
    o = root
    for p in parts[:-1]:
        o = getattr(o, p)
    setattr(o, parts[-1], value)


class IDPassContext:
    def __init__(self, scene, material, device='CPU'):
        self.scene, self.material, self.device = scene, material, device

    def __enter__(self):
        s, vl = self.scene, self.scene.view_layers[0]
        self.items = []
        for path, value in _ID_PASS:
            try:
                _get_path(s, path)
                self.items.append((path, value))
            except AttributeError:
                pass
        self.saved = {p: _get_path(s, p) for p, _ in self.items}
        self.saved_override = vl.material_override
        self.saved_world = s.world
        for path, value in self.items:
            _set_path(s, path, value)
        s.cycles.device = self.device
        vl.material_override = self.material

        black = bpy.data.worlds.get("IDPassBlack")
        if black is None:
            black = bpy.data.worlds.new("IDPassBlack")
            black.use_nodes = True
            bg = black.node_tree.nodes.get("Background")
            if bg:
                bg.inputs["Color"].default_value = (0, 0, 0, 1)
                bg.inputs["Strength"].default_value = 0.0
        s.world = black
        return self

    def __exit__(self, *exc):
        s, vl = self.scene, self.scene.view_layers[0]
        for path, _ in self.items:
            _set_path(s, path, self.saved[path])
        vl.material_override = self.saved_override
        s.world = self.saved_world
        return False


# ---------------------------------------------------------------------------
# annotation - pure numpy, no external libraries
# ---------------------------------------------------------------------------

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def read_png_raw(path):
    """Load a PNG as raw 0-1 values with no colour transform applied.

    'Non-Color' stops Blender applying its own sRGB conversion, so the numbers
    match exactly what the encoder wrote and the decode below is identical to
    the standalone coco_from_masks.py.
    """
    if not os.path.exists(path):
        raise RuntimeError(f"ID image missing: {path}")
    img = bpy.data.images.load(path)
    try:
        try:
            img.colorspace_settings.name = 'Non-Color'
        except TypeError:
            img.colorspace_settings.name = 'Linear'
        w, h = img.size
        if w == 0 or h == 0:
            raise RuntimeError(f"ID image has zero size: {path}")
        ch = img.channels or 4
        buf = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(buf)
        # Blender stores images bottom-up; flip to normal orientation.
        arr = buf.reshape(h, w, ch)[::-1]
        if ch >= 3:
            return np.ascontiguousarray(arr[:, :, :3])
        return np.ascontiguousarray(np.repeat(arr[:, :, :1], 3, axis=2))
    finally:
        bpy.data.images.remove(img)


def decode_ids(path):
    rgb = read_png_raw(path)
    lin = srgb_to_linear(rgb) * (ID_LEVELS - 1)
    q = np.clip(np.rint(lin), 0, ID_LEVELS - 1).astype(np.int32)
    return (q[:, :, 0] * ID_LEVELS * ID_LEVELS
            + q[:, :, 1] * ID_LEVELS
            + q[:, :, 2])


def erode(mask):
    """4-connected 1px erosion."""
    e = mask.copy()
    e[1:, :] &= mask[:-1, :]
    e[:-1, :] &= mask[1:, :]
    e[:, 1:] &= mask[:, :-1]
    e[:, :-1] &= mask[:, 1:]
    return e


def mask_to_rle(mask):
    """COCO uncompressed RLE, column-major, fully vectorised."""
    f = mask.T.reshape(-1)
    idx = np.flatnonzero(np.diff(f)) + 1
    bounds = np.concatenate(([0], idx, [f.size]))
    counts = np.diff(bounds).astype(int).tolist()
    if f[0]:
        counts = [0] + counts     # RLE must open with a background run
    return {"counts": counts, "size": [int(mask.shape[0]), int(mask.shape[1])]}


def annotate(id_path, min_area=MIN_AREA):
    """One ID image -> (records, width, height).

    Works for both passes: the instance pass gives one record per towel, the
    stack pass gives one per pile, because every towel in a pile was painted
    the same colour.
    """
    ids = decode_ids(id_path)
    h, w = ids.shape
    records = []
    for inst in np.unique(ids):
        if inst == 0:
            continue
        mask = ids == inst
        area = int(mask.sum())
        if area < min_area:
            continue
        # Antialiased edge pixels can decode to a valid but wrong id. They form
        # 1px fringes, which vanish under erosion; real towels don't.
        if int(erode(mask).sum()) < EROSION_SURVIVAL * area:
            continue
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        records.append({
            "instance_id": int(inst),
            "bbox": [float(x0), float(y0),
                     float(int(x1) - int(x0) + 1), float(int(y1) - int(y0) + 1)],
            "area": area,
            "segmentation": mask_to_rle(mask),
        })
    return records, int(w), int(h)


def annotate_folder(out_dir, log, progress=None, seeds=None):
    """Annotate every PNG in <out_dir>/ids and write the COCO files.

    Split out from the render loop so it can be re-run on its own - if a
    render finished but annotation failed, this recovers the labels in
    seconds instead of re-rendering everything.
    """
    id_dir = os.path.join(out_dir, "ids")
    if not os.path.isdir(id_dir):
        raise RuntimeError(f"No ids folder at {id_dir}. Re-render with "
                           f"'Create Annotations' ticked.")
    files = sorted(f for f in os.listdir(id_dir) if f.lower().endswith(".png"))
    if not files:
        raise RuntimeError(f"No PNG files in {id_dir}")

    stack_dir = os.path.join(out_dir, "stacks")
    with_stacks = os.path.isdir(stack_dir)

    log(f"Annotating {len(files)} ID images from {id_dir}"
        + ("  (+ stack outlines)" if with_stacks else ""))
    coco = {
        "info": {"description": "Synthetic folded-linen shelf dataset",
                 "generator": "TowelStackGenerator", "segmentation": "rle"},
        "licenses": [],
        "images": [], "annotations": [],
        # Two classes: individual items, and the pile each belongs to. A model
        # can be trained on either or both.
        "categories": [{"id": 1, "name": "towel", "supercategory": "linen"},
                       {"id": 2, "name": "stack", "supercategory": "linen"}],
    }
    counts, ann_id = [], 1
    n_towels = n_stacks = 0

    for i, name in enumerate(files):
        records, w, h = annotate(os.path.join(id_dir, name), MIN_AREA)
        stacks = []
        if with_stacks:
            spath = os.path.join(stack_dir, name)
            if os.path.exists(spath):
                stacks, _, _ = annotate(spath, MIN_STACK_AREA)

        image_id = i + 1
        entry = {"id": image_id, "file_name": name, "width": w, "height": h,
                 "visible_towels": len(records)}
        if with_stacks:
            entry["visible_stacks"] = len(stacks)
        if seeds and name in seeds:
            entry["seed"] = seeds[name]
        coco["images"].append(entry)

        for cat, group in ((1, records), (2, stacks)):
            for r in group:
                coco["annotations"].append({
                    "id": ann_id, "image_id": image_id, "category_id": cat,
                    "bbox": r["bbox"], "area": r["area"], "iscrowd": 0,
                    "segmentation": r["segmentation"],
                    "instance_id": r["instance_id"]})
                ann_id += 1

        n_towels += len(records)
        n_stacks += len(stacks)
        counts.append([name, len(records), len(stacks) if with_stacks else ""])
        if progress:
            progress((i + 1) / len(files))
        log(f"  {name}: {len(records)} towels"
            + (f", {len(stacks)} stacks" if with_stacks else "") + " visible")

    with open(os.path.join(out_dir, "annotations.json"), "w") as fh:
        json.dump(coco, fh)
    with open(os.path.join(out_dir, "counts.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh)
        w_.writerow(["image", "visible_towels", "visible_stacks"])
        w_.writerows(counts)

    n = max(1, len(files))
    log(f"Wrote annotations.json - {len(files)} images, "
        f"{n_towels} towels (avg {n_towels / n:.1f}/image)"
        + (f", {n_stacks} stacks (avg {n_stacks / n:.1f}/image)"
           if with_stacks else ""))
    return n_towels + n_stacks


# ---------------------------------------------------------------------------
# visual check - draw the labels back onto the renders
# ---------------------------------------------------------------------------

def rle_to_mask(rle):
    """Inverse of mask_to_rle."""
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    return flat.reshape(w, h).T


def _instance_colour(i):
    """Distinct hue per instance via the golden angle, converted HSV->RGB."""
    hgt = (i * 0.61803398875) % 1.0
    h6, f = hgt * 6.0, (hgt * 6.0) % 1.0
    v, p, q, t = 1.0, 0.15, 1.0 - 0.85 * f, 0.15 + 0.85 * f
    return [(v, t, p), (q, v, p), (p, v, t),
            (p, q, v), (t, p, v), (v, p, q)][int(h6) % 6]


def write_label_previews(out_dir, log, limit=12, progress=None):
    """Composite each annotation's mask and box onto its render.

    This is purely for eyeballing - it is not part of the training data. If the
    tinted shapes line up with the towels, the labels are correct.
    """
    ann_path = os.path.join(out_dir, "annotations.json")
    if not os.path.exists(ann_path):
        raise RuntimeError("No annotations.json yet - run 'Render Dataset' or "
                           "'Annotate Existing IDs' first.")
    with open(ann_path) as fh:
        coco = json.load(fh)

    img_dir = os.path.join(out_dir, "images")
    prev_dir = os.path.join(out_dir, "preview")
    os.makedirs(prev_dir, exist_ok=True)

    by_image = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)

    todo = coco["images"][:limit]
    log(f"Drawing label previews for {len(todo)} image(s)")
    written = 0
    for n, meta in enumerate(todo):
        src = os.path.join(img_dir, meta["file_name"])
        if not os.path.exists(src):
            log(f"  skipping {meta['file_name']} - render not found")
            continue

        # One bad frame must not lose the rest of the batch.
        try:
            written += _draw_one_preview(src, prev_dir, meta,
                                         by_image.get(meta["id"], []), log)
        except Exception as exc:
            log(f"  FAILED on {meta['file_name']}: {exc}")

        if progress:
            progress((n + 1) / max(1, len(todo)))

    if written:
        log(f"{written} preview(s) written to {prev_dir}")
    else:
        log(f"WARNING: no previews written. Check that {img_dir} contains the "
            f"renders named in annotations.json.")
    return prev_dir


def _draw_one_preview(src, prev_dir, meta, anns, log):
    """Composite one image's labels and save it. Returns 1 on success."""
    # 'Non-Color' keeps the stored display values, so the preview looks
    # exactly like the render rather than being double-transformed.
    img = bpy.data.images.load(src)
    try:
        img.colorspace_settings.name = 'Non-Color'
        w, h = img.size
        ch = img.channels or 4
        if w == 0 or h == 0:
            raise RuntimeError("render has zero size")
        buf = np.empty(w * h * ch, dtype=np.float32)
        img.pixels.foreach_get(buf)
        canvas = buf.reshape(h, w, ch)[::-1].copy()
    finally:
        bpy.data.images.remove(img)

    # Towels first, stacks on top, so the stack outline stays readable over
    # the tinted towels inside it.
    n_towels = n_stacks = 0
    for a in sorted(anns, key=lambda r: r.get("category_id", 1)):
        is_stack = a.get("category_id", 1) == 2
        seg = a.get("segmentation")

        if is_stack:
            n_stacks += 1
            # Outline only, in white - filling would bury the towels.
            if isinstance(seg, dict) and "counts" in seg:
                m = rle_to_mask(seg)
                if m.shape == (h, w):
                    edge = m & ~erode(erode(m))
                    canvas[..., :3][edge] = (1.0, 1.0, 1.0)
            colour = np.array((1.0, 1.0, 1.0), dtype=np.float32)
            thick = 3
        else:
            n_towels += 1
            colour = np.array(_instance_colour(a["id"]), dtype=np.float32)
            if isinstance(seg, dict) and "counts" in seg:
                m = rle_to_mask(seg)
                if m.shape == (h, w):
                    canvas[..., :3][m] = (canvas[..., :3][m] * 0.55
                                          + colour * 0.45)
            thick = 2

        x, y, bw, bh = [int(round(v)) for v in a["bbox"]]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w - 1, x + bw - 1), min(h - 1, y + bh - 1)
        if x1 > x0 and y1 > y0:
            canvas[y0:y0 + thick, x0:x1 + 1, :3] = colour
            canvas[y1 - thick + 1:y1 + 1, x0:x1 + 1, :3] = colour
            canvas[y0:y1 + 1, x0:x0 + thick, :3] = colour
            canvas[y0:y1 + 1, x1 - thick + 1:x1 + 1, :3] = colour

    name = meta["file_name"].replace(".png", "_labels.png")
    path = os.path.join(prev_dir, name)
    out_img = bpy.data.images.new(name, width=w, height=h, alpha=(ch == 4))
    try:
        out_img.colorspace_settings.name = 'Non-Color'
        out_img.pixels.foreach_set(
            np.ascontiguousarray(canvas[::-1]).reshape(-1))
        out_img.filepath_raw = path
        out_img.file_format = 'PNG'
        out_img.save()
    finally:
        # Blender may have renamed the datablock on a name clash; the file
        # still went to `path`, which is what matters.
        bpy.data.images.remove(out_img)

    if not os.path.exists(path):
        raise RuntimeError(f"save produced no file at {path}")
    log(f"  {name}: {n_towels} towels, {n_stacks} stacks drawn")
    return 1

    log(f"Previews written to {prev_dir}")
    return prev_dir


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def resolve_out(cfg):
    out = bpy.path.abspath(cfg.output_dir)
    if not out or out in ("//", ""):
        raise RuntimeError("Pick an output folder first.")
    return os.path.abspath(out)


def render_dataset(scene, cfg, log, progress=None):
    generators = get_generators(scene)
    if not generators:
        raise RuntimeError("No TowelStacks object found. Run "
                           "build_towel_stacks.py first.")
    out = resolve_out(cfg)
    img_dir = os.path.join(out, "images")
    id_dir = os.path.join(out, "ids")
    stack_dir = os.path.join(out, "stacks")
    os.makedirs(img_dir, exist_ok=True)
    if cfg.write_ids:
        os.makedirs(id_dir, exist_ok=True)
        if cfg.write_stacks:
            os.makedirs(stack_dir, exist_ok=True)

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '8'
    configure_engine(scene, cfg.engine, cfg.device, cfg.samples, log)
    ensure_lighting(scene, generators, log)
    if cfg.write_ids:
        assign_id_offsets(generators)

    for _, mod in generators:
        set_input(mod, "Seed", cfg.start_seed)
    bpy.context.view_layer.update()

    fatal, n_inst = preflight(scene, generators)
    if fatal:
        raise RuntimeError("Renders would be empty:\n  - " + "\n  - ".join(fatal))
    log(f"{n_inst} towels in the scene, camera framed OK")

    rows = [(f"frame_{i:06d}.png", cfg.start_seed + i) for i in range(cfg.count)]
    n_passes = 1 + (1 if cfg.write_ids else 0) \
        + (1 if cfg.write_ids and cfg.write_stacks else 0)
    total_steps = len(rows) * n_passes
    step = 0
    t0 = time.time()

    if cfg.jitter_camera or cfg.jitter_light:
        bits = []
        if cfg.jitter_camera:
            bits.append(f"camera +/-{cfg.cam_angle:.1f} deg, "
                        f"+/-{cfg.cam_shift * 100:.0f} cm")
        if cfg.jitter_light:
            lo = max(1.0 - cfg.light_var, LIGHT_FLOOR)
            bits.append(f"light {lo * 100:.0f}-{(1 + cfg.light_var) * 100:.0f}% "
                        f"of your setup")
        log("Per-frame variation: " + "; ".join(bits))

    base = Baseline(scene)
    try:
        # Pass 1 - beauty. Kept separate from the ID pass so the engine switch
        # (and its full scene re-sync) happens once, not twice per frame.
        log(f"Pass 1: rendering {cfg.count} images")
        for name, seed in rows:
            for _, mod in generators:
                set_input(mod, "Seed", seed)
            apply_variation(base, seed, cfg)
            bpy.context.view_layer.update()
            scene.render.filepath = os.path.join(img_dir, name)
            bpy.ops.render.render(write_still=True)
            step += 1
            if progress:
                progress(step / total_steps)
            log(f"  image {step}/{len(rows)}  {name}")

        with open(os.path.join(out, "seeds.csv"), "w", newline="") as fh:
            w_ = csv.writer(fh)
            w_.writerow(["image", "seed"])
            w_.writerows(rows)

        if cfg.write_ids:
            # Both ID passes share one context, so the switch to Cycles and
            # its scene re-sync happens once for the whole run.
            passes = [("towel", id_dir, build_id_material("inst_color"))]
            if cfg.write_stacks:
                passes.append(("stack", stack_dir,
                               build_id_material("stack_color")))

            with IDPassContext(scene, passes[0][2], cfg.device) as ctx:
                for label, folder, mat in passes:
                    scene.view_layers[0].material_override = mat
                    log(f"Rendering {cfg.count} {label} ID passes "
                        f"(Cycles, 1 sample)")
                    for i, (name, seed) in enumerate(rows):
                        for _, mod in generators:
                            set_input(mod, "Seed", seed)
                        # Same seed -> same camera pose as pass 1. This has to
                        # match exactly or every mask is offset from its image.
                        apply_variation(base, seed, cfg)
                        bpy.context.view_layer.update()
                        scene.render.filepath = os.path.join(folder, name)
                        bpy.ops.render.render(write_still=True)
                        step += 1
                        if progress:
                            progress(step / total_steps)
                        log(f"  {label} ids {i + 1}/{len(rows)}  {name}")
    finally:
        # Always hand the scene back exactly as we found it, even on failure.
        base.restore()
        bpy.context.view_layer.update()

    if cfg.write_ids:
        # Annotation is deliberately AFTER all rendering, so a failure here
        # never costs you the renders - press "Annotate Existing IDs" to retry.
        log("Annotating")
        annotate_folder(out, log, seeds=dict(rows))

        # Previews are written automatically so you always have something to
        # eyeball without knowing to press a second button. Never fatal - the
        # dataset is already complete and correct by this point.
        if cfg.auto_preview:
            try:
                write_label_previews(out, log, cfg.preview_count)
            except Exception as exc:
                log(f"Preview step failed ({exc}) - the dataset itself is "
                    f"fine. Press 'Draw Label Previews' to retry.")

    log(f"DONE in {(time.time() - t0) / 60.0:.1f} min")
    log(f"Saved to {out}")
    return out


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class TOWEL_Settings(PropertyGroup):
    output_dir: StringProperty(
        name="Output Folder", subtype='DIR_PATH', default="//dataset/",
        description="Where to save images and annotations")
    count: IntProperty(
        name="Images", default=50, min=1, max=100000,
        description="How many pictures to render")
    start_seed: IntProperty(
        name="Start Seed", default=0, min=0,
        description="Each image uses the next seed. Change this to generate a "
                    "different batch that doesn't repeat the last one")
    engine: EnumProperty(
        name="Engine",
        items=[('BLENDER_EEVEE', "EEVEE (fast)",
                "Seconds per image. Best for large datasets"),
               ('CYCLES', "Cycles (realistic)",
                "Minutes per image. More photoreal")],
        default='BLENDER_EEVEE')
    device: EnumProperty(
        name="Device",
        items=[('CPU', "CPU", "Always works"),
               ('GPU', "GPU", "Faster if your machine supports it")],
        default='CPU', description="Cycles only")
    samples: IntProperty(
        name="Quality", default=64, min=1, max=4096,
        description="Higher is cleaner but slower")
    write_ids: BoolProperty(
        name="Create Annotations", default=True,
        description="Also work out where every towel is and write "
                    "annotations.json. Roughly doubles the render time")
    write_stacks: BoolProperty(
        name="Also Count Stacks", default=True,
        description="As well as labelling each towel, label each whole pile - "
                    "its outline and how many there are. Adds one more render "
                    "pass, roughly a second per image")
    auto_preview: BoolProperty(
        name="Preview After Render", default=True,
        description="Automatically draw label previews when a render "
                    "finishes, so you can check the labels without pressing "
                    "anything else")
    preview_count: IntProperty(
        name="Preview Images", default=6, min=1, max=200,
        description="How many label previews to draw, both automatically "
                    "and when you press Draw Label Previews")

    jitter_camera: BoolProperty(
        name="Camera Shake", default=True,
        description="Nudge the camera slightly on every image, so the set "
                    "looks like photos taken by hand rather than one locked-"
                    "off tripod shot. Your camera is put back afterwards")
    cam_angle: FloatProperty(
        name="Angle (deg)", default=1.5, min=0.0, max=20.0,
        description="Maximum tilt in degrees, applied to pitch, yaw and roll. "
                    "1-3 looks like a steady hand, 5+ like a rushed snapshot")
    cam_shift: FloatProperty(
        name="Position", default=0.03, min=0.0, max=1.0, subtype='DISTANCE',
        description="Maximum sideways/forward movement in metres. Keep this "
                    "small - a few centimetres reads as someone shifting "
                    "their weight")
    jitter_light: BoolProperty(
        name="Light Variation", default=True,
        description="Vary the brightness of your existing lights per image. "
                    "Relative to whatever you set up, and floored so frames "
                    "never go dark")
    light_var: FloatProperty(
        name="Brightness", default=0.25, min=0.0, max=0.6, subtype='FACTOR',
        description="How much the lighting may vary, e.g. 0.25 = plus or "
                    "minus 25%. The dark end is capped at 65% of your setup "
                    "no matter what you put here")


def _run(op, fn, cfg, label):
    """Shared operator body: progress cursor, logging, visible errors."""
    wm = bpy.context.window_manager
    try:
        out = resolve_out(cfg)
    except Exception as exc:
        op.report({'ERROR'}, str(exc))
        return {'CANCELLED'}

    log = Log(out)
    wm.progress_begin(0.0, 1.0)
    try:
        fn(log, wm.progress_update)
        op.report({'INFO'}, f"{label} finished - saved to {out}")
        return {'FINISHED'}
    except Exception as exc:
        log.error(exc)
        # Keep it short in the status bar, full detail in the log file.
        first = str(exc).splitlines()[0][:180]
        op.report({'ERROR'}, f"{first}  (see pipeline_log.txt)")
        return {'CANCELLED'}
    finally:
        wm.progress_end()


class TOWEL_OT_render(Operator):
    bl_idname = "towel.render_dataset"
    bl_label = "Render Dataset"
    bl_description = "Render the images and write the training annotations"

    def execute(self, context):
        cfg = context.scene.towel_settings
        return _run(self, lambda log, prog: render_dataset(
            context.scene, cfg, log, prog), cfg, "Render")


class TOWEL_OT_annotate(Operator):
    bl_idname = "towel.annotate_existing"
    bl_label = "Annotate Existing IDs"
    bl_description = ("Rebuild annotations.json from the ids folder already on "
                      "disk. Use this if rendering finished but the "
                      "annotations are missing - takes seconds")

    def execute(self, context):
        cfg = context.scene.towel_settings
        out = None
        try:
            out = resolve_out(cfg)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        seeds = {}
        seed_csv = os.path.join(out, "seeds.csv")
        if os.path.exists(seed_csv):
            try:
                with open(seed_csv) as fh:
                    for row in csv.DictReader(fh):
                        seeds[row["image"]] = int(row["seed"])
            except Exception:
                pass
        return _run(self, lambda log, prog: annotate_folder(
            out, log, prog, seeds), cfg, "Annotation")


class TOWEL_OT_previews(Operator):
    bl_idname = "towel.label_previews"
    bl_label = "Draw Label Previews"
    bl_description = ("Write a preview folder with the annotations painted "
                      "onto the renders, so you can see that the labels line "
                      "up with the towels. Not part of the training data")

    def execute(self, context):
        cfg = context.scene.towel_settings
        try:
            out = resolve_out(cfg)
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return _run(self, lambda log, prog: write_label_previews(
            out, log, cfg.preview_count, prog), cfg, "Previews")


class TOWEL_OT_check(Operator):
    bl_idname = "towel.check_setup"
    bl_label = "Check Setup"
    bl_description = "Look for problems before committing to a long render"

    def execute(self, context):
        scene = context.scene
        gens = get_generators(scene)
        if not gens:
            self.report({'ERROR'}, "No TowelStacks object - run "
                                   "build_towel_stacks.py first")
            return {'CANCELLED'}
        bpy.context.view_layer.update()
        fatal, n = preflight(scene, gens)
        if fatal:
            for f in fatal:
                print(f"[towel] PROBLEM: {f}")
            self.report({'ERROR'}, fatal[0])
            return {'CANCELLED'}
        self.report({'INFO'}, f"All good - {n} towels, camera framed correctly")
        return {'FINISHED'}


class TOWEL_OT_preview(Operator):
    bl_idname = "towel.preview_one"
    bl_label = "Preview One Image"
    bl_description = ("Render a single image so you can check it before "
                      "starting a long run")

    def execute(self, context):
        scene = context.scene
        cfg = scene.towel_settings
        gens = get_generators(scene)
        if not gens:
            self.report({'ERROR'}, "No TowelStacks object found")
            return {'CANCELLED'}
        configure_engine(scene, cfg.engine, cfg.device, cfg.samples, print)
        ensure_lighting(scene, gens, print)
        for _, mod in gens:
            set_input(mod, "Seed", cfg.start_seed)
        bpy.context.view_layer.update()
        fatal, _ = preflight(scene, gens)
        if fatal:
            self.report({'ERROR'}, fatal[0])
            return {'CANCELLED'}
        bpy.ops.render.render('INVOKE_DEFAULT')
        return {'FINISHED'}


class TOWEL_PT_panel(Panel):
    bl_label = "Towel Dataset"
    bl_idname = "TOWEL_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Towel Data"

    def draw(self, context):
        layout = self.layout
        cfg = context.scene.towel_settings
        gens = get_generators(context.scene)

        box = layout.box()
        if gens:
            box.label(text=f"{len(gens)} generator ready", icon='CHECKMARK')
        else:
            box.label(text="No generator found", icon='ERROR')
            box.label(text="Run build_towel_stacks.py")

        layout.prop(cfg, "output_dir")
        col = layout.column(align=True)
        col.prop(cfg, "count")
        col.prop(cfg, "start_seed")

        layout.separator()
        layout.prop(cfg, "engine")
        row = layout.row()
        row.enabled = cfg.engine == 'CYCLES'
        row.prop(cfg, "device")
        layout.prop(cfg, "samples")
        layout.prop(cfg, "write_ids")
        sub = layout.row()
        sub.enabled = cfg.write_ids
        sub.prop(cfg, "write_stacks")

        layout.separator()
        box = layout.box()
        box.label(text="Realism", icon='SHADERFX')
        box.prop(cfg, "jitter_camera")
        sub = box.column(align=True)
        sub.enabled = cfg.jitter_camera
        sub.prop(cfg, "cam_angle")
        sub.prop(cfg, "cam_shift")
        box.prop(cfg, "jitter_light")
        sub = box.column(align=True)
        sub.enabled = cfg.jitter_light
        sub.prop(cfg, "light_var")
        if cfg.jitter_light:
            lo = max(1.0 - cfg.light_var, LIGHT_FLOOR)
            sub.label(text=f"Range: {lo * 100:.0f}% - "
                           f"{(1 + cfg.light_var) * 100:.0f}% of your lights")

        layout.separator()
        layout.operator("towel.check_setup", icon='VIEWZOOM')
        layout.operator("towel.preview_one", icon='RENDER_STILL')
        row = layout.row()
        row.scale_y = 1.6
        row.operator("towel.render_dataset", icon='RENDER_ANIMATION')

        layout.separator()
        box = layout.box()
        box.label(text="Check the labels", icon='SEQ_PREVIEW')
        box.prop(cfg, "auto_preview")
        box.prop(cfg, "preview_count")
        box.operator("towel.label_previews", icon='IMAGE_RGB_ALPHA')
        box.operator("towel.annotate_existing", icon='FILE_REFRESH')

        est = cfg.count * (2.5 if cfg.engine == 'BLENDER_EEVEE' else 45.0)
        if cfg.write_ids:
            est += cfg.count * 1.5
        layout.label(text=f"Rough estimate: {est / 60:.0f} min")
        if cfg.count > 20:
            layout.label(text="Blender will be busy until done", icon='INFO')


CLASSES = (TOWEL_Settings, TOWEL_OT_render, TOWEL_OT_annotate,
           TOWEL_OT_previews, TOWEL_OT_check, TOWEL_OT_preview, TOWEL_PT_panel)


def register():
    for c in CLASSES:
        try:
            bpy.utils.register_class(c)
        except ValueError:
            bpy.utils.unregister_class(c)
            bpy.utils.register_class(c)
    bpy.types.Scene.towel_settings = PointerProperty(type=TOWEL_Settings)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except RuntimeError:
            pass
    if hasattr(bpy.types.Scene, "towel_settings"):
        del bpy.types.Scene.towel_settings


if __name__ == "__main__":
    register()
    if "--" in sys.argv:
        # blender -b file.blend --python towel_pipeline.py -- \
        #     --out ./dataset --count 300 [--engine cycles] [--no-ids]
        args = sys.argv[sys.argv.index("--") + 1:]
        cfg = bpy.context.scene.towel_settings
        for i, a in enumerate(args):
            if a == "--out" and i + 1 < len(args):
                cfg.output_dir = args[i + 1]
            elif a == "--count" and i + 1 < len(args):
                cfg.count = int(args[i + 1])
            elif a == "--start" and i + 1 < len(args):
                cfg.start_seed = int(args[i + 1])
            elif a == "--samples" and i + 1 < len(args):
                cfg.samples = int(args[i + 1])
            elif a == "--engine" and i + 1 < len(args):
                cfg.engine = ('CYCLES' if args[i + 1].lower() == "cycles"
                              else 'BLENDER_EEVEE')
            elif a == "--device" and i + 1 < len(args):
                cfg.device = args[i + 1].upper()
            elif a == "--no-ids":
                cfg.write_ids = False
        log = Log(resolve_out(cfg))
        render_dataset(bpy.context.scene, cfg, log)
    else:
        print("[towel] Panel registered. Press N in the 3D view and open the "
              "'Towel Data' tab.")

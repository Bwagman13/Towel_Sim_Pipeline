"""
build_towel_stacks.py  -  TowelStackGenerator for Blender 4.0 / 4.1

Builds an editable Geometry Nodes group that scatters folded-linen OBJ
instances into randomised piles across a shelf grid, for synthetic CV data.

Graph outline
-------------
  Mesh Grid (slot lattice)  ->  Mesh to Points
        |                              |
  Mesh Line (shelf tiers) --> Instance on Points --> Realize Instances
        |
  Store "slot_id" (which pile) and "shelf_variant" (what this shelf holds)
        |
  Duplicate Elements (POINT)
     Selection = Random Bool(Fill Probability, ID = slot_id)
     Amount    = Random Int(Min/Max Towels,    ID = slot_id), capped by headroom
        |
  resolve this towel's variant -> its own measured thickness
        |
  Accumulate Field (prefix sum of heights within the pile) -> Z
  Set Position (+ pile and per-towel XY jitter)
        |
  Instance on Points (Collection Info, Pick Instance = resolved variant)
        |
  Rotate -> Scale -> Store "inst_color" + "stack_color" -> Group Output

Per-variant thickness
---------------------
Each source object's real height is measured on build and baked into the node
graph as a comparison chain, so a 45 mm rag and a 90 mm sheet stack correctly
in the same scene with no manual setup. Re-run this script after adding or
swapping objects and the chain updates itself.

Every random stream is driven by one master `Seed` plus a fixed per-stream
offset, so a single integer reproduces an entire scene exactly.

Usage
-----
  # inside Blender: paste into the Scripting tab's TEXT EDITOR, press Run
  # headless:
  blender --background --python build_towel_stacks.py -- \
      --towels /path/to/folder_of_obj_files \
      --out    /path/to/towel_generator.blend
"""

import bpy
import os
import sys

GROUP_NAME = "TowelStackGenerator"
GEN_OBJ_NAME = "TowelStacks"

# ---- EDIT THIS if your towels already live in a collection --------------
# Set it to the exact name of your existing collection (case-sensitive).
# If the collection doesn't exist yet it will be created.
TOWEL_COLLECTION = "TowelSourceObjects"

# Quantisation levels per colour channel for the ID passes.
# 16 -> 4095 addressable ids with +/-4 units of 8-bit noise tolerance.
# Must match ID_LEVELS in towel_pipeline.py / coco_from_masks.py.
ID_LEVELS = 16

# Move the source towels out of the way and hide them from renders.
PARK_SOURCES = True

# Apply each source object's rotation/scale into its mesh before use.
# Needed because Collection Info's "Reset Children" discards object
# transforms - an unapplied 90-degree rotation would silently vanish.
APPLY_SOURCE_TRANSFORMS = True


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sock_in(node, name, socket_type):
    """Find an input socket by (name, type).

    Nodes like Random Value and Store Named Attribute declare several sockets
    that share a name ("Min", "Max", "Value") and differ only by data type.
    Indexing numerically is brittle across Blender versions; this is not.
    Socket .type values: 'VALUE' (float), 'INT', 'BOOLEAN', 'VECTOR', 'RGBA'.
    """
    for s in node.inputs:
        if s.name == name and s.type == socket_type:
            return s
    raise KeyError(f"input '{name}' ({socket_type}) not found on {node.bl_idname}")


def sock_out(node, name, socket_type):
    for s in node.outputs:
        if s.name == name and s.type == socket_type:
            return s
    raise KeyError(f"output '{name}' ({socket_type}) not found on {node.bl_idname}")


def sock_in_any(node, names, socket_type):
    """Accumulate Field's grouping input is 'Group Index' on 3.x and
    'Group ID' on 4.x; accept either rather than pinning one version."""
    for name in names:
        for s in node.inputs:
            if s.name == name and s.type == socket_type:
                return s
    raise KeyError(f"none of {names} ({socket_type}) on {node.bl_idname}")


class Builder:
    """Thin wrapper so node creation and linking stay readable."""

    def __init__(self, node_tree):
        self.nt = node_tree

    def new(self, bl_idname, location=(0, 0), label=None, **props):
        n = self.nt.nodes.new(bl_idname)
        n.location = location
        if label:
            n.label = label
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, from_socket, to_socket):
        return self.nt.links.new(from_socket, to_socket)

    def feed(self, socket, value):
        """Link if `value` is a socket, otherwise write it as a default."""
        if hasattr(value, "is_output"):
            self.link(value, socket)
        else:
            socket.default_value = value

    def math(self, operation, a=None, b=None, c=None, location=(0, 0), label=None):
        n = self.new("ShaderNodeMath", location=location, label=label)
        n.operation = operation
        for i, v in enumerate((a, b, c)):
            if v is not None:
                self.feed(n.inputs[i], v)
        return n.outputs[0]

    def vmath(self, operation, a=None, b=None, location=(0, 0), label=None):
        n = self.new("ShaderNodeVectorMath", location=location, label=label)
        n.operation = operation
        for i, v in enumerate((a, b)):
            if v is not None:
                self.feed(n.inputs[i], v)
        return n.outputs[0]

    def combine_xyz(self, x=0.0, y=0.0, z=0.0, location=(0, 0), label=None):
        n = self.new("ShaderNodeCombineXYZ", location=location, label=label)
        for socket, v in zip(n.inputs, (x, y, z)):
            self.feed(socket, v)
        return n.outputs[0]

    def lerp(self, a, bb, t, location=(0, 0), label=None):
        """a + (bb - a) * t.

        With t exactly 0 or 1 this is an exact select, which is why it stands
        in for a Switch node - Switch's socket layout shifted between 4.0,
        4.1 and 4.2.
        """
        return self.math('ADD', a,
                         self.math('MULTIPLY',
                                   self.math('SUBTRACT', bb, a,
                                             location=(location[0] - 320,
                                                       location[1])),
                                   t, location=(location[0] - 160, location[1])),
                         location=location, label=label)

    # --- random value variants ---------------------------------------------

    def _random(self, data_type, location, label, seed, id_socket):
        n = self.new("FunctionNodeRandomValue", location=location, label=label)
        n.data_type = data_type
        if id_socket is not None:
            self.link(id_socket, sock_in(n, "ID", "INT"))
        self.feed(sock_in(n, "Seed", "INT"), seed)
        return n

    def rand_float(self, lo, hi, seed, id_socket=None, location=(0, 0), label=None):
        n = self._random("FLOAT", location, label, seed, id_socket)
        self.feed(sock_in(n, "Min", "VALUE"), lo)
        self.feed(sock_in(n, "Max", "VALUE"), hi)
        return sock_out(n, "Value", "VALUE")

    def rand_int(self, lo, hi, seed, id_socket=None, location=(0, 0), label=None):
        n = self._random("INT", location, label, seed, id_socket)
        self.feed(sock_in(n, "Min", "INT"), lo)
        self.feed(sock_in(n, "Max", "INT"), hi)
        return sock_out(n, "Value", "INT")

    def rand_vector(self, lo, hi, seed, id_socket=None, location=(0, 0), label=None):
        n = self._random("FLOAT_VECTOR", location, label, seed, id_socket)
        self.feed(sock_in(n, "Min", "VECTOR"), lo)
        self.feed(sock_in(n, "Max", "VECTOR"), hi)
        return sock_out(n, "Value", "VECTOR")

    def rand_bool(self, probability, seed, id_socket=None, location=(0, 0), label=None):
        n = self._random("BOOLEAN", location, label, seed, id_socket)
        self.feed(sock_in(n, "Probability", "VALUE"), probability)
        return sock_out(n, "Value", "BOOLEAN")


# ---------------------------------------------------------------------------
# node group parameters
# ---------------------------------------------------------------------------
# (name, socket_type, default, min, max, subtype, description)
PARAMS = [
    ("Towel Collection", "NodeSocketCollection", None, None, None, "NONE",
     "Collection holding the folded-linen source objects"),
    ("Towel Variants", "NodeSocketInt", 4, 1, 64, "NONE",
     "How many objects are in the collection (drives random variant pick)"),
    ("Variant Layout", "NodeSocketInt", 2, 0, 2, "NONE",
     "0 = Mixed (any towel in a pile may be a different object). "
     "1 = Per pile (each pile is one object type). "
     "2 = Per shelf (each shelf is one object type). All re-roll on Seed"),
    ("Cycle Variants By Shelf", "NodeSocketBool", True, None, None, "NONE",
     "ON: shelves cycle through the variants (0,1,0,1...) with a per-seed "
     "rotation, so every object type is guaranteed to appear. "
     "OFF: each shelf draws independently. Only used when Variant Layout = 2"),
    ("Seed", "NodeSocketInt", 0, 0, 2 ** 31 - 1, "NONE",
     "Master seed - one integer reproduces the whole scene"),
    ("ID Offset", "NodeSocketInt", 0, 0, 4000, "NONE",
     "Added to the IDs in the segmentation passes. Only matters if you run "
     "two generator objects at once - each needs its own offset or their "
     "masks collide. towel_pipeline.py sets this automatically"),

    ("Slots X", "NodeSocketInt", 5, 2, 64, "NONE",
     "Piles across the shelf width. This is exactly how many you get"),
    ("Slots Y", "NodeSocketInt", 2, 2, 64, "NONE",
     "Piles across the shelf depth. This is exactly how many you get"),
    ("Slot Spacing X", "NodeSocketFloat", 0.55, 0.01, 10.0, "DISTANCE",
     "Distance between pile centres along X"),
    ("Slot Spacing Y", "NodeSocketFloat", 0.45, 0.01, 10.0, "DISTANCE",
     "Distance between pile centres along Y"),
    ("Shelf Count", "NodeSocketInt", 3, 1, 32, "NONE",
     "Number of vertically stacked shelves"),
    ("Shelf Spacing Z", "NodeSocketFloat", 0.62, 0.01, 10.0, "DISTANCE",
     "Vertical distance between shelves"),

    ("Fill Probability", "NodeSocketFloat", 0.85, 0.0, 1.0, "FACTOR",
     "Chance a given slot contains a pile at all (creates empty gaps)"),
    ("Min Towels", "NodeSocketInt", 2, 1, 200, "NONE",
     "Minimum towels per pile"),
    ("Max Towels", "NodeSocketInt", 8, 1, 200, "NONE",
     "Maximum towels per pile (further capped by shelf headroom)"),
    ("Shelf Headroom", "NodeSocketFloat", 0.9, 0.1, 2.0, "FACTOR",
     "Fraction of the shelf gap a pile may fill; caps Max Towels so "
     "piles cannot clip through the shelf above"),

    ("Thickness Scale", "NodeSocketFloat", 1.0, 0.1, 5.0, "FACTOR",
     "Global multiplier on every item's measured height. Leave at 1.0 - "
     "raise it only if piles look too tightly packed, lower it if they "
     "look gappy"),
    ("Thickness Jitter", "NodeSocketFloat", 0.012, 0.0, 1.0, "DISTANCE",
     "Per-towel variation in rise; accumulates up the pile"),
    ("Pile XY Jitter", "NodeSocketFloat", 0.05, 0.0, 2.0, "DISTANCE",
     "How far a whole pile may drift from its slot centre"),
    ("Towel XY Jitter", "NodeSocketFloat", 0.018, 0.0, 2.0, "DISTANCE",
     "How far an individual towel slides within its pile"),

    ("Z Rotation Range", "NodeSocketFloat", 9.0, 0.0, 180.0, "NONE",
     "Max +/- yaw per towel, in degrees - the main 'untidy stack' cue"),
    ("Tilt Range", "NodeSocketFloat", 2.5, 0.0, 45.0, "NONE",
     "Max +/- pitch/roll per towel, in degrees"),
    ("Scale Jitter", "NodeSocketFloat", 0.04, 0.0, 0.5, "FACTOR",
     "Uniform scale variation, e.g. 0.04 = +/-4%"),
]

# Fixed per-stream seed offsets. Distinct values keep the random streams
# decorrelated; changing them reshuffles everything, so leave them alone.
S_FILL, S_COUNT, S_THICK = 101, 211, 331
S_PILE_XY, S_TOWEL_XY = 439, 547
S_VARIANT, S_ROT, S_SCALE = 653, 761, 877
S_SHELF_VAR, S_SHELF_ROT = 983, 1091


def build_node_group(thicknesses=None):
    """Create (or rebuild) the node group.

    `thicknesses` is one measured height per variant, in collection order.
    It is baked into the graph as a comparison chain, which is how per-object
    stacking works without an array-lookup node (Geometry Nodes has none in
    4.0/4.1 - Index Switch only arrived in 4.2).
    """
    thicknesses = list(thicknesses) if thicknesses else [0.075]

    old = bpy.data.node_groups.get(GROUP_NAME)
    if old:
        bpy.data.node_groups.remove(old)

    ng = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
    b = Builder(ng)
    lerp = b.lerp

    # --- interface ---------------------------------------------------------
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT',
                            socket_type='NodeSocketGeometry')
    # Unused, but the modifier UI expects a geometry input to exist.
    ng.interface.new_socket(name="Geometry", in_out='INPUT',
                            socket_type='NodeSocketGeometry')

    for name, stype, default, lo, hi, subtype, desc in PARAMS:
        s = ng.interface.new_socket(name=name, in_out='INPUT',
                                    socket_type=stype, description=desc)
        if default is not None:
            s.default_value = default
        if lo is not None and hasattr(s, "min_value"):
            s.min_value = lo
        if hi is not None and hasattr(s, "max_value"):
            s.max_value = hi
        if subtype != "NONE" and hasattr(s, "subtype"):
            s.subtype = subtype

    gin = b.new("NodeGroupInput", location=(-1600, 0))
    gout = b.new("NodeGroupOutput", location=(3600, 250))
    P = {s.name: s for s in gin.outputs if s.name}

    seed = P["Seed"]

    def seeded(offset, y):
        """Seed + constant, so each random stream is independent."""
        return b.math('ADD', seed, float(offset), location=(-1400, y),
                      label=f"seed +{offset}")

    layout = P["Variant Layout"]
    is_mixed = b.math('COMPARE', layout, 0.0, 0.5, location=(-1400, -700),
                      label="layout = mixed")
    is_pile = b.math('COMPARE', layout, 1.0, 0.5, location=(-1400, -780),
                     label="layout = per pile")
    is_shelf = b.math('COMPARE', layout, 2.0, 0.5, location=(-1400, -860),
                      label="layout = per shelf")

    # --- 1. slot lattice ---------------------------------------------------
    # Spacing, not extent, is the user-facing knob, so Slots X/Y map one-to-one
    # onto the piles you actually get. No trimming, no second lattice.
    size_x = b.math('MULTIPLY', b.math('SUBTRACT', P["Slots X"], 1.0,
                                       location=(-1240, 500)),
                    P["Slot Spacing X"], location=(-1080, 500), label="width")
    size_y = b.math('MULTIPLY', b.math('SUBTRACT', P["Slots Y"], 1.0,
                                       location=(-1240, 400)),
                    P["Slot Spacing Y"], location=(-1080, 400), label="depth")

    grid = b.new("GeometryNodeMeshGrid", location=(-900, 450),
                 label="Slot lattice")
    b.link(size_x, grid.inputs["Size X"])
    b.link(size_y, grid.inputs["Size Y"])
    b.link(P["Slots X"], grid.inputs["Vertices X"])
    b.link(P["Slots Y"], grid.inputs["Vertices Y"])

    grid_pts = b.new("GeometryNodeMeshToPoints", location=(-720, 450))
    grid_pts.mode = 'VERTICES'
    b.link(grid.outputs["Mesh"], grid_pts.inputs["Mesh"])
    grid_pts.inputs["Radius"].default_value = 0.01

    # --- 2. stack the lattice into shelf tiers -----------------------------
    tiers = b.new("GeometryNodeMeshLine", location=(-900, 150),
                  label="Shelf tiers")
    tiers.mode = 'OFFSET'
    tiers.count_mode = 'TOTAL'
    b.link(P["Shelf Count"], tiers.inputs["Count"])
    b.link(b.combine_xyz(0.0, 0.0, P["Shelf Spacing Z"], location=(-1080, 110)),
           tiers.inputs["Offset"])

    shelf_iop = b.new("GeometryNodeInstanceOnPoints", location=(-540, 300),
                      label="Grid per shelf")
    b.link(tiers.outputs["Mesh"], shelf_iop.inputs["Points"])
    b.link(grid_pts.outputs["Points"], shelf_iop.inputs["Instance"])

    realize = b.new("GeometryNodeRealizeInstances", location=(-360, 300))
    b.link(shelf_iop.outputs["Instances"], realize.inputs[0])

    # --- 3. which object does each shelf hold? -----------------------------
    pos = b.new("GeometryNodeInputPosition", location=(-540, -100))
    sep = b.new("ShaderNodeSeparateXYZ", location=(-380, -100))
    b.link(pos.outputs["Position"], sep.inputs["Vector"])
    shelf = b.math('ROUND', b.math('DIVIDE', sep.outputs["Z"],
                                   P["Shelf Spacing Z"], location=(-220, -100)),
                   location=(-60, -100), label="shelf index")

    # Independent draw per shelf - can legitimately give every shelf the same
    # object, which is realistic but not always what you want.
    shelf_rand = b.rand_int(0, b.math('SUBTRACT', P["Towel Variants"], 1.0,
                                      location=(-380, -240)),
                            seeded(S_SHELF_VAR, -300), shelf,
                            location=(-60, -260), label="random per shelf")

    # Cycled draw: (shelf + per-seed rotation) mod Variants. Guarantees every
    # variant appears while still re-rolling on Seed. The rotation uses the
    # default ID of 0 so it is one value for the whole scene, not per point.
    rotation = b.rand_int(0, b.math('SUBTRACT', P["Towel Variants"], 1.0,
                                    location=(-380, -400)),
                          seeded(S_SHELF_ROT, -460), None,
                          location=(-220, -420), label="per-seed rotation")
    shelf_cycled = b.math('MODULO',
                          b.math('ADD', shelf, rotation, location=(-60, -400)),
                          P["Towel Variants"], location=(100, -400),
                          label="cycled per shelf")

    shelf_variant = lerp(shelf_rand, shelf_cycled, P["Cycle Variants By Shelf"],
                         location=(580, -260), label="variant for shelf")

    # --- 4. tag every slot -------------------------------------------------
    # slot_id keeps a pile coherent: all its towels share one random draw for
    # the pile's drift, and it is what the stack segmentation pass colours by.
    # shelf_variant is baked here because the stacking Set Position later
    # changes Z, so it could not be recovered from position downstream.
    index_a = b.new("GeometryNodeInputIndex", location=(760, 100))
    store = b.new("GeometryNodeStoreNamedAttribute", location=(940, 300),
                  label="slot_id")
    store.data_type = 'INT'
    store.domain = 'POINT'
    b.link(realize.outputs[0], store.inputs["Geometry"])
    store.inputs["Name"].default_value = "slot_id"
    b.link(index_a.outputs["Index"], sock_in(store, "Value", "INT"))

    store_sv = b.new("GeometryNodeStoreNamedAttribute", location=(1120, 300),
                     label="shelf_variant")
    store_sv.data_type = 'INT'
    store_sv.domain = 'POINT'
    b.link(store.outputs["Geometry"], store_sv.inputs["Geometry"])
    store_sv.inputs["Name"].default_value = "shelf_variant"
    b.link(shelf_variant, sock_in(store_sv, "Value", "INT"))

    slot_attr = b.new("GeometryNodeInputNamedAttribute", location=(1120, 60),
                      label="read slot_id")
    slot_attr.data_type = 'INT'
    slot_attr.inputs["Name"].default_value = "slot_id"
    slot_id = sock_out(slot_attr, "Attribute", "INT")

    sv_attr = b.new("GeometryNodeInputNamedAttribute", location=(1120, -40),
                    label="read shelf_variant")
    sv_attr.data_type = 'INT'
    sv_attr.inputs["Name"].default_value = "shelf_variant"
    shelf_variant_ds = sock_out(sv_attr, "Attribute", "INT")

    # --- 5. one point per towel -------------------------------------------
    # Duplicate Elements is what lets each slot have a *different* count;
    # Instance on Points alone cannot vary its count per point.
    dup = b.new("GeometryNodeDuplicateElements", location=(1400, 300),
                label="Pile height")
    dup.domain = 'POINT'
    b.link(store_sv.outputs["Geometry"], dup.inputs["Geometry"])
    b.link(b.rand_bool(P["Fill Probability"], seeded(S_FILL, -140), slot_id,
                       location=(1120, -160), label="slot occupied?"),
           dup.inputs["Selection"])

    # Cap the pile so it cannot grow through the shelf above. Divide by the
    # tallest item, since a pile could hold any variant - conservative, but it
    # guarantees nothing clips through.
    tallest = max(thicknesses)
    fit = b.math('FLOOR',
                 b.math('DIVIDE',
                        b.math('MULTIPLY', P["Shelf Spacing Z"],
                               P["Shelf Headroom"], location=(1120, -520)),
                        b.math('MULTIPLY', tallest, P["Thickness Scale"],
                               location=(1120, -600)),
                        location=(1280, -540)),
                 location=(1440, -540), label="towels that fit")
    capped_max = b.math('MINIMUM', P["Max Towels"], fit,
                        location=(1600, -460), label="effective max")
    b.link(b.rand_int(P["Min Towels"],
                      b.math('MAXIMUM', capped_max, 1.0, location=(1760, -460)),
                      seeded(S_COUNT, -340), slot_id,
                      location=(1920, -420), label="towels in pile"),
           dup.inputs["Amount"])

    index_b = b.new("GeometryNodeInputIndex", location=(1400, 60),
                    label="per-towel id")
    towel_id = index_b.outputs["Index"]

    # --- 5b. which object is this towel, and how thick is it? --------------
    # Which ID feeds the draw is the whole mechanism:
    #   layout 0 (mixed)     -> ID = per-towel Index -> every towel rolls its own
    #   layout 1 (per pile)  -> ID = slot_id         -> pile shares one roll
    #   layout 2 (per shelf) -> take the shelf's own draw directly
    variant_id = b.math(
        'ADD',
        b.math('MULTIPLY', towel_id, is_mixed, location=(1400, -860)),
        b.math('MULTIPLY', slot_id, is_pile, location=(1400, -940)),
        location=(1560, -900), label="variant ID source")

    rolled = b.rand_int(0, b.math('SUBTRACT', P["Towel Variants"], 1.0,
                                  location=(1400, -1040)),
                        seeded(S_VARIANT, -1000), variant_id,
                        location=(1720, -960), label="which object")
    final_variant = lerp(rolled, shelf_variant_ds, is_shelf,
                         location=(2200, -960), label="final variant")

    # Per-variant thickness as a comparison chain:
    #     t(v) = t0 + sum_k (t_k - t_{k-1}) * (v >= k)
    # Each term switches on only once the variant index reaches that object,
    # so the running total lands on exactly that object's measured height.
    # Values are baked from the meshes at build time; re-run to refresh.
    thick = thicknesses[0]
    for k in range(1, len(thicknesses)):
        delta = thicknesses[k] - thicknesses[k - 1]
        if abs(delta) < 1e-9:
            continue
        reached = b.math('GREATER_THAN', final_variant, k - 0.5,
                         location=(2360, -900 - k * 90),
                         label=f"is variant >= {k}")
        thick = b.math('ADD', thick,
                       b.math('MULTIPLY', delta, reached,
                              location=(2520, -900 - k * 90)),
                       location=(2680, -900 - k * 90),
                       label=f"+ variant {k} height")
    towel_thickness = b.math('MULTIPLY', thick, P["Thickness Scale"],
                             location=(2840, -960), label="height of this item")

    # --- 6. position each towel -------------------------------------------
    # Each towel contributes its own height, so a pile mixing a 45 mm rag with
    # a 90 mm sheet stacks correctly. Accumulate Field's Trailing output is the
    # exclusive prefix sum within each group, i.e. the summed height of every
    # towel below this one in the same pile - which is exactly its Z.
    per_towel_rise = b.math(
        'ADD', towel_thickness,
        b.rand_float(b.math('MULTIPLY', P["Thickness Jitter"], -1.0,
                            location=(2360, -700)),
                     P["Thickness Jitter"], seeded(S_THICK, -640),
                     towel_id, location=(2680, -640), label="rise jitter"),
        location=(3000, -740), label="this towel's height")

    accum = b.new("GeometryNodeAccumulateField", location=(3160, -640),
                  label="height of towels below")
    accum.data_type = 'FLOAT'
    accum.domain = 'POINT'
    b.link(per_towel_rise, sock_in(accum, "Value", "VALUE"))
    b.link(slot_id, sock_in_any(accum, ("Group ID", "Group Index"), "INT"))
    z_offset = sock_out(accum, "Trailing", "VALUE")

    def sym_vec(param, y):
        """(-p,-p,0) .. (p,p,0) min/max pair for planar jitter."""
        neg = b.math('MULTIPLY', param, -1.0, location=(1560, y - 60))
        return (b.combine_xyz(neg, neg, 0.0, location=(1740, y - 60)),
                b.combine_xyz(param, param, 0.0, location=(1740, y)))

    pmin, pmax = sym_vec(P["Pile XY Jitter"], 200)
    pile_off = b.rand_vector(pmin, pmax, seeded(S_PILE_XY, 60), slot_id,
                             location=(1920, 140), label="pile drift")

    tmin, tmax = sym_vec(P["Towel XY Jitter"], -20)
    towel_off = b.rand_vector(tmin, tmax, seeded(S_TOWEL_XY, -160), towel_id,
                              location=(1920, -60), label="towel slide")

    offset = b.vmath('ADD',
                     b.vmath('ADD', pile_off, towel_off, location=(3160, 60)),
                     b.combine_xyz(0.0, 0.0, z_offset, location=(3340, -200)),
                     location=(3500, 60), label="total offset")

    setpos = b.new("GeometryNodeSetPosition", location=(3660, 300))
    b.link(dup.outputs["Geometry"], setpos.inputs["Geometry"])
    b.link(offset, setpos.inputs["Offset"])

    # --- 7. instance the resolved variant ----------------------------------
    coll = b.new("GeometryNodeCollectionInfo", location=(3660, 100),
                 label="Towel variants")
    coll.transform_space = 'ORIGINAL'
    b.link(P["Towel Collection"], coll.inputs["Collection"])
    coll.inputs["Separate Children"].default_value = True
    coll.inputs["Reset Children"].default_value = True

    iop = b.new("GeometryNodeInstanceOnPoints", location=(3860, 300))
    b.link(setpos.outputs["Geometry"], iop.inputs["Points"])
    b.link(coll.outputs["Instances"], iop.inputs["Instance"])
    iop.inputs["Pick Instance"].default_value = True
    # Reusing the same field guarantees the object placed is the one whose
    # thickness was used to position it.
    b.link(final_variant, iop.inputs["Instance Index"])

    # --- 8. rotation & scale ------------------------------------------------
    zr = b.math('RADIANS', P["Z Rotation Range"], location=(3340, 620))
    tr = b.math('RADIANS', P["Tilt Range"], location=(3340, 540))
    zr_n = b.math('MULTIPLY', zr, -1.0, location=(3500, 620))
    tr_n = b.math('MULTIPLY', tr, -1.0, location=(3500, 540))

    rot = b.rand_vector(b.combine_xyz(tr_n, tr_n, zr_n, location=(3680, 580)),
                        b.combine_xyz(tr, tr, zr, location=(3680, 680)),
                        seeded(S_ROT, 660), towel_id,
                        location=(3840, 640), label="lean + yaw")

    rotate = b.new("GeometryNodeRotateInstances", location=(4060, 300))
    b.link(iop.outputs["Instances"], rotate.inputs["Instances"])
    b.link(rot, rotate.inputs["Rotation"])
    rotate.inputs["Local Space"].default_value = True

    scale = b.new("GeometryNodeScaleInstances", location=(4220, 300))
    b.link(rotate.outputs["Instances"], scale.inputs["Instances"])
    b.link(b.rand_float(b.math('SUBTRACT', 1.0, P["Scale Jitter"],
                               location=(3860, -180)),
                        b.math('ADD', 1.0, P["Scale Jitter"],
                               location=(3860, -280)),
                        seeded(S_SCALE, -340), towel_id,
                        location=(4020, -240), label="size variation"),
           scale.inputs["Scale"])

    # --- 9. ID colours for the segmentation passes -------------------------
    # Two attributes, each encoding an integer into an RGB triple at
    # ID_LEVELS steps per channel:
    #   inst_color  - unique per towel  -> per-object masks and counts
    #   stack_color - shared per pile   -> per-stack masks and counts
    # Rendered flat through an emission override, each decodes back to an
    # exact integer. +1 because 0 is reserved for background.
    L = float(ID_LEVELS)

    def encode(value_socket, x, label):
        """integer -> RGB at ID_LEVELS steps per channel"""
        def channel(divisor, y):
            v = value_socket if divisor == 1 else b.math(
                'FLOOR', b.math('DIVIDE', value_socket, float(divisor),
                                location=(x, y)),
                location=(x + 140, y))
            return b.math('DIVIDE', b.math('MODULO', v, L, location=(x + 280, y)),
                          L - 1.0, location=(x + 420, y))
        return b.combine_xyz(channel(ID_LEVELS * ID_LEVELS, -160),
                             channel(ID_LEVELS, -300),
                             channel(1, -440),
                             location=(x + 580, -300), label=label)

    inst_idx = b.new("GeometryNodeInputIndex", location=(4220, -220),
                     label="instance index")
    inst_id1 = b.math('ADD',
                      b.math('ADD', inst_idx.outputs["Index"], 1.0,
                             location=(4380, -120)),
                      P["ID Offset"], location=(4380, -200), label="towel id")
    inst_color = encode(inst_id1, 4540, "towel ID colour")

    # slot_id rides along from the points onto the instances, so every towel
    # in a pile carries the same stack number.
    stack_id1 = b.math('ADD',
                       b.math('ADD', slot_id, 1.0, location=(4380, -1120)),
                       P["ID Offset"], location=(4380, -1200), label="stack id")
    stack_color = encode(stack_id1, 4540, "stack ID colour")

    store_inst = b.new("GeometryNodeStoreNamedAttribute", location=(5300, 300),
                       label="inst_color")
    store_inst.data_type = 'FLOAT_COLOR'
    store_inst.domain = 'INSTANCE'
    b.link(scale.outputs["Instances"], store_inst.inputs["Geometry"])
    store_inst.inputs["Name"].default_value = "inst_color"
    b.link(inst_color, sock_in(store_inst, "Value", "RGBA"))

    store_stack = b.new("GeometryNodeStoreNamedAttribute", location=(5480, 300),
                        label="stack_color")
    store_stack.data_type = 'FLOAT_COLOR'
    store_stack.domain = 'INSTANCE'
    b.link(store_inst.outputs["Geometry"], store_stack.inputs["Geometry"])
    store_stack.inputs["Name"].default_value = "stack_color"
    b.link(stack_color, sock_in(store_stack, "Value", "RGBA"))

    gout.location = (5680, 300)
    b.link(store_stack.outputs["Geometry"], gout.inputs["Geometry"])
    return ng


# ---------------------------------------------------------------------------
# scene assembly
# ---------------------------------------------------------------------------

def socket_identifier(ng, name):
    for item in ng.interface.items_tree:
        if getattr(item, "item_type", "") == 'SOCKET' \
                and item.in_out == 'INPUT' and item.name == name:
            return item.identifier
    raise KeyError(f"no input socket named '{name}'")


def set_modifier_input(mod, ng, name, value):
    mod[socket_identifier(ng, name)] = value


def import_towels(folder):
    """Import every .obj in `folder` into the source collection.

    With no folder, whatever is already in the collection is used untouched -
    so running this on a file where you've placed towels by hand is safe.
    """
    coll = bpy.data.collections.get(TOWEL_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(TOWEL_COLLECTION)
        bpy.context.scene.collection.children.link(coll)
        print(f"[towels] created collection '{TOWEL_COLLECTION}'")

    if not folder:
        n = len([o for o in coll.objects if o.type == 'MESH'])
        print(f"[towels] using {n} object(s) already in '{TOWEL_COLLECTION}'")
        if n == 0:
            print(f"[towels] WARNING: '{TOWEL_COLLECTION}' is empty. Put your "
                  f"towel object(s) in it, or set TOWEL_COLLECTION at the top "
                  f"of this script to the name of the collection you're using.")
        return coll

    if not os.path.isdir(folder):
        raise SystemExit(f"[towels] not a folder: {folder}")

    for obj in list(coll.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".obj"))
    for fname in files:
        before = set(bpy.context.scene.objects)
        bpy.ops.wm.obj_import(filepath=os.path.join(folder, fname))
        for obj in set(bpy.context.scene.objects) - before:
            for c in list(obj.users_collection):
                c.objects.unlink(obj)
            coll.objects.link(obj)

    print(f"[towels] imported {len(coll.objects)} object(s) from {folder}")
    return coll


def normalise_towel_origins(coll):
    """Bake object transforms into the mesh, then put the origin at the
    centre of the footprint with the base at Z = 0.

    Two problems solved here:
      1. Collection Info's "Reset Children" throws away object-level rotation
         and scale, so an OBJ needing a 90-degree flip to lie flat would
         instance standing on its edge.
      2. An arbitrary export pivot makes the item float or sink, and its
         measured thickness stops matching the visible gap.

    Idempotent - running it twice changes nothing the second time.
    """
    done = set()
    for obj in coll.objects:
        if obj.type != 'MESH':
            continue
        mesh = obj.data
        if not mesh.vertices or mesh.name in done:
            continue
        done.add(mesh.name)     # shared mesh data must only be baked once

        if APPLY_SOURCE_TRANSFORMS:
            mesh.transform(obj.matrix_basis)
            obj.matrix_basis.identity()

        xs = [v.co.x for v in mesh.vertices]
        ys = [v.co.y for v in mesh.vertices]
        zs = [v.co.z for v in mesh.vertices]
        shift = ((min(xs) + max(xs)) / 2.0,
                 (min(ys) + max(ys)) / 2.0,
                 min(zs))
        for v in mesh.vertices:
            v.co.x -= shift[0]
            v.co.y -= shift[1]
            v.co.z -= shift[2]
        mesh.update()
        obj.location = (0.0, 0.0, 0.0)


def measure_variants(coll):
    """(name, width, depth, height) per variant, in collection order.

    Collection order is variant order: Collection Info with Separate Children
    hands its children out in the same sequence.
    """
    out = []
    for obj in coll.objects:
        if obj.type != 'MESH' or not obj.data.vertices:
            continue
        vs = obj.data.vertices
        xs = [v.co.x for v in vs]
        ys = [v.co.y for v in vs]
        zs = [v.co.z for v in vs]
        out.append((obj, max(xs) - min(xs), max(ys) - min(ys),
                    max(zs) - min(zs)))
    return out


def report_variants(coll):
    """Print each source object's footprint and flag misalignment.

    The classic failure: you add an OBJ to the collection by hand after
    running this script, so it never gets origin-normalised. Its mesh still
    sits wherever the OBJ file put it, and that whole variant renders as a
    cluster offset from everything else.
    """
    variants = measure_variants(coll)
    if not variants:
        return []
    print(f"[variants] {len(variants)} object(s) in '{coll.name}':")

    misaligned, widths = [], []
    for i, (obj, w, d, h) in enumerate(variants):
        vs = obj.data.vertices
        xs = [v.co.x for v in vs]
        ys = [v.co.y for v in vs]
        zs = [v.co.z for v in vs]
        off = max(abs((min(xs) + max(xs)) / 2), abs((min(ys) + max(ys)) / 2),
                  abs(min(zs)))
        tag = ""
        if off > 0.01 * max(w, d, h, 0.01):
            misaligned.append(obj.name)
            tag = f"   <-- OFF-ORIGIN by {off:.3f} m"
        widths.append(w)
        print(f"  [{i}] {obj.name:<26} {w:.3f} x {d:.3f} x {h:.3f} m"
              f"   thickness {h:.3f}{tag}")

    if misaligned:
        print(f"[variants] WARNING: {', '.join(misaligned)} not centred on the "
              f"origin. These will render as a separate offset cluster. "
              f"Re-run this script to normalise them.")
    if len(variants) == 1:
        print("[variants] NOTE: only one object, so Variant Layout has "
              "nothing to switch between. Add a second object to use it.")
    if len(widths) > 1 and max(widths) > 1.6 * min(widths):
        print(f"[variants] NOTE: footprints differ by "
              f"{max(widths) / min(widths):.1f}x. Raise Slot Spacing X/Y so "
              f"the biggest item doesn't overlap its neighbours.")
    return [h for _, _, _, h in variants]


def hide_source_objects(coll):
    """Keep the originals out of renders without breaking Collection Info.

    `hide_render` only affects ray visibility - Collection Info reads the
    object data directly, so instances still render. Deliberately NOT using
    `hide_viewport` or view-layer exclusion: those pull objects out of the
    depsgraph, which can make instances fall back to unevaluated data.
    """
    if not PARK_SOURCES:
        return
    for i, obj in enumerate(coll.objects):
        # Never hide or move the generator itself - easy to hit if someone
        # drops TowelStacks into the source collection. The viewport would
        # look fine and every render would come back empty.
        if obj.name == GEN_OBJ_NAME or any(
                m.type == 'NODES' and m.node_group
                and m.node_group.name.startswith(GROUP_NAME)
                for m in obj.modifiers):
            print(f"[build] WARNING: generator '{obj.name}' is inside the "
                  f"source collection '{coll.name}'. Skipping it so it stays "
                  f"renderable, but you should drag it out.")
            continue
        obj.hide_render = True
        obj.location = (-6.0 - i * 0.8, -6.0, 0.0)


def build_scene(towel_folder=None):
    coll = import_towels(towel_folder)
    normalise_towel_origins(coll)
    thicknesses = report_variants(coll)
    if not thicknesses:
        thicknesses = [0.075]

    ng = build_node_group(thicknesses)

    obj = bpy.data.objects.get(GEN_OBJ_NAME)
    if obj is None:
        mesh = bpy.data.meshes.new(GEN_OBJ_NAME)
        obj = bpy.data.objects.new(GEN_OBJ_NAME, mesh)
        bpy.context.scene.collection.objects.link(obj)

    obj.modifiers.clear()
    mod = obj.modifiers.new("TowelStacks", "NODES")
    mod.node_group = ng

    n_variants = len(thicknesses)
    set_modifier_input(mod, ng, "Towel Collection", coll)
    set_modifier_input(mod, ng, "Towel Variants", n_variants)
    set_modifier_input(mod, ng, "Seed", 0)
    set_modifier_input(mod, ng, "Variant Layout", 2 if n_variants > 1 else 0)

    print(f"[build] per-variant thickness baked in: "
          + ", ".join(f"{t:.4f} m" for t in thicknesses))

    hide_source_objects(coll)
    obj.update_tag()
    print(f"[build] '{GROUP_NAME}' ready on object '{GEN_OBJ_NAME}' "
          f"({n_variants} variant(s))")
    return obj, mod, ng


# ---------------------------------------------------------------------------

def parse_args(argv):
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    out = {"towels": None, "out": None}
    for i, a in enumerate(args):
        if a in ("--towels", "--out") and i + 1 < len(args):
            out[a.lstrip("-")] = args[i + 1]
    return out


if __name__ == "__main__":
    opts = parse_args(sys.argv)
    build_scene(opts["towels"])
    if opts["out"]:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(opts["out"]))
        print(f"[build] saved {opts['out']}")

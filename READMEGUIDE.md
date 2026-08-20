# Towel Shelf Dataset Generator — User Guide

A Blender file that produces unlimited labelled training photos of folded
linen stacked on shelves, for training a computer-vision model to count them.

Every image comes with machine-readable labels: how many towels are visible,
where each one is, and its exact outline. You never label anything by hand.

**You do not need to know Blender, and you never need a terminal.**

---

## Contents

1. [What you need](#1-what-you-need)
2. [Your first dataset in five steps](#2-your-first-dataset-in-five-steps)
3. [The control panel explained](#3-the-control-panel-explained)
4. [What lands in your output folder](#4-what-lands-in-your-output-folder)
5. [Checking the labels are correct](#5-checking-the-labels-are-correct)
6. [Changing how the shelves look](#6-changing-how-the-shelves-look)
7. [Adding new linen models](#7-adding-new-linen-models)
8. [Recommended settings](#8-recommended-settings)
9. [Troubleshooting](#9-troubleshooting)
10. [Glossary](#10-glossary)

---

## 1. What you need

- **Blender 4.0 or 4.1** — free from [blender.org](https://www.blender.org/download/)
- The supplied `.blend` file
- Disk space: roughly **1.5 MB per image**, so about 1.5 GB for 1000 images

Nothing else. No Python, no `pip`, no command line. Blender includes
everything the tools need.

---

## 2. Your first dataset in five steps

### Step 1 — Open the file

Double-click the `.blend` file.

If a yellow warning bar appears saying scripts are blocked, click
**Allow Execution**. This file uses a small script to add its control panel;
Blender blocks that by default until you approve it.

> To stop being asked every time: **Edit ▸ Preferences ▸ Save & Load ▸
> tick "Auto Run Python Scripts"**, then **Save Preferences**.

### Step 2 — Open the control panel

In the main 3D window, press the **N** key. A sidebar opens on the right.
Click the **"Towel Data"** tab running down its edge.

> **No "Towel Data" tab?** Go to the **Scripting** tab along the very top of
> the window, and click the ▶ play button above the code. Then come back to
> **Layout** and press **N** again.

### Step 3 — Choose where to save

Click the folder icon next to **Output Folder** and pick or create a folder.
An empty folder is best — the tool creates its own subfolders inside it.

### Step 4 — Check everything is ready

Press **Check Setup**.

You want a green message like *"All good — 187 towels, camera framed
correctly."* If you get a red message instead, jump to
[Troubleshooting](#9-troubleshooting).

Then press **Preview One Image** to render a single picture. A window opens
showing what your dataset will look like. Close it when you're happy.

> This takes seconds and can save you hours. Always do it before a big run.

### Step 5 — Render

Set **Images** to how many you want, then press **Render Dataset**.

**Blender will freeze until it finishes. This is normal.** It is working, not
crashed. Don't force-quit it.

When it's done you'll see *"Render finished — saved to …"* at the bottom of
the window.

> **Start with 10 images**, look at the results, then commit to a big run.

---

## 3. The control panel explained

### Main settings

| Control | What it does |
|---|---|
| **Output Folder** | Where everything is saved. |
| **Images** | How many pictures to make. |
| **Start Seed** | The "recipe number" for the first image. Each image uses the next number up. See below. |
| **Engine** | **EEVEE** is fast (seconds per image). **Cycles** is slower but more photo-realistic (minutes per image). |
| **Device** | Cycles only. **CPU** always works. **GPU** is faster if your machine supports it, and falls back to CPU automatically if not. |
| **Quality** | Higher is cleaner but slower. 64 is a good balance. |
| **Create Annotations** | Leave this **on**. Without it you get pictures but no labels, which are the whole point. Roughly doubles render time. |

### About Start Seed

Every image is built from a seed number, which decides where each towel lands.
The same seed always produces exactly the same image.

This matters when you want **more** data later. If you already rendered 500
images at Start Seed 0, set Start Seed to **500** next time. Leave it at 0 and
you'll render the same 500 pictures again.

| Run | Start Seed | Images | Produces |
|---|---|---|---|
| First | 0 | 500 | seeds 0–499 |
| Second | 500 | 500 | seeds 500–999 (all new) |
| Third | 1000 | 1000 | seeds 1000–1999 (all new) |

### Realism

Makes the set look like photos taken by hand rather than one locked-off
tripod shot. Real-world photos are never perfectly consistent, and a model
trained only on identical framing copes badly with real ones.

| Control | What it does |
|---|---|
| **Camera Shake** | Nudges the camera slightly on every image. |
| **Angle (deg)** | How much tilt. **1–3** looks like a steady hand, **5+** like a rushed snapshot. |
| **Position** | How far the camera drifts. A few centimetres reads as someone shifting their weight. |
| **Light Variation** | Varies brightness image to image. |
| **Brightness** | How much it may vary, e.g. 0.25 = ±25%. |

The panel shows the resulting range live, e.g. *"Range: 75% – 125% of your
lights"*. Brightness is always **relative to the scene's own lighting** and is
floored at 65% — images can never come out too dark to be useful.

Your camera and lights are put back exactly as they were when the run
finishes.

### Check the labels

| Control | What it does |
|---|---|
| **Preview Images** | How many label previews to draw. |
| **Draw Label Previews** | Paints the labels onto the pictures so you can see them. See [section 5](#5-checking-the-labels-are-correct). |
| **Annotate Existing IDs** | Rebuilds the label file from pictures already rendered. Takes seconds. Use it if rendering finished but `annotations.json` is missing. |

---

## 4. What lands in your output folder

```
your-folder/
    images/            ← the training pictures
    ids/               ← internal working files
    preview/           ← label previews (only if you asked for them)
    annotations.json   ← the labels, COCO format
    counts.csv         ← how many towels are visible per picture
    seeds.csv          ← which seed made which picture
    pipeline_log.txt   ← a record of what happened
```

### The two you'll actually use

**`images/`** — ordinary PNG pictures. This is your training data.

**`annotations.json`** — the labels, in **COCO format**, the standard that
PyTorch, detectron2, mmdetection, YOLO converters and most annotation tools
read directly. For each towel it records:

- its **bounding box** — `[x, y, width, height]` in pixels
- its **exact outline** — as an RLE mask (see [Glossary](#10-glossary))
- its **area** in pixels

And for each picture, `visible_towels` — the count.

### `counts.csv`

Opens in Excel or Numbers. Two columns: picture name, and how many towels are
visible in it. Handy for a quick sanity check that the numbers look sensible.

### `ids/`

Internal. Flat blocks of solid colour on black — one colour per towel. This is
how the tool knows exactly which pixels belong to which towel. **It's meant to
look like that**, it isn't a broken render, and it isn't training data. Keep
it if you might want to rebuild the labels later; delete it to save space.

### `pipeline_log.txt`

A plain-text record of every step and any errors. **If something goes wrong,
read this first** — it will say what failed in plain language.

---

## 5. Checking the labels are correct

Rather than trusting a JSON file you can't read, look at the labels directly.

1. Set **Preview Images** to 6
2. Press **Draw Label Previews**
3. Open the **`preview/`** folder in your output folder

Each picture has every labelled towel tinted in its own colour with a box
drawn round it.

**What you want to see:** each tinted shape sitting exactly on top of a towel,
with different colours for touching towels, and no tint on the shelves or the
background.

**Do this once after your first run.** If the shapes line up, the labels are
correct and you can trust every image the tool makes. It's also the clearest
thing to show a colleague who wants proof the data is sound.

---

## 6. Changing how the shelves look

Every aspect of the stacks is adjustable. In the 3D window click the
**TowelStacks** object, then open **Modifier Properties** — the small blue
wrench icon in the right-hand column of buttons.

Change any value and the shelves update live in the viewport.

### The ones you'll want most

| Setting | Does |
|---|---|
| **Slots X / Slots Y** | How many stacks across and deep on each shelf |
| **Slot Spacing X / Y** | How far apart the stacks sit |
| **Shelf Count** | How many shelves high |
| **Shelf Spacing Z** | Gap between shelves |
| **Min Towels / Max Towels** | How tall the piles get |
| **Fill Probability** | Chance a spot has a pile at all. Lower = more empty gaps, which is realistic and useful |

### Making stacks messier or tidier

| Setting | Does |
|---|---|
| **Z Rotation Range** | How much each towel is twisted. **The main untidiness control** |
| **Tilt Range** | How much each towel leans. Keep small — towels don't perch |
| **Pile XY Jitter** | How far a whole pile drifts from its spot |
| **Towel XY Jitter** | How far one towel slides within its pile |
| **Scale Jitter** | Size variation between towels |

### Using more than one type of linen

If your collection holds several models — say a hand towel and a bed sheet —
**Variant Layout** decides how they're arranged:

| Value | Result |
|---|---|
| **0** | Mixed — a single pile can contain different items |
| **1** | Each pile is one item type, different piles side by side |
| **2** | Each shelf is one item type *(recommended)* |

**Cycle Variants By Shelf** (on by default) guarantees every item type appears
in every picture. Turn it off if you want some pictures to contain only one
type.

Because a bed sheet is bigger than a hand towel, items are split into two
groups with their own dimensions:

- **Group B From Variant** — where group B starts. With two models, set to
  **1**: the first model is group A, the second is group B.
- **Slot Spacing X B / Y B** — how far apart group B's stacks sit
- **Towel Thickness / Towel Thickness B** — how tall one item of each group
  is. Set automatically, but adjust if a pile looks like it's floating or the
  items overlap.

> Group B settings only take effect when **Variant Layout is 2**.

### Getting back to normal

Changed too much? Close Blender **without saving** and reopen. Nothing is lost
except unsaved edits — your rendered pictures are already on disk.

---

## 7. Adding new linen models

To teach the model about a new item — a pillowcase, a different fold, a
coloured towel:

1. **File ▸ Import ▸ Wavefront (.obj)** and pick your model
2. In the **Outliner** (top-right list), drag the new object into the towel
   collection alongside the existing models
3. Go to the **Scripting** tab and open **`build_towel_stacks.py`** from the
   dropdown at the top of the text area
4. Press the ▶ play button

The tool measures the new model, adds it to the generator, and fixes its
position and pivot automatically.

5. Back in **Modifier Properties**, check **Towel Variants** matches how many
   models you now have

> **Step 3 is not optional.** A model added without re-running the script
> won't be positioned correctly and will appear off to one side.

---

## 8. Recommended settings

### A first look

Images **10** · Engine **EEVEE** · Quality **32** · Create Annotations **on**

Minutes. Confirms everything works before you commit.

### A full training set

Images **1000–3000** · Engine **EEVEE** · Quality **64** · Camera Shake **on**
· Light Variation **0.25**

For counting and detection, **more images beats prettier images**. EEVEE gives
you ten to fifty times the pictures per hour, and dataset size and variety
matter far more than photo-realism. Run it overnight.

### A photo-realistic comparison set

Images **200** · Engine **Cycles** · Device **CPU** · Quality **96**

Slow — allow several hours. Worth having a small Cycles set to check the model
isn't relying on quirks of EEVEE's shading.

### Growing an existing set

Set **Start Seed** to the number of images you already have. See
[Start Seed](#about-start-seed).

---

## 9. Troubleshooting

**Always read `pipeline_log.txt` in your output folder first.** It records
every step and every error in plain language.

| Problem | Cause and fix |
|---|---|
| No "Towel Data" tab | Scripting tab ▸ press ▶. If a script warning appeared when opening, click **Allow Execution** and reopen the file. |
| *"No TowelStacks object found"* | The generator hasn't been built. Scripting tab ▸ open `build_towel_stacks.py` ▸ press ▶. |
| *"No towels are inside the camera frame"* | The camera is pointing elsewhere. In the 3D view press **Numpad 0** to look through it, then **View ▸ Align View ▸ Align Active Camera to View** to re-aim it. |
| *"is hidden from renders"* | Something has its camera icon switched off in the Outliner. Click the camera icon next to **TowelStacks** to switch it back on. |
| *"The generator produced 0 towels"* | In Modifier Properties, check **Fill Probability** is above 0 and **Towel Variants** matches how many models are in the collection. |
| Pictures are very dark | Switch the viewport to **Rendered** shading (the rightmost sphere icon, top-right of the 3D view). If it's dark there too, the scene needs brighter lights — Material Preview uses its own studio lighting that doesn't exist in the real render. |
| Blender looks frozen | It's rendering. Check the picture count rising in `images/`. Don't force-quit. |
| `annotations.json` is missing | Press **Annotate Existing IDs**. It rebuilds the labels from what's already rendered, in seconds. |
| Preview shapes don't line up | Press **Annotate Existing IDs**, then draw previews again. If they still don't line up, contact whoever supplied the file. |
| Everything is one giant pile | **Slot Spacing X / Y** is smaller than your models. Increase it. |
| A pile floats or overlaps itself | **Towel Thickness** doesn't match that model's real height. Adjust it (or **Towel Thickness B** for the second group). |

---

## 10. Glossary

**Seed** — a number that decides where every towel lands. The same seed always
produces the identical picture, so any image can be reproduced exactly.

**COCO format** — the standard file layout for training data, understood by
PyTorch, detectron2, mmdetection and most annotation tools. `annotations.json`
is in this format.

**Bounding box** — a rectangle around an object, given as
`[x, y, width, height]` in pixels, measured from the top-left corner.

**RLE mask** — "run-length encoding", a compact way to record an object's
exact outline pixel by pixel rather than as a rough rectangle. Standard COCO
and read directly by the usual training libraries.

**Instance** — one individual towel. A picture with 187 instances has 187
separate towels in it.

**EEVEE / Cycles** — Blender's two renderers. EEVEE is fast and approximate,
Cycles is slow and physically accurate.

**Visible towels** — towels the camera can actually see. Towels completely
hidden behind others aren't labelled, because a model can't be expected to
count something invisible.

---

*Generated with the Towel Shelf Dataset Generator. Keep this guide with the
`.blend` file.*

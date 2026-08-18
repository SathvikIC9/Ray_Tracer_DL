import os
import csv
import time
import zipfile
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import torch
# =========================================================================
# 0. GPU / Device Setup Snippet
# =========================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")



# =========================================================================
# 1. Redesigned SEM Generator (Reference is cropped/zoomed FROM the search
#    image itself, + sparse via dots + marked search for visualization)
# =========================================================================

def get_die_architecture_params(die_col, die_row, base_seed=42, force_arch=None):
    state = np.random.RandomState(int((die_col * 127 + die_row * 31 + base_seed) % 2**31))
    arch_type = force_arch if force_arch is not None else \
        state.choice(['standard_rect', 'dense_rect', 'wide_rect', 'fine_mesh'])

    # Diagonal comet-dash orientation/elongation, varied slightly per die for
    # natural-looking variety while staying broadly consistent with a "tilted
    # SEM capture" look (dashes trailing in roughly the same direction).
    streak_angle_deg = state.uniform(30, 50)
    streak_elong = state.uniform(2.2, 3.2)

    if arch_type == 'standard_rect':
        params = {"style": "comet", "pitch_x": 110, "pitch_y": 70, "line_w": 22, "via_r": 12, "center_dot_r": 5}
    elif arch_type == 'dense_rect':
        params = {"style": "comet", "pitch_x": 80, "pitch_y": 50, "line_w": 16, "via_r": 9, "center_dot_r": 4}
    elif arch_type == 'wide_rect':
        params = {"style": "comet", "pitch_x": 140, "pitch_y": 90, "line_w": 28, "via_r": 16, "center_dot_r": 6}
    else:  # fine_mesh -- fine bright crosshatch grid, dark background, sparse dark X marks
        params = {"style": "mesh", "pitch_x": state.uniform(13, 17), "pitch_y": state.uniform(13, 17),
                  "line_w": state.uniform(1.2, 1.8), "via_r": state.uniform(2.5, 3.5), "center_dot_r": 2}

    params["streak_angle"] = np.deg2rad(streak_angle_deg)
    params["streak_elong"] = streak_elong
    return params


def _junction_keep_mask(ix, iy, col, row, seed, dot_sparsity):
    """
    Deterministically decide, per grid-line intersection (ix, iy) inside die (col, row),
    whether that junction should render a via/center-dot.

    dot_sparsity: fraction of junctions that keep their dot marker.
        1.0 -> every junction gets a dot
        0.15 -> only ~15% of junctions get a dot, the rest are bare line-crossings
    """
    if dot_sparsity >= 1.0:
        return np.ones_like(ix, dtype=bool)

    junc_id = (ix.astype(np.int64) * 1_000_003
               + iy.astype(np.int64) * 97
               + col * 998_244_353
               + row * 100_000_007
               + seed * 15_485_863)

    h = (junc_id.astype(np.uint64) * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    frac = (h % np.uint64(1_000_000)).astype(np.float64) / 1_000_000.0
    return frac < dot_sparsity


def compute_heterogeneous_sem_tile(x_coords, y_coords, die_size=2400, street_width=200,
                                    seed=42, dot_sparsity=1.0, force_arch=None, rotation_deg=0.0):
    X, Y = np.meshgrid(x_coords, y_coords)

    if rotation_deg != 0.0:
        # Rotate the underlying wafer coordinates that get sampled, so the whole
        # die/via pattern appears tilted by a few degrees within the fixed pixel
        # canvas -- mimics a slightly rotated SEM stage/sample. Since this rotates
        # the *content*, not the output pixel grid, bounding boxes computed in pixel
        # space downstream stay valid with no extra correction needed.
        theta = np.deg2rad(rotation_deg)
        cx = (x_coords.min() + x_coords.max()) / 2.0
        cy = (y_coords.min() + y_coords.max()) / 2.0
        Xc, Yc = X - cx, Y - cy
        X = Xc * np.cos(theta) - Yc * np.sin(theta) + cx
        Y = Xc * np.sin(theta) + Yc * np.cos(theta) + cy

    die_period = die_size + street_width
    surface = np.ones_like(X, dtype=np.float32) * 0.12  # Kerf street dark floor

    min_col = int(np.floor(X.min() / die_period)) - 1
    max_col = int(np.floor(X.max() / die_period)) + 1
    min_row = int(np.floor(Y.min() / die_period)) - 1
    max_row = int(np.floor(Y.max() / die_period)) + 1

    for col in range(min_col, max_col + 1):
        for row in range(min_row, max_row + 1):
            p = get_die_architecture_params(col, row, base_seed=seed, force_arch=force_arch)

            die_x0, die_x1 = col * die_period, col * die_period + die_size
            die_y0, die_y1 = row * die_period, row * die_period + die_size

            die_mask = (X >= die_x0) & (X < die_x1) & (Y >= die_y0) & (Y < die_y1)
            if not np.any(die_mask):
                continue

            local_x = X - die_x0
            local_y = Y - die_y0

            if p["style"] == "mesh":
                # ---- Fine bright crosshatch mesh, dark background, sparse dark
                #      X marks at a few intersections (matches a woven/plaid-style
                #      SEM capture rather than the rectangular via-array look). ----
                distx = np.abs(((local_x + p["pitch_x"] / 2) % p["pitch_x"]) - p["pitch_x"] / 2)
                disty = np.abs(((local_y + p["pitch_y"] / 2) % p["pitch_y"]) - p["pitch_y"] / 2)
                line_x = np.exp(-(distx ** 2) / (2 * (p["line_w"] ** 2)))
                line_y = np.exp(-(disty ** 2) / (2 * (p["line_w"] ** 2)))
                mesh_grid = np.maximum(line_x, line_y)

                near_via_x = np.round(local_x / p["pitch_x"]) * p["pitch_x"]
                near_via_y = np.round(local_y / p["pitch_y"]) * p["pitch_y"]
                ix = np.round(local_x / p["pitch_x"]).astype(np.int64)
                iy = np.round(local_y / p["pitch_y"]).astype(np.int64)
                dx = local_x - near_via_x
                dy = local_y - near_via_y

                # Only a sparse subset of intersections get a dark X mark
                keep_mask = _junction_keep_mask(ix, iy, col, row, seed, dot_sparsity * 0.05)
                diag1 = dx + dy
                diag2 = dx - dy
                cross_shape = np.clip(
                    np.exp(-(diag1 ** 2) / (2 * (p["center_dot_r"] ** 2)))
                    + np.exp(-(diag2 ** 2) / (2 * (p["center_dot_r"] ** 2))),
                    0.0, 1.0,
                )
                dark_cross = 0.35 * cross_shape * keep_mask

                die_signal = 0.12 + 0.55 * mesh_grid - dark_cross
                surface[die_mask] = np.clip(die_signal, 0.0, 1.0)[die_mask]
                continue

            # ---- Rectangular grid + comet-dash via style (original path) ----
            # 1. Bold Rectangular Line Framework (always fully present)
            wordlines = 0.35 * np.clip(np.sin(2 * np.pi * local_y / p["pitch_y"]) + 0.3, 0, 1)
            bitlines = 0.35 * np.clip(np.sin(2 * np.pi * local_x / p["pitch_x"]) + 0.3, 0, 1)
            grid_pattern = np.maximum(wordlines, bitlines)

            # 2. Nearest junction indices for every pixel
            near_via_x = np.round(local_x / p["pitch_x"]) * p["pitch_x"]
            near_via_y = np.round(local_y / p["pitch_y"]) * p["pitch_y"]
            ix = np.round(local_x / p["pitch_x"]).astype(np.int64)
            iy = np.round(local_y / p["pitch_y"]).astype(np.int64)
            dx = local_x - near_via_x
            dy = local_y - near_via_y

            # Only a subset of junctions (dot_sparsity fraction) get a dash marker
            keep_mask = _junction_keep_mask(ix, iy, col, row, seed, dot_sparsity)

            # 3. Rotate into the die's diagonal streak frame: u = along the dash,
            #    v = across it. This turns each junction marker into a comet-shaped
            #    diagonal dash (bright head + trailing tail) instead of a round dot.
            angle = p["streak_angle"]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a

            core_r = p["center_dot_r"]
            tail_v_sigma = p["via_r"] * 0.45          # dash thickness
            tail_u_sigma = p["via_r"] * p["streak_elong"]  # dash length (elongated)

            # Bright compact head, centered on the junction
            head = 0.55 * np.exp(-(u ** 2 + v ** 2) / (2 * (core_r ** 2)))

            # Trailing tail: elongated only toward +u, fading out along its length
            # and staying thin across v -- gives the comet/dash look instead of a
            # symmetric blob.
            u_sigma = np.where(u >= 0, tail_u_sigma, core_r * 0.8)
            tail = 0.30 * np.exp(-(u ** 2) / (2 * (u_sigma ** 2)) - (v ** 2) / (2 * (tail_v_sigma ** 2)))

            dash = (head + tail) * keep_mask

            die_signal = 0.20 + grid_pattern + dash
            surface[die_mask] = np.clip(die_signal, 0.0, 1.0)[die_mask]

    return surface


def apply_sem_effects(img_array, blur_rad, noise_std, rng):
    pil_img = Image.fromarray((np.clip(img_array, 0.0, 1.0) * 255).astype(np.uint8))
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_rad))
    np_img = np.array(blurred).astype(np.float32) / 255.0
    grain_noise = rng.normal(0, noise_std, size=np_img.shape)
    return np.clip(np_img + grain_noise, 0.0, 1.0)


def _crop_and_zoom(raw_array, x0_px, y0_px, box_w_px, box_h_px, out_size):
    """
    Crop a (small) axis-aligned region out of raw_array using pixel coordinates,
    then upsample it to (out_size, out_size) to simulate zooming in -- this is how
    the reference image is produced directly FROM the search image instead of being
    rendered independently.
    """
    h, w = raw_array.shape
    x0 = int(np.floor(x0_px))
    y0 = int(np.floor(y0_px))
    x1 = int(np.ceil(x0_px + box_w_px))
    y1 = int(np.ceil(y0_px + box_h_px))

    # clip to valid array bounds, keep at least 1 px
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))

    patch = raw_array[y0:y1, x0:x1]
    patch_img = Image.fromarray((np.clip(patch, 0.0, 1.0) * 255).astype(np.uint8))
    zoomed = patch_img.resize((out_size, out_size), resample=Image.BICUBIC)
    return np.array(zoomed).astype(np.float32) / 255.0, (x0, y0, x1, y1)


def draw_reference_box(search_processed, box_px, color=(255, 60, 60), width=3):
    """
    Returns an RGB copy of the (grayscale, 0-1 float) search image with a rectangle
    drawn around the region the reference crop was taken from.
    """
    x0, y0, x1, y1 = box_px
    base = Image.fromarray((np.clip(search_processed, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(base)
    draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=color, width=width)
    return np.array(base)


def make_heterogeneous_pair(rng, pair_seed, crop_size=1000,
                             ref_zoom=10.0, search_dot_sparsity=0.7,
                             src_nm_per_pixel=10.0, force_arch=None,
                             rotate_prob=0.0, rotate_range=(2.0, 8.0),
                             mark_search=True):
    """
    The SEARCH tile is rendered first, at `src_nm_per_pixel` resolution (default
    10 nm/pixel -- the real wafer resolution), with `search_dot_sparsity` controlling
    what fraction of grid junctions actually show a via/dash/X marker (default 0.7 ->
    noticeably more empty junctions than a fully populated array).

    The REFERENCE image is then produced by cropping a small region directly out of
    that SAME search tile and zooming it up (bicubic upsample) by `ref_zoom` relative
    to the search resolution -- so the reference is guaranteed to be a real, consistent
    piece of the search image rather than an independently-rendered lookalike.

    `ref_zoom` is the magnification relative to the search image. The default of 10.0
    gives an exact reference:search pixel-pitch ratio of 1 : 10 (reference is 10x finer
    than search), regardless of what `src_nm_per_pixel` itself is set to.

    `force_arch`: pass a die style name (e.g. 'fine_mesh') to force every die in this
    tile to that single style, instead of the normal random per-die mix.

    `rotate_prob` / `rotate_range`: with probability `rotate_prob`, the whole tile's
    underlying pattern is rotated by a random angle (deg, sign randomized) drawn from
    `rotate_range` before rendering -- simulates a slightly tilted SEM capture. Because
    the reference is cropped from the (already rotated) search array and the bounding
    box is computed in output-pixel space, no separate ground-truth correction is
    needed.

    If mark_search=True, an extra RGB copy of the search image is returned with a
    rectangle drawn around the exact region the reference was cropped from, for
    visualization/sanity-checking (this marked copy is NOT meant to be used as
    training input, only for humans to verify the crop is correct).
    """
    ref_nm_per_pixel = src_nm_per_pixel / ref_zoom   # reference is zoomed in by ref_zoom

    die_size, die_period = 2400, 2600

    # How many whole dies actually fit across the rendered search extent at this
    # resolution -- keeps the randomly-chosen die (and therefore the reference crop)
    # guaranteed to fall inside the search tile even when src_nm_per_pixel is small
    # (i.e. the search tile itself only spans a fraction of a die).
    search_extent_nm = crop_size * src_nm_per_pixel
    max_die_index = max(0, int(search_extent_nm // die_period) - 1)
    random_die_col = rng.integers(0, max_die_index + 1)
    random_die_row = rng.integers(0, max_die_index + 1)

    # Physical footprint (in nm) that the reference crop covers on the wafer
    ref_physical_footprint = crop_size * ref_nm_per_pixel
    usable_span = int(die_size - ref_physical_footprint - 400)
    if usable_span < 1:
        raise ValueError("ref_zoom too small for this die_size/crop_size combo -- "
                          "the reference footprint doesn't fit inside a single die.")

    internal_x = rng.integers(200, 200 + usable_span)
    internal_y = rng.integers(200, 200 + usable_span)

    ref_origin_x = random_die_col * die_period + internal_x
    ref_origin_y = random_die_row * die_period + internal_y

    # Randomly rotate the underlying pattern for a subset of samples
    rotation_deg = 0.0
    if rotate_prob > 0.0 and rng.random() < rotate_prob:
        sign = rng.choice([-1.0, 1.0])
        rotation_deg = float(sign * rng.uniform(*rotate_range))

    # --- Render the SEARCH tile (this is the "real" wafer data) ---
    search_x = np.arange(0, crop_size) * src_nm_per_pixel
    search_y = np.arange(0, crop_size) * src_nm_per_pixel
    search_raw = compute_heterogeneous_sem_tile(search_x, search_y, seed=pair_seed,
                                                 dot_sparsity=search_dot_sparsity,
                                                 force_arch=force_arch,
                                                 rotation_deg=rotation_deg)
    search_processed = apply_sem_effects(search_raw, blur_rad=0.8, noise_std=0.05, rng=rng)

    # --- Crop + zoom the REFERENCE directly out of the search tile ---
    # Convert the reference's physical origin/footprint into search-pixel coordinates
    box_x0_px = ref_origin_x / src_nm_per_pixel
    box_y0_px = ref_origin_y / src_nm_per_pixel
    box_w_px = ref_physical_footprint / src_nm_per_pixel
    box_h_px = box_w_px

    ref_raw_zoomed, box_px = _crop_and_zoom(search_raw, box_x0_px, box_y0_px,
                                             box_w_px, box_h_px, out_size=crop_size)
    # Light extra blur/noise on top of the upsample so it still reads as its own capture
    ref_processed = apply_sem_effects(ref_raw_zoomed, blur_rad=1.2, noise_std=0.015, rng=rng)

    # --- Ground-truth target box (center + corners), in search-pixel coordinates ---
    x0, y0, x1, y1 = box_px
    gt_x = (x0 + x1) / 2.0
    gt_y = (y0 + y1) / 2.0

    search_marked = draw_reference_box(search_processed, box_px) if mark_search else None

    result = {
        "reference": (ref_processed * 255).astype(np.uint8),
        "search": (search_processed * 255).astype(np.uint8),
        "search_marked": search_marked,
        "gt_x": gt_x,
        "gt_y": gt_y,
        "bbox": box_px,  # (x0, y0, x1, y1) in search-pixel coordinates
        "rotation_deg": rotation_deg,
    }
    return result

# =========================================================================
# 2. Dataset Generation Pipeline
# =========================================================================

def generate_mixed_dataset(n_samples=1000, output_dir="/kaggle/working/mixed_wafer_dataset",
                            ref_zoom=10.0, search_dot_sparsity=0.7,
                            rotate_prob=0.35, rotate_range=(2.0, 8.0), save_marked=True):
    ref_dir = os.path.join(output_dir, "references")
    search_dir = os.path.join(output_dir, "searches")
    marked_dir = os.path.join(output_dir, "searches_marked")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)
    if save_marked:
        os.makedirs(marked_dir, exist_ok=True)

    rng = np.random.default_rng(42)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    print(f"Generating {n_samples} high-fidelity SEM pairs "
          f"(ref_zoom={ref_zoom}, search_dot_sparsity={search_dot_sparsity}, "
          f"rotate_prob={rotate_prob})...")
    t0 = time.time()

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "reference_path", "search_path", "search_marked_path",
                          "gt_x", "gt_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
                          "rotation_deg"])

        for i in range(n_samples):
            pair_seed = int(rng.integers(1000, 1000000))
            result = make_heterogeneous_pair(
                rng, pair_seed,
                ref_zoom=ref_zoom,
                search_dot_sparsity=search_dot_sparsity,
                rotate_prob=rotate_prob,
                rotate_range=rotate_range,
                mark_search=save_marked,
            )

            ref_name = f"ref_{i:05d}.png"
            search_name = f"search_{i:05d}.png"
            marked_name = f"search_marked_{i:05d}.png" if save_marked else ""

            Image.fromarray(result["reference"]).save(os.path.join(ref_dir, ref_name))
            Image.fromarray(result["search"]).save(os.path.join(search_dir, search_name))
            if save_marked:
                Image.fromarray(result["search_marked"]).save(os.path.join(marked_dir, marked_name))

            x0, y0, x1, y1 = result["bbox"]
            writer.writerow([
                i, f"references/{ref_name}", f"searches/{search_name}",
                f"searches_marked/{marked_name}" if save_marked else "",
                f"{result['gt_x']:.3f}", f"{result['gt_y']:.3f}",
                x0, y0, x1, y1, f"{result['rotation_deg']:.3f}",
            ])

            if (i + 1) % 20 == 0 or (i + 1) == n_samples:
                print(f"Progress: [{i+1}/{n_samples}] pairs completed in {time.time() - t0:.1f}s")

    zip_path = f"{output_dir}.zip"
    print(f"Compressing dataset into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                fp = os.path.join(root, file)
                zipf.write(fp, arcname=os.path.relpath(fp, start=os.path.dirname(output_dir)))

    print("Dataset generation complete!")


def generate_mesh_dataset(n_samples=1000, output_dir="/kaggle/working/mesh_wafer_dataset",
                           search_nm_per_pixel=3.0, ref_zoom=10.0, mesh_dot_sparsity=0.7,
                           rotate_prob=0.35, rotate_range=(2.0, 8.0), save_marked=True):
    """
    A separate dataset generator that forces EVERY die to the 'fine_mesh' crosshatch
    style (no rect/comet dies mixed in), and renders the search image itself a bit
    more zoomed-in than the main dataset: `search_nm_per_pixel` defaults to 3.0
    (vs. 10.0 in generate_mixed_dataset), i.e. the search tile is ~3.3x finer/more
    zoomed than the standard wafer-level search image. The reference stays a further
    exact 10x zoom on top of that (ref_zoom=10.0 -> reference = search_nm_per_pixel/10,
    e.g. 0.3 nm/pixel), preserving the same 1:10 ref:search ratio as the main dataset.
    """
    ref_dir = os.path.join(output_dir, "references")
    search_dir = os.path.join(output_dir, "searches")
    marked_dir = os.path.join(output_dir, "searches_marked")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)
    if save_marked:
        os.makedirs(marked_dir, exist_ok=True)

    rng = np.random.default_rng(43)
    manifest_path = os.path.join(output_dir, "manifest.csv")

    print(f"Generating {n_samples} fine-mesh SEM pairs "
          f"(search_nm_per_pixel={search_nm_per_pixel}, ref_zoom={ref_zoom}, "
          f"mesh_dot_sparsity={mesh_dot_sparsity}, rotate_prob={rotate_prob})...")
    t0 = time.time()

    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "reference_path", "search_path", "search_marked_path",
                          "gt_x", "gt_y", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1",
                          "rotation_deg"])

        for i in range(n_samples):
            pair_seed = int(rng.integers(1000, 1000000))
            result = make_heterogeneous_pair(
                rng, pair_seed,
                ref_zoom=ref_zoom,
                search_dot_sparsity=mesh_dot_sparsity,
                src_nm_per_pixel=search_nm_per_pixel,
                force_arch='fine_mesh',
                rotate_prob=rotate_prob,
                rotate_range=rotate_range,
                mark_search=save_marked,
            )

            ref_name = f"ref_{i:05d}.png"
            search_name = f"search_{i:05d}.png"
            marked_name = f"search_marked_{i:05d}.png" if save_marked else ""

            Image.fromarray(result["reference"]).save(os.path.join(ref_dir, ref_name))
            Image.fromarray(result["search"]).save(os.path.join(search_dir, search_name))
            if save_marked:
                Image.fromarray(result["search_marked"]).save(os.path.join(marked_dir, marked_name))

            x0, y0, x1, y1 = result["bbox"]
            writer.writerow([
                i, f"references/{ref_name}", f"searches/{search_name}",
                f"searches_marked/{marked_name}" if save_marked else "",
                f"{result['gt_x']:.3f}", f"{result['gt_y']:.3f}",
                x0, y0, x1, y1, f"{result['rotation_deg']:.3f}",
            ])

            if (i + 1) % 20 == 0 or (i + 1) == n_samples:
                print(f"Progress: [{i+1}/{n_samples}] pairs completed in {time.time() - t0:.1f}s")

    zip_path = f"{output_dir}.zip"
    print(f"Compressing dataset into {zip_path}...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                fp = os.path.join(root, file)
                zipf.write(fp, arcname=os.path.relpath(fp, start=os.path.dirname(output_dir)))

    print("Mesh dataset generation complete!")


if __name__ == "__main__":
    generate_mixed_dataset(n_samples=1000, ref_zoom=10.0, search_dot_sparsity=0.7,
                            rotate_prob=0.35, rotate_range=(2.0, 8.0), save_marked=True)
    generate_mesh_dataset(n_samples=1000, search_nm_per_pixel=3.0, ref_zoom=10.0,
                           mesh_dot_sparsity=0.7, rotate_prob=0.35, rotate_range=(2.0, 8.0),
                           save_marked=True)

image_tensor = torch.from_numpy(ref_processed).unsqueeze(0).unsqueeze(0).float() # (B, C, H, W)
image_tensor = image_tensor.to(device)
"""Evidence rendering: annotated full frame + zoomed/local crops as PNG panels."""
import os
from PIL import Image, ImageDraw

RED = (255, 40, 40)


def render_panel(img_path, box, out_path, max_full=640, crop_px=280):
    """Render one evidence panel: full frame with box + tight zoom + wide context.

    Local enlargement is what lets a VLM judge small targets.
    """
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    full = im.copy()
    d = ImageDraw.Draw(full)
    if box:
        d.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    outline=RED, width=3)
    scale = max_full / float(W)
    full = full.resize((max_full, max(1, int(H * scale))))

    crops = []
    if box:
        cx, cy = box[0] + box[2] / 2.0, box[1] + box[3] / 2.0
        base = max(box[2], box[3])
        for tag, side in (("zoom", base * 1.6 + 8), ("ctx", base * 5.0 + 16)):
            side = min(side, max(W, H))
            x0, y0 = cx - side / 2.0, cy - side / 2.0
            x0 = max(0, min(x0, W - side)) if side <= W else 0
            y0 = max(0, min(y0, H - side)) if side <= H else 0
            crop = im.crop((int(x0), int(y0),
                            int(min(W, x0 + side)), int(min(H, y0 + side))))
            crop = crop.resize((crop_px, crop_px))
            dc = ImageDraw.Draw(crop)
            dc.rectangle([box[0] - x0, box[1] - y0,
                          box[0] + box[2] - x0, box[1] + box[3] - y0],
                         outline=RED, width=3)
            dc.text((6, 6), tag, fill=(255, 255, 0))
            crops.append(crop)

    pad = 4
    height = full.height + (crop_px + pad if crops else 0)
    panel = Image.new("RGB", (max_full, height), (24, 24, 24))
    panel.paste(full, (0, 0))
    x = 0
    for c in crops:
        panel.paste(c, (x, full.height + pad))
        x += crop_px + pad
    panel.save(out_path)
    return out_path


def render_modality_panel(seq, modality, idx, out_dir):
    m = seq.modalities.get(modality)
    if m is None or m.frame_path(idx) is None:
        return None
    box = m.boxes[idx] if idx < len(m.boxes) else None
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{modality}_f{idx:04d}.png")
    try:
        render_panel(m.frame_path(idx), box, out)
    except Exception as e:
        return None
    return out


def contact_sheet(panel_paths, out_path, cols=3, cell_w=420):
    from PIL import Image as I
    panels = [I.open(p) for p in panel_paths if os.path.exists(p)]
    if not panels:
        return None
    cell_h = max(p.height for p in panels)
    rows = (len(panels) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (24, 24, 24))
    for k, p in enumerate(panels):
        r = p.width / float(cell_w)
        p2 = p.resize((cell_w, max(1, int(p.height / r))))
        sheet.paste(p2, ((k % cols) * cell_w, (k // cols) * cell_h))
    sheet.save(out_path)
    return out_path

"""Dataset discovery and annotation loading for RGB-T tracking audit."""
import os
import re

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def category_from_name(name):
    m = re.match(r"^(.*)_(\d+)$", name)
    return m.group(1) if m else name


def parse_box(line):
    s = line.strip().replace(",", " ").split()
    if len(s) != 4:
        return None
    try:
        vals = [float(v) for v in s]
    except ValueError:
        return None
    if not all(v == v for v in vals):
        return None
    return vals


def _find_img_dir(seq_root, mod):
    for sub in ("NotAlign", "Align", ""):
        d = os.path.join(seq_root, sub, mod)
        if os.path.isdir(d):
            return d
    return None


def _probe_size(img_dir):
    if not img_dir or not os.path.isdir(img_dir):
        return None
    try:
        from PIL import Image
        files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))
        if not files:
            return None
        with Image.open(os.path.join(img_dir, files[0])) as im:
            return im.size
    except Exception:
        return None


class ModalityData:
    def __init__(self, name, img_dir, anno_path):
        self.name = name
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.frames = []
        if img_dir and os.path.isdir(img_dir):
            self.frames = sorted(f for f in os.listdir(img_dir)
                                 if f.lower().endswith(IMG_EXTS))
        self.raw_bytes = None
        self.raw_lines = []
        self.boxes = []
        self.size = None
        self.load()

    def load(self):
        if self.anno_path and os.path.exists(self.anno_path):
            with open(self.anno_path, "rb") as f:
                self.raw_bytes = f.read()
            self.raw_lines = self.raw_bytes.decode("utf-8", "replace").splitlines()
        self.boxes = [parse_box(l) for l in self.raw_lines]
        self.size = _probe_size(self.img_dir)

    def frame_path(self, idx):
        if 0 <= idx < len(self.frames):
            return os.path.join(self.img_dir, self.frames[idx])
        return None


class Sequence:
    def __init__(self, name, root):
        self.name = name
        self.root = root
        self.category = category_from_name(name)
        self.modalities = {}
        # layout flags: shared_gt = one GT file serves both modalities
        # (registered dataset); relaxed_count = MOT-style rows, missing
        # frames are legitimate target absence, not a count error
        self.shared_gt = False
        self.relaxed_count = False
        self.per_frame_json = False  # UAV-CB LabelMe: one JSON per frame

    def frame_path(self, modality, idx):
        m = self.modalities.get(modality)
        return m.frame_path(idx) if m else None


def _find_m3track_dirs(seq_root):
    vdir = tdir = None
    for d in sorted(os.listdir(seq_root)):
        p = os.path.join(seq_root, d)
        if not os.path.isdir(p):
            continue
        if d.endswith("V_cut") or d.endswith("V_undistort"):
            vdir = p
        elif d.endswith("T_cut") or d.endswith("T_undistort"):
            tdir = p
    return vdir, tdir


def _load_m3track_gt(gt_path):
    """Parse MOT csv (frame,id,x,y,w,h,...) -> dominant-id trajectory.

    Returns boxes aligned to frame index (0-based, frames assumed 1..N
    contiguous); missing frames -> None.
    """
    import collections
    per_id = collections.defaultdict(dict)
    max_frame = 0
    with open(gt_path) as f:
        for line in f:
            s = line.strip().replace(",", " ").split()
            if len(s) < 6:
                continue
            try:
                frame, tid = int(float(s[0])), int(float(s[1]))
                x, y, w, h = (float(v) for v in s[2:6])
            except ValueError:
                continue
            per_id[tid][frame] = [x, y, w, h]
            max_frame = max(max_frame, frame)
    if not per_id or max_frame == 0:
        return [], None
    # dominant id = most annotated frames
    tid, frames = max(per_id.items(), key=lambda kv: len(kv[1]))
    boxes = [frames.get(i) for i in range(1, max_frame + 1)]
    return boxes, tid


def _m3track_modality(name, img_dir, boxes, size, gt_path):
    m = ModalityData(name, img_dir, None)
    m.anno_path = gt_path
    m.boxes = boxes
    m.raw_lines = [",".join(str(v) for v in b) if b else ""
                   for b in boxes]
    m.size = size or _probe_size(img_dir)
    return m


def _load_m3track(seq, name):
    gt = os.path.join(seq.root, "gt", "gt.txt")
    vdir, tdir = _find_m3track_dirs(seq.root)
    if not os.path.exists(gt) or not vdir:
        return False
    boxes, tid = _load_m3track_gt(gt)
    if not boxes:
        return False
    size = _probe_size(vdir)
    seq.shared_gt = True
    seq.relaxed_count = True
    base = re.sub(r"^\d+", "", name).rsplit("_View", 1)[0]
    seq.category = base or name
    seq.modalities["visible"] = _m3track_modality("visible", vdir, boxes, size, gt)
    if tdir:
        seq.modalities["infrared"] = _m3track_modality(
            "infrared", tdir, boxes, _probe_size(tdir), gt)
    return True


def _parse_labelme_rect(shape):
    """LabelMe rectangle shape -> [x, y, w, h] from corner points."""
    pts = shape.get("points") or []
    if len(pts) < 2 or shape.get("shape_type") != "rectangle":
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    return [x, y, w, h]


def _load_uavcb_category(cat_dir, category):
    """UAV-CB LabelMe layout: <cat>/{visible,thermal}/<cat>_<mod>_<video>_<idx>.{jpg,json}.

    One sequence per <category>_<video>; per-frame JSON annotations become the
    trajectory. Unregistered: modalities carry their own resolutions, so the
    dual-copy check stays active. Returns [] if the layout does not match.
    """
    import json as _json
    mods = {}
    for mod in ("visible", "thermal"):
        d = os.path.join(cat_dir, mod)
        if not os.path.isdir(d):
            return []
        rx = re.compile(r"^%s_%s_(\d+)_\d+\.(?:jpg|jpeg|png|bmp)$"
                        % (re.escape(category), mod), re.I)
        files = [f for f in os.listdir(d) if rx.match(f)]
        if not files:
            return []
        mods[mod] = (d, rx, files)
    videos = set()
    for d, rx, files in mods.values():
        for f in files:
            videos.add(rx.match(f).group(1))
    seqs = []
    for vid in sorted(videos):
        seq = Sequence("%s_%s" % (category, vid), cat_dir)
        seq.category = category
        seq.per_frame_json = True
        for mod, (d, rx, files) in mods.items():
            frames = sorted(f for f in files if rx.match(f).group(1) == vid)
            boxes, size = [], None
            raw_lines = []
            for fn in frames:
                jf = os.path.join(d, os.path.splitext(fn)[0] + ".json")
                if not os.path.exists(jf):
                    # per-frame JSON layout: a missing JSON is a missing
                    # annotation, not target absence (empty shapes = absence)
                    boxes.append(None)
                    raw_lines.append("MISSING_JSON:%s" % fn)
                    continue
                try:
                    with open(jf) as fh:
                        data = _json.load(fh)
                except ValueError:
                    boxes.append(None)
                    raw_lines.append("INVALID_JSON:%s" % fn)
                    continue
                if size is None and data.get("imageWidth") and data.get("imageHeight"):
                    size = (data["imageWidth"], data["imageHeight"])
                rect = None
                for shape in data.get("shapes", []):
                    rect = _parse_labelme_rect(shape)
                    if rect:
                        break
                boxes.append(rect)
                raw_lines.append(",".join(str(v) for v in rect) if rect else "")
            if not frames:
                continue
            m = ModalityData(mod, d, os.path.join(d, os.path.splitext(frames[0])[0] + ".json"))
            m.frames = frames
            m.boxes = boxes
            m.raw_lines = raw_lines
            m.raw_bytes = "\n".join(raw_lines).encode()
            m.size = size or _probe_size(d)
            seq.modalities[mod] = m
        if seq.modalities:
            seqs.append(seq)
    return seqs


def discover_sequences(root, pattern=None, seqs_file=None, limit=None):
    """Discover RGB-T SOT sequences under root; layout auto-detected.

    LUART layout: <root>/<seq>/{visible,infrared}.txt + [NotAlign|Align]/<mod>/.
    M3Track layout: <root>/{train,test}/<seq>/img{1V,1T}_cut/ + gt/gt.txt
    (MOT csv; the dominant target id becomes the SOT trajectory).
    UAV-CB layout: <root>/<category>/{visible,thermal}/<cat>_<mod>_<video>_
    <idx>.{jpg,json} (LabelMe per-frame JSON, unregistered resolutions).
    """
    wanted = None
    if seqs_file:
        with open(seqs_file) as f:
            wanted = set(l.strip() for l in f if l.strip())
    rx = re.compile(pattern) if pattern else None
    seqs = []

    def want(name):
        return ((wanted is None or name in wanted)
                and (rx is None or rx.search(name)))

    for sub in ("train", "test", ""):
        base = os.path.join(root, sub) if sub else root
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if not os.path.isdir(p) or not want(name):
                continue
            seq = Sequence(name, p)
            if os.path.isdir(os.path.join(p, "gt")) and _load_m3track(seq, name):
                seqs.append(seq)
            else:
                uav = _load_uavcb_category(p, name)
                if uav:
                    for s in uav:
                        if want(s.name):
                            seqs.append(s)
                else:
                    _load_luart(seq, p)
                    if seq.modalities:
                        seqs.append(seq)
            if limit and len(seqs) >= limit:
                return seqs
    return seqs


def _load_luart(seq, p):
    for mod, anno in (("visible", "visible.txt"), ("infrared", "infrared.txt")):
        anno_path = os.path.join(p, anno)
        if not os.path.exists(anno_path):
            anno_path = os.path.join(p, "NotAlign", anno)
        if not os.path.exists(anno_path):
            anno_path = None
        img_dir = _find_img_dir(p, mod)
        if img_dir or anno_path:
            seq.modalities[mod] = ModalityData(mod, img_dir, anno_path)


def load_single_sequence(name, vis_img_dir, ir_img_dir, vis_anno, ir_anno):
    seq = Sequence(name, name)
    if vis_img_dir or vis_anno:
        seq.modalities["visible"] = ModalityData("visible", vis_img_dir, vis_anno)
    if ir_img_dir or ir_anno:
        seq.modalities["infrared"] = ModalityData("infrared", ir_img_dir, ir_anno)
    return seq

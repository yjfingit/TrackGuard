"""Tier-2 temporal trajectory analysis.

Each modality is analyzed independently; cross-modal coordinates are never
compared because the data is unregistered.
"""
import math
import statistics


def _center(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def _median_filter(points, k=5):
    half = k // 2
    out = []
    for i in range(len(points)):
        lo, hi = max(0, i - half), min(len(points), i + half + 1)
        win = points[lo:hi]
        out.append((statistics.median(p[0] for p in win),
                    statistics.median(p[1] for p in win)))
    return out


def analyze(boxes, size):
    """Return (signals, suspicious_frames). signals: list of dicts.

    suspicious_frames is capped and ranked by severity (normalized jump
    displacement / drift residual) so that on highly dynamic targets a
    real annotation break is not crowded out by natural motion jitter.
    """
    signals = []
    sus = {}  # frame -> severity score
    n = len(boxes)
    valid = [(i, b) for i, b in enumerate(boxes) if b is not None]
    if n < 3 or len(valid) < 3:
        return signals, sorted(sus)
    W, H = size if size else (1920, 1080)
    diag = math.hypot(W, H)
    idxs = [i for i, _ in valid]
    centers = {i: _center(b) for i, b in valid}

    def mark(frames_iter, score):
        for f in frames_iter:
            if score > sus.get(f, 0.0):
                sus[f] = score

    # consecutive-frame motion / IoU stats (per-frame normalized for gaps)
    steps = []
    for a, b in zip(idxs, idxs[1:]):
        gap = max(1, b - a)
        dx = centers[b][0] - centers[a][0]
        dy = centers[b][1] - centers[a][1]
        steps.append((a, b, math.hypot(dx, dy) / (gap * diag), _iou(boxes[a], boxes[b])))
    disps = [s[2] for s in steps]
    med = statistics.median(disps)
    mad = statistics.median(abs(d - med) for d in disps)
    sigma = 1.4826 * mad + 0.002

    jumps = [s for s in steps
             if s[2] > 0.06 and s[3] < 0.05 and abs(s[2] - med) > 6 * sigma]
    if jumps:
        frames = sorted(set(sum(([s[0], s[1]] for s in jumps), [])))
        signals.append({"code": "traj_jump", "frames": frames[:50],
                        "detail": f"{len(jumps)} abrupt motion jump(s), "
                                  f"largest {max(s[2] for s in jumps):.1%} of diag/frame"})
        for s in jumps:
            mark((s[0], s[1]), s[2])

    # frozen box runs
    best_start, best_len, cur = 0, 1, 1
    for a, b in zip(idxs, idxs[1:]):
        if boxes[a] == boxes[b]:
            cur += 1
            if cur > best_len:
                best_start, best_len = b - cur + 1, cur
        else:
            cur = 1
    if best_len > max(30, int(0.15 * n)):
        signals.append({"code": "traj_frozen", "frames": [best_start],
                        "detail": f"identical box repeated for {best_len} "
                                  f"consecutive frames (static target or stuck box)"})
        mark((best_start,), 0.06)

    # size anomalies
    size_anom = [s for s in steps
                 if boxes[s[0]][2] * boxes[s[0]][3] > 0
                 and (boxes[s[1]][2] * boxes[s[1]][3]
                      / (boxes[s[0]][2] * boxes[s[0]][3])) > 8
                 and s[3] < 0.1]
    size_anom += [s for s in steps
                  if boxes[s[1]][2] * boxes[s[1]][3] > 0
                  and (boxes[s[0]][2] * boxes[s[0]][3]
                       / (boxes[s[1]][2] * boxes[s[1]][3])) > 8
                  and s[3] < 0.1]
    if size_anom:
        frames = sorted(set(sum(([s[0], s[1]] for s in size_anom), [])))
        signals.append({"code": "traj_size_anomaly", "frames": frames[:50],
                        "detail": f"{len(size_anom)} frame pair(s) with >8x area "
                                  f"change and IoU<0.1"})
        mark(frames, 0.08)

    # drift: persistent deviation from median-filtered trajectory
    pts = [centers[i] for i in idxs]
    smooth = _median_filter(pts)
    resids = []
    for pos, i in enumerate(idxs):
        r = math.hypot(pts[pos][0] - smooth[pos][0], pts[pos][1] - smooth[pos][1])
        resids.append((r / diag, i))
    big = [(r, i) for r, i in resids if r > 0.05]
    if len(big) > max(3, int(0.1 * n)):
        big.sort(reverse=True)
        signals.append({"code": "traj_drift", "frames": [i for _, i in big[:50]],
                        "detail": f"{len(big)} frames deviate >5% diag from "
                                  f"smoothed trajectory"})
        for r, i in big[:12]:
            mark((i,), r)

    ranked = sorted(sus.items(), key=lambda kv: (-kv[1], kv[0]))
    return signals, ranked[:16]

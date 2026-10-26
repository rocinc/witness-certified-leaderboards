"""Programmatic acceptance checks for manuscript figures.

check_figure(fig)  -- collisions inside a live matplotlib figure, measured in
                      display coordinates after a draw (handles twinx, log
                      axes and blended transforms uniformly).
check_pdf(path, w) -- printed size, font sizes and font families of a saved
                      PDF that LaTeX includes at width w (inches).
"""
import sys

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from shapely.geometry import LineString, Point, box

AREA_TOL = 1.0      # px^2, ignore float-level overlaps
EDGE_TOL = 1.0      # px


def _text_box(t, r):
    patch = t.get_bbox_patch()
    bb = (patch.get_window_extent(r) if patch is not None and patch.get_visible()
          else t.get_window_extent(r))
    return box(bb.x0, bb.y0, bb.x1, bb.y1)


def _drawn_ticklabels(axis):
    """Tick labels matplotlib actually draws (inside the view interval)."""
    lo, hi = sorted(axis.get_view_interval())
    span = hi - lo
    out = []
    for tick in axis.get_major_ticks() + axis.get_minor_ticks():
        loc = tick.get_loc()
        if lo - 1e-9 * span <= loc <= hi + 1e-9 * span:
            out += [lab for lab in (tick.label1, tick.label2) if lab.get_visible()]
    return out


def _collect_texts(fig):
    axis_texts, legend_texts = set(), set()
    for ax in fig.axes:
        axis_texts.update([ax.xaxis.label, ax.yaxis.label, ax.title,
                           ax._left_title, ax._right_title])
        axis_texts.update(_drawn_ticklabels(ax.xaxis) + _drawn_ticklabels(ax.yaxis))
        leg = ax.get_legend()
        if leg is not None:
            legend_texts.update(leg.get_texts())
            if leg.get_title().get_text():
                legend_texts.add(leg.get_title())
    for leg in fig.legends:
        legend_texts.update(leg.get_texts())
    hidden_ticks = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                hidden_ticks.update([tick.label1, tick.label2])
    hidden_ticks -= axis_texts
    out = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or t in hidden_ticks:
            continue
        if getattr(t, "_figcheck_skip", False):
            continue
        if t in axis_texts:
            out.append((t, "axis"))
        elif t in legend_texts:
            out.append((t, "legend"))
        else:
            out.append((t, "free"))
    return out


def _lines(fig):
    segs = []
    for ax in fig.axes:
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_linestyle() in ("None", "", " "):
                continue
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
            xy = xy[np.all(np.isfinite(xy), axis=1)]
            if len(xy) >= 2:
                segs.append((ln, LineString(xy)))
    return segs


def _points(fig):
    pts = []
    for ax in fig.axes:
        for ln in ax.get_lines():           # marker-only lines
            if ln.get_marker() in (None, "None", "", " ") or not ln.get_visible():
                continue
            rad = ln.get_markersize() * fig.dpi / 72.0 / 2
            xy = ln.get_transform().transform(np.column_stack(ln.get_data()))
            pts += [(ln, Point(p).buffer(rad)) for p in xy if np.all(np.isfinite(p))]
        for col in ax.collections:
            if not isinstance(col, PathCollection) or not col.get_visible():
                continue
            offs = col.get_offset_transform().transform(col.get_offsets())
            sizes = col.get_sizes() if len(col.get_sizes()) else [36.0]
            for i, p in enumerate(offs):
                s = sizes[i % len(sizes)]
                rad = np.sqrt(s) * fig.dpi / 72.0 / 2
                if np.all(np.isfinite(p)):
                    pts.append((col, Point(p).buffer(rad)))
    return pts


def _own_line(t, ln, geom, bx):
    """True if ln is the leader/curve this label is deliberately attached to."""
    if getattr(t, "_figcheck_owner", None) is ln:
        return True
    if ln.get_gid() == "figcheck-leader":
        # a leader belongs to the label that one of its endpoints touches
        x0, y0 = geom.coords[0]
        x1, y1 = geom.coords[-1]
        pad = bx.buffer(EDGE_TOL + 1.5)
        return pad.contains(Point(x0, y0)) or pad.contains(Point(x1, y1))
    return False


def check_figure(fig, name="figure", exempt_line_overlap=()):
    """Return a report dict; print it. Non-empty problems -> failure."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    fig_box = box(-EDGE_TOL, -EDGE_TOL, W + EDGE_TOL, H + EDGE_TOL)
    texts = [(t, kind, _text_box(t, r)) for t, kind in _collect_texts(fig)]
    lines, points = _lines(fig), _points(fig)
    probs = []

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i], texts[j]
            if a[1] == b[1] and a[1] in ("axis", "legend"):
                continue
            if a[2].intersection(b[2]).area > AREA_TOL:
                probs.append(("text-text", a[0].get_text(), b[0].get_text()))
    for t, kind, bx in texts:
        label = t.get_text()
        if not fig_box.contains(bx):
            probs.append(("off-figure", label, ""))
        if kind in ("axis", "legend") or label in exempt_line_overlap:
            continue
        ax = t.axes
        if ax is not None and t.get_clip_on() and not getattr(t, "_figcheck_outside_ok", False):
            ab = ax.get_window_extent(r)
            if not box(ab.x0 - EDGE_TOL, ab.y0 - EDGE_TOL,
                       ab.x1 + EDGE_TOL, ab.y1 + EDGE_TOL).contains(bx):
                probs.append(("off-axes", label, ""))
        for ln, geom in lines:
            if _own_line(t, ln, geom, bx):
                continue
            if geom.intersects(bx):
                probs.append(("text-line", label, ln.get_label()))
        for art, disk in points:
            if disk.intersection(bx).area > AREA_TOL:
                probs.append(("text-point", label, getattr(art, "get_label", lambda: "")()))
    rep = {"name": name, "n_text": len(texts), "problems": probs}
    tag = "OK " if not probs else "FAIL"
    print(f"[{tag}] {name}: {len(texts)} texts, {len(probs)} problems")
    for p in probs:
        print("       ", p)
    return rep


def check_pdf(path, width_in, lo=5.95, hi=9.5, font_ok=("STIX",)):
    # Elsevier: 7 pt text, >= 6 pt sub/superscripts; nothing above caption size
    """Printed-size checks on a saved PDF included at width_in inches."""
    import pymupdf
    page = pymupdf.open(path)[0]
    scale = width_in * 72.0 / page.rect.width
    sizes, bad_fonts = [], set()
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for sp in l["spans"]:
                if not sp["text"].strip():
                    continue
                sizes.append(sp["size"] * scale)
                if not any(f in sp["font"] for f in font_ok):
                    bad_fonts.add(sp["font"])
    out = [s for s in sizes if s < lo or s > hi]
    ok = bool(sizes) and abs(scale - 1.0) < 0.01 and not out and not bad_fonts
    tag = "OK " if ok else "FAIL"
    rng = f"{min(sizes):.2f}-{max(sizes):.2f}pt" if sizes else "no text"
    print(f"[{tag}] {path.split('/')[-1]}: scale={scale:.3f}, sizes {rng}, "
          f"out-of-range={len(out)}, fonts-bad={sorted(bad_fonts)}")
    return ok


if __name__ == "__main__":
    # usage: python figcheck.py file.pdf width_in [file.pdf width_in ...]
    args = sys.argv[1:]
    ok = all(check_pdf(args[i], float(args[i + 1])) for i in range(0, len(args), 2))
    sys.exit(0 if ok else 1)

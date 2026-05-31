"""Parser for the Billiards Sports Analytics trajectory files.

Each strike is stored as Kinovea-exported SpreadsheetML (``.xml``) holding a
single track: rows of ``x, y, t`` where x/y are table coordinates in
centimeters and t is a timestamp string. A strike folder ``object N`` contains:

  * ``cue ball.xml``  – the cue ball's path during that strike
  * ``object N.xml``  – the struck object ball's path

This module exposes small helpers to read those tracks and to enumerate strikes
across the dataset, so the evaluation code stays focused on the metrics.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_NS = {"ss": "urn:schemas-microsoft-com:office:spreadsheet"}


def read_track(xml_path: str | Path) -> list[tuple[float, float]]:
    """Return the (x, y) points (cm) of a Kinovea track file, in order."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ws = root.find(".//ss:Worksheet", _NS)
    if ws is None:
        return []
    table = ws.find("ss:Table", _NS)
    if table is None:
        return []

    pts: list[tuple[float, float]] = []
    for row in table.findall("ss:Row", _NS):
        vals = []
        for cell in row.findall("ss:Cell", _NS):
            data = cell.find("ss:Data", _NS)
            vals.append(data.text if data is not None else None)
        if len(vals) < 2:
            continue
        try:
            x = float(vals[0])
            y = float(vals[1])
        except (TypeError, ValueError):
            continue  # header rows ("x", "y", labels, etc.)
        pts.append((x, y))
    return pts


@dataclass
class Strike:
    """One recorded strike: cue path, struck-ball id and its path."""

    match: str
    round_name: str
    object_id: int
    cue_path: list[tuple[float, float]]
    object_path: list[tuple[float, float]]


def iter_strikes(trajectories_root: str | Path):
    """Yield :class:`Strike` for every strike that has both cue and object tracks."""
    root = Path(trajectories_root)
    for cue_file in root.rglob("cue ball.xml"):
        strike_dir = cue_file.parent           # .../trajectory/round K/object N
        obj_files = sorted(strike_dir.glob("object *.xml"))
        if not obj_files:
            continue
        obj_file = obj_files[0]
        try:
            object_id = int(obj_file.stem.split()[-1])
        except ValueError:
            object_id = -1

        cue_path = read_track(cue_file)
        object_path = read_track(obj_file)
        if len(cue_path) < 2 or len(object_path) < 2:
            continue

        # .../<match>/trajectory/<round>/<object>
        round_name = strike_dir.parent.name
        match = strike_dir.parents[2].name
        yield Strike(match, round_name, object_id, cue_path, object_path)


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/data_trajectories"
    n = 0
    xs, ys = [], []
    for s in iter_strikes(root):
        n += 1
        for x, y in s.cue_path + s.object_path:
            xs.append(x)
            ys.append(y)
    if xs:
        print(f"strikes with cue+object tracks: {n}")
        print(f"x range: {min(xs):.1f} .. {max(xs):.1f} cm")
        print(f"y range: {min(ys):.1f} .. {max(ys):.1f} cm")
    else:
        print("no strikes found under", root)

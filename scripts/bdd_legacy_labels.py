r"""Convert the 2018 per-image BDD100K labels zip into the legacy ``bdd100k_labels_images_<split>.json``.

P1.1.0's competence study needed BDD **train** (70k frames). The train images came from the
``bdd100k_images_100k.zip`` release, but its labels only ship as ``bdd100k_labels.zip``: one JSON per image
(``100k/<split>/<name>.json``) carrying boxes, lanes and drivable-area polygons under the 2018 category
names. :mod:`cafl4ds.data` reads the legacy single-file format instead. This merges the per-image files into
it — the image attributes (weather / scene / time of day) verbatim, boxes only, and the 2018 names mapped
back to the legacy ones (``bike → bicycle``, ``motor → motorcycle``, ``person → pedestrian``).

Checked against the legacy val release: attributes match exactly, per-image box counts agree to ~1%.

The val file on this node carries each box's category only. P1.2.5 needs val's box geometry, so a second
source is accepted: the FiftyOne export of BDD val (``samples.json``, from ``dgural/bdd100k`` — the copy the
val images came from). Its boxes (relative ``[x, y, w, h]``) become ``box2d`` in pixels, and the records keep
the ``--reference`` file's order and attributes, so a pool read from it is frame-for-frame the old one.

Examples::

    uv run python scripts/bdd_legacy_labels.py bdd100k_labels.zip train out/labels_train.json
    uv run python scripts/bdd_legacy_labels.py samples.json val out/labels_val.json \
        --reference bdd100k_labels_images_val.json
"""

import argparse
import collections
import json
import zipfile
from pathlib import Path
from typing import Any

from loguru import logger

RENAME = {"bike": "bicycle", "motor": "motorcycle", "person": "pedestrian"}


def legacy_records(zip_path: str | Path, split: str) -> list[dict[str, Any]]:
    """One legacy record (``name``, ``attributes``, box ``labels``) per per-image JSON of ``split``."""
    records = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if not (info.filename.startswith(f"100k/{split}/") and info.filename.endswith(".json")):
                continue
            image = json.loads(archive.read(info))
            boxes = [
                {**obj, "category": RENAME.get(obj["category"], obj["category"])}
                for frame in image.get("frames", [])
                for obj in frame.get("objects", [])
                if "box2d" in obj
            ]
            records.append({"name": f"{image['name']}.jpg", "attributes": image["attributes"], "labels": boxes})
    return records


def fiftyone_records(samples_path: str | Path, reference: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``reference`` records, in its order and with its attributes, with boxes taken from a FiftyOne export.

    Each FiftyOne detection's relative ``bounding_box`` ``[x, y, w, h]`` becomes a pixel ``box2d`` on the image's
    own size; the category is the detection's label (already legacy names in this export).

    Raises:
        ValueError: If a reference image is missing from the export.
    """
    samples = json.loads(Path(samples_path).read_text(encoding="utf-8"))["samples"]
    by_name = {Path(s["filepath"]).name: s for s in samples}
    missing = [r["name"] for r in reference if r["name"] not in by_name]
    if missing:
        raise ValueError(f"{len(missing)} reference images are missing from {samples_path} (e.g. {missing[:3]}).")
    records = []
    for ref in reference:
        sample = by_name[ref["name"]]
        width, height = sample["metadata"]["width"], sample["metadata"]["height"]
        boxes = []
        for det in (sample.get("detections") or {}).get("detections", []):
            x, y, w, h = det["bounding_box"]
            box2d = {"x1": x * width, "y1": y * height, "x2": (x + w) * width, "y2": (y + h) * height}
            boxes.append({"category": det["label"], "box2d": box2d})
        records.append({"name": ref["name"], "attributes": ref["attributes"], "labels": boxes})
    return records


def compare(records: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, int]:
    """Coverage, attribute and per-image box-category-count agreement against a legacy release."""

    def counts(record: dict[str, Any]) -> collections.Counter[str]:
        return collections.Counter(box.get("category") for box in record.get("labels") or [])

    ref, mine = {r["name"]: r for r in reference}, {r["name"]: r for r in records}
    shared = [name for name in ref if name in mine]
    return {
        "reference": len(ref),
        "converted": len(mine),
        "missing": len(set(ref) - set(mine)),
        "extra": len(set(mine) - set(ref)),
        "attribute_mismatches": sum(ref[n]["attributes"] != mine[n]["attributes"] for n in shared),
        "box_count_mismatches": sum(counts(ref[n]) != counts(mine[n]) for n in shared),
    }


def main() -> None:
    """Convert one split; optionally check it against a legacy release of the same split."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", help="bdd100k_labels.zip (the 2018 per-image release) or a FiftyOne samples.json")
    parser.add_argument("split", help="train | val")
    parser.add_argument("out", help="where to write the legacy-format JSON")
    parser.add_argument(
        "--reference",
        help="a legacy bdd100k_labels_images_<split>.json to check against (and, for samples.json, to order by)",
    )
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text()) if args.reference else None
    if args.source.endswith(".json"):
        if reference is None:
            parser.error("a FiftyOne samples.json needs --reference (the records' order and attributes)")
        records = fiftyone_records(args.source, reference)
    else:
        records = legacy_records(args.source, args.split)
    Path(args.out).write_text(json.dumps(records), encoding="utf-8")
    logger.info(f"{args.split}: {len(records)} records -> {args.out}")
    if reference is not None:
        logger.info(f"against {args.reference}: {compare(records, reference)}")


if __name__ == "__main__":
    main()

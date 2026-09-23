#!/usr/bin/env python3
"""Organize the competition E-problem data without reading file contents.

Default usage from the project directory:
    python organize_data.py --source "./E题数据" --target "./DATA"

The script only inspects directory entries and file names, copies files, and
renames paths. It does not deserialize PKL files or open XLSX/MP4 contents.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


ATTACHMENT_PREFIXES = {
    1: "附件1-",
    2: "附件2-",
    3: "附件3-",
    4: "附件4-",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy and normalize the E-problem data directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("./E题数据"),
        help="Source directory containing the supplied data (default: ./E题数据).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("./DATA"),
        help="Organized output directory (default: ./DATA).",
    )
    return parser.parse_args()


def attachment_dirs(path: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not path.is_dir():
        return result
    for child in path.iterdir():
        if not child.is_dir():
            continue
        for number, prefix in ATTACHMENT_PREFIXES.items():
            if child.name.startswith(prefix):
                result[number] = child
    return result


def locate_dataset_root(source: Path) -> tuple[Path, dict[int, Path]]:
    """Find the first shallow directory containing all four attachments."""
    source = source.resolve()
    candidates = [source]
    candidates.extend(p for p in source.iterdir() if p.is_dir())
    for candidate in candidates:
        found = attachment_dirs(candidate)
        if set(found) == {1, 2, 3, 4}:
            return candidate, found
    raise FileNotFoundError(
        f"Could not find all four attachment directories under: {source}"
    )


def only_readme_exists(target: Path) -> bool:
    if not target.exists():
        return True
    entries = {p.name for p in target.iterdir()}
    return entries.issubset({"README.md"})


def copy_file(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def find_named_dir(root: Path, name: str) -> Path:
    matches = [p for p in root.rglob(name) if p.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one directory named {name!r} under {root}, "
            f"found {len(matches)}"
        )
    return matches[0]


def numeric_sample_name(filename: str) -> str:
    match = re.search(r"(\d+)(?=\.pkl$)", filename, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract numeric sample id from: {filename}")
    return f"sample_{int(match.group(1)):02d}.pkl"


def copy_attachment_1(source: Path, target: Path) -> None:
    video_roots = [
        p
        for p in source.iterdir()
        if p.is_dir() and "100" in p.name
    ]
    if len(video_roots) != 1:
        raise RuntimeError(f"Cannot uniquely locate the 100-video directory in {source}")
    video_root = video_roots[0]
    label_files = list(video_root.glob("*.xlsx"))
    if len(label_files) != 1:
        raise RuntimeError(f"Expected one XLSX label file in {video_root}")

    copy_file(label_files[0], target / "labels_100.xlsx")
    mp4_files = sorted(video_root.glob("*/*.mp4"))
    if len(mp4_files) != 100:
        raise RuntimeError(f"Expected 100 MP4 files, found {len(mp4_files)}")
    for source_file in mp4_files:
        video_id = source_file.parent.name
        copy_file(source_file, target / "videos" / video_id / source_file.name)


def copy_attachment_2(source: Path, target: Path) -> None:
    expected = {
        "aligned_50.pkl": "aligned_50.pkl",
        "unaligned_50.pkl": "unaligned_50.pkl",
        "label.xlsx": "labels.xlsx",
    }
    for old_name, new_name in expected.items():
        source_file = source / old_name
        if not source_file.is_file():
            raise FileNotFoundError(source_file)
        copy_file(source_file, target / new_name)


def copy_pkl_samples(source: Path, target: Path, expected_count: int) -> None:
    source_files = sorted(source.glob("*.pkl"))
    if len(source_files) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} PKL files in {source}, found {len(source_files)}"
        )
    for source_file in source_files:
        copy_file(source_file, target / numeric_sample_name(source_file.name))


def copy_attachment_3(source: Path, target: Path) -> None:
    aligned = find_named_dir(source, "对齐版本")
    unaligned = find_named_dir(source, "未对齐版本")
    copy_pkl_samples(aligned, target / "aligned", 30)
    copy_pkl_samples(unaligned, target / "unaligned", 30)


def copy_attachment_4_variant(source: Path, target: Path) -> None:
    copy_pkl_samples(source, target / "features", 20)
    video_dir = source / "videos"
    video_files = sorted(video_dir.glob("*.mp4"))
    if len(video_files) != 20:
        raise RuntimeError(
            f"Expected 20 MP4 files in {video_dir}, found {len(video_files)}"
        )
    for source_file in video_files:
        copy_file(
            source_file,
            target / "videos" / f"sample_{int(source_file.stem):02d}.mp4",
        )


def copy_attachment_4(source: Path, target: Path) -> None:
    # rglob intentionally removes the duplicated attachment-4 wrapper level.
    aligned = find_named_dir(source, "对齐版本")
    unaligned = find_named_dir(source, "未对齐版本")
    copy_attachment_4_variant(aligned, target / "aligned")
    copy_attachment_4_variant(unaligned, target / "unaligned")


def validate_target(target: Path) -> None:
    expected = {
        "attachment_1_raw_samples": {".mp4": 100, ".xlsx": 1},
        "attachment_2_standard_features": {".pkl": 2, ".xlsx": 1},
        "attachment_3_missing_modality": {".pkl": 60},
        "attachment_4_explainability": {".pkl": 40, ".mp4": 40},
    }
    for directory, extension_counts in expected.items():
        root = target / directory
        for extension, expected_count in extension_counts.items():
            actual_count = sum(1 for p in root.rglob(f"*{extension}") if p.is_file())
            if actual_count != expected_count:
                raise RuntimeError(
                    f"Validation failed for {root}: expected {expected_count} "
                    f"{extension} files, found {actual_count}"
                )


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if source == target or source in target.parents:
        raise ValueError("Target must not be the source directory or inside it.")
    if not only_readme_exists(target):
        raise FileExistsError(
            f"Target is not empty: {target}. Use a new/empty target directory."
        )

    dataset_root, attachments = locate_dataset_root(source)
    target.mkdir(parents=True, exist_ok=True)

    copy_attachment_1(
        attachments[1], target / "attachment_1_raw_samples"
    )
    copy_attachment_2(
        attachments[2], target / "attachment_2_standard_features"
    )
    copy_attachment_3(
        attachments[3], target / "attachment_3_missing_modality"
    )
    copy_attachment_4(
        attachments[4], target / "attachment_4_explainability"
    )
    validate_target(target)

    print(f"Source dataset root: {dataset_root}")
    print(f"Organized data written to: {target}")
    print("Validation passed: 244 data files copied; .DS_Store ignored.")


if __name__ == "__main__":
    main()

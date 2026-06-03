#!/usr/bin/env python3
"""Recursive DDP-to-FLAC extractor with ZIP traversal.

Looks in the current directory for DDP sets, including those inside ZIP
archives (with support for nested ZIPs). For every DDP found, splits the
raw CD image into individual FLAC files, using PQDESCR to get track
boundaries and CDTEXT.BIN to get disc-level album/artist.

All FLACs are written into a single ./flacs directory next to the script.
Filenames follow the pattern:

    <label>_Track_NN.flac

where <label> comes from the nearest enclosing ZIP name (or the DDP
folder name if there is no ZIP). If a filename already exists, _2, _3,
... are appended to avoid overwriting.

(C) John Warburton 2026, GNU Public Licence v. 3 applies.
"""

from __future__ import annotations

import os
import re
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root folder to scan (current directory)
DDP_ROOT = Path("./")

# Where extracted FLACs will be written (single shared folder)
OUTPUT_ROOT = Path("./flacs")

# Where ZIP contents will be extracted (remove after use)
EXTRACT_ROOT = Path("./_extracted")

# Path to the FLAC encoder binary (can be just "flac" if on PATH)
# You might need to install this binary from a suitable source.
FLAC_BIN = "flac"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    """Represents one CD track as defined by the DDP PQ / CD-Text.

    All positions are expressed relative to the start of the disc.
    `start_frames` and `end_frames` are in CD frames (75 frames per second).
    """

    number: int
    start_frames: int  # CD frames from disc start (INDEX 01 position)
    end_frames: int    # CD frames from disc start (start of next track or lead-out)

    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    isrc: Optional[str] = None

    def duration_frames(self) -> int:
        return self.end_frames - self.start_frames


# ---------------------------------------------------------------------------
# CD time helpers (MSF <-> frames / seconds)
# ---------------------------------------------------------------------------

CD_FRAMES_PER_SECOND = 75
BYTES_PER_CD_FRAME = 2352  # 588 stereo samples * 4 bytes/sample
SAMPLE_RATE = 44100
CHANNELS = 2
BITS_PER_SAMPLE = 16


def msf_to_frames(m: int, s: int, f: int) -> int:
    """Convert MM:SS:FF (75 fps) to absolute CD frames.

    frames = ((M * 60) + S) * 75 + F
    """

    return (m * 60 + s) * CD_FRAMES_PER_SECOND + f


def frames_to_seconds(frames: int) -> float:
    """Convert CD frames to seconds (float)."""

    return frames / float(CD_FRAMES_PER_SECOND)


def frames_to_byte_offset(frames: int) -> int:
    """Byte offset into the raw audio image for a given CD frame index."""

    return frames * BYTES_PER_CD_FRAME


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def safe_filename_component(text: str) -> str:
    """Make a string safe for use as part of a filename."""
    # Remove NULs just in case
    text = text.replace("\x00", "")
    # Collapse whitespace
    text = " ".join(text.split())
    # Replace problematic chars (keep letters, digits, space, _, -, ., parentheses)
    return re.sub(r"[^\w\s\-\.\(\)]", "_", text)


def unique_out_path(directory: Path, filename: str) -> Path:
    """Return a unique output path in `directory` for `filename`.

    If `directory/filename` exists, append _2, _3, ... before the suffix.
    """
    directory.mkdir(parents=True, exist_ok=True)
    base_path = directory / filename
    if not base_path.exists():
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# DDP discovery and basic structure
# ---------------------------------------------------------------------------

def looks_like_ddp(ddp_dir: Path) -> bool:
    """Heuristic: a DDP directory contains DDPID and AUDIO.DAT or IMAGE.DAT."""

    if not ddp_dir.is_dir():
        return False

    files_upper = {p.name.upper() for p in ddp_dir.iterdir() if p.is_file()}
    has_ddpid = "DDPID" in files_upper
    has_audio = any(name in files_upper for name in ("AUDIO.DAT", "IMAGE.DAT"))
    return has_ddpid and has_audio


def locate_audio_file(ddp_dir: Path) -> Path:
    """Find the raw 16-bit 44.1 kHz stereo audio file inside a DDP directory."""

    candidates = ["AUDIO.DAT", "IMAGE.DAT"]

    for name in candidates:
        p = ddp_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(f"No audio image (AUDIO.DAT / IMAGE.DAT) found in {ddp_dir}")


# ---------------------------------------------------------------------------
# PQ / CD-Text parsing (PQDESCR)
# ---------------------------------------------------------------------------

def parse_pq(ddp_dir: Path) -> List["TrackInfo"]:
    """Parse PQDESCR in a DDP directory and return a list of TrackInfo.

    PQDESCR is treated as fixed-size 64-byte records. Each relevant record:

      0-3   : b"VVVS"
      4-5   : track code (ASCII)   e.g. "00", "01", ..., "99", "AA" (lead-out)
      6-7   : index code (ASCII)   e.g. "00", "01"
      8-9   : b"  "
      10-15 : time "MMSSFF" (ASCII) minutes / seconds / frames @ 75 fps
      16-17 : flag/constant (ignored)
      18-63 : padding (ignored)

    Semantics:
      * track "00"         : disc start / index 0 – ignored for track boundaries.
      * track "01"-"99"    : audio tracks.
      * track "AA", index "01" : lead-out time for the disc.
      * index "01"         : INDEX 01 of that track – used as track start.
      * index "00"         : INDEX 00 (pregap) – currently ignored.

    Returns TrackInfo objects with only number/start/end populated; title/
    artist/album/isrc are left as None for now.
    """

    # Locate PQDESCR (try a few common variants)
    pq_candidates = [
        "PQDESCR",
        "PQDESCR.DAT",
        "pqdescr",
        "pqdescr.dat",
    ]

    pq_path: Optional[Path] = None
    for name in pq_candidates:
        candidate = ddp_dir / name
        if candidate.exists():
            pq_path = candidate
            break

    if pq_path is None:
        raise FileNotFoundError(f"No PQDESCR file found in {ddp_dir}")

    data = pq_path.read_bytes()

    # Some implementations appear to terminate the file with CRLF. If that
    # makes the length divisible by 64, strip it so we have whole records.
    if len(data) >= 2 and data[-2:] == b"\r\n" and (len(data) - 2) % 64 == 0:
        data = data[:-2]

    RECORD_SIZE = 64
    if len(data) % RECORD_SIZE not in (0,):
        # Not an exact multiple of 64; we'll still parse the complete records
        # and ignore any trailing partial bytes.
        pass

    track_entries: List[tuple[int, int]] = []  # (track_number, start_frames)
    lead_out_frames: Optional[int] = None

    # Walk records in 64-byte chunks
    num_records = len(data) // RECORD_SIZE
    for i in range(num_records):
        offset = i * RECORD_SIZE
        rec = data[offset : offset + RECORD_SIZE]

        if len(rec) < 18:
            continue

        if rec[0:4] != b"VVVS":
            continue

        try:
            track_code = rec[4:6].decode("ascii", errors="strict")
            index_code = rec[6:8].decode("ascii", errors="strict")
            time_str = rec[10:16].decode("ascii", errors="strict")
        except UnicodeDecodeError:
            continue

        if len(time_str) != 6 or not time_str.isdigit():
            continue

        mm = int(time_str[0:2])
        ss = int(time_str[2:4])
        ff = int(time_str[4:6])

        frames = msf_to_frames(mm, ss, ff)

        # Lead-out: track AA, index 01
        if track_code == "AA" and index_code == "01":
            if lead_out_frames is None or frames < lead_out_frames:
                lead_out_frames = frames
            continue

        # Disc start / index 0
        if track_code == "00":
            continue

        # Only numeric track codes 01-99 are considered audio tracks
        if not track_code.isdigit():
            continue

        track_num = int(track_code)

        # We take only INDEX 01 as track start
        if index_code == "01":
            track_entries.append((track_num, frames))

    if not track_entries:
        raise ValueError(f"No track INDEX 01 entries found in {pq_path}")

    # Sort entries by their frame position (start time), not just track number,
    # in case of non-sequential or hidden tracks.
    track_entries.sort(key=lambda tf: tf[1])

    if lead_out_frames is None:
        raise ValueError(f"No lead-out (AA01) record found in {pq_path}")

    tracks: List["TrackInfo"] = []
    for idx, (track_num, start_frames) in enumerate(track_entries):
        if idx + 1 < len(track_entries):
            end_frames = track_entries[idx + 1][1]
        else:
            end_frames = lead_out_frames

        if end_frames <= start_frames:
            raise ValueError(
                f"Non-positive duration for track {track_num}: "
                f"start={start_frames}, end={end_frames}"
            )

        tracks.append(
            TrackInfo(
                number=track_num,
                start_frames=start_frames,
                end_frames=end_frames,
                title=None,
                artist=None,
                album=None,
                isrc=None,
            )
        )

    return tracks


# ---------------------------------------------------------------------------
# Parse CD-TEXT from CDTEXT.BIN (disc-level only for now)
# ---------------------------------------------------------------------------

def _read_cdtext_packs(cdtext_path: Path) -> List[bytes]:
    """Read CDTEXT.BIN and return a list of 18-byte CD-Text packs."""
    data = cdtext_path.read_bytes()

    PACK_SIZE = 18
    num_packs = len(data) // PACK_SIZE
    packs = []

    for i in range(num_packs):
        start = i * PACK_SIZE
        pack = data[start : start + PACK_SIZE]
        if len(pack) == PACK_SIZE:
            packs.append(pack)

    return packs


def _collect_cdtext_strings(packs: List[bytes]) -> Dict[tuple[int, int], str]:
    """Group CD-Text packs by (type, track) and reconstruct strings.

    Very simple version:
      * Concatenate payloads per (pack_type, track_no).
      * In each pack, only keep bytes up to the first NUL (0x00).
      * Ignore character-position inheritance and multi-string-per-pack
        rules for now.
    """

    # (pack_type, track) -> list of payload byte chunks (in file order)
    chunks: Dict[tuple[int, int], List[bytes]] = {}

    for p in packs:
        pack_type = p[0]
        track_no = p[1]
        payload = p[4:16]

        # Keep only up to the first NUL; anything after is for the next text
        nul_pos = payload.find(b"\x00")
        if nul_pos != -1:
            payload = payload[:nul_pos]

        if not payload:
            continue

        key = (pack_type, track_no)
        chunks.setdefault(key, []).append(payload)

    strings: Dict[tuple[int, int], str] = {}

    for key, payload_list in chunks.items():
        combined = b"".join(payload_list)
        combined = combined.strip()
        text = combined.decode("latin-1", errors="ignore")
        strings[key] = text

    return strings


def apply_cdtext_metadata(ddp_dir: Path, tracks: List["TrackInfo"]) -> None:
    """Read CDTEXT.BIN and update TrackInfo metadata in-place.

    For now we only use disc-level info:
      - ALBUM from disc TITLE (type 0x80, track 0)
      - ARTIST from disc PERFORMER (type 0x81, track 0), if present
    """

    cdtext_candidates = ["CDTEXT.BIN", "cdtext.bin"]
    cdtext_path: Optional[Path] = None
    for name in cdtext_candidates:
        candidate = ddp_dir / name
        if candidate.exists():
            cdtext_path = candidate
            break

    if cdtext_path is None:
        return

    packs = _read_cdtext_packs(cdtext_path)
    strings = _collect_cdtext_strings(packs)

    disc_title = strings.get((0x80, 0))      # album title
    disc_performer = strings.get((0x81, 0))  # disc artist

    print("CD-Text disc title:", disc_title)
    print("CD-Text disc performer:", disc_performer)

    for t in tracks:
        if disc_title:
            t.album = disc_title
        if disc_performer:
            t.artist = disc_performer


# ---------------------------------------------------------------------------
# FLAC encoding helpers
# ---------------------------------------------------------------------------

def _frames_to_flac_time(frames: int) -> str:
    """Convert CD frames (75 fps) to a flac mm:ss.ss time string.

    We work in integer centiseconds and floor, so the resulting time
    is always <= the true position, avoiding 'after end of input' errors.
    """
    if frames < 0:
        frames = 0

    # 1 frame = 1/75 s -> 100 centiseconds = 1 s
    total_centis = (frames * 100) // CD_FRAMES_PER_SECOND

    minutes = total_centis // (60 * 100)
    rem_centis = total_centis - minutes * 60 * 100
    seconds = rem_centis // 100
    centis = rem_centis % 100

    return f"{minutes:d}:{seconds:02d}.{centis:02d}"


def build_flac_command(audio_path: Path, track: TrackInfo, out_path: Path) -> List[str]:
    """Construct the command-line for FLAC to encode a single track.

    Input: raw PCM, 16-bit, 44.1 kHz, stereo, little-endian.
    We use --skip (absolute, from disc start) and --until=+... (relative
    to skip), with times derived directly from CD frames.
    """

    start_frames = track.start_frames
    end_frames = track.end_frames
    duration_frames = end_frames - start_frames

    if duration_frames <= 0:
        raise ValueError(
            f"Track {track.number} has non-positive duration in frames: "
            f"{duration_frames}"
        )

    skip_str = _frames_to_flac_time(start_frames)
    until_str = _frames_to_flac_time(duration_frames)

    cmd: List[str] = [
        FLAC_BIN,
        "--force-raw-format",
        "--endian=little",
        "--sign=signed",
        f"--channels={CHANNELS}",
        f"--bps={BITS_PER_SAMPLE}",
        f"--sample-rate={SAMPLE_RATE}",
        f"--skip={skip_str}",
        f"--until=+{until_str}",
        "-o",
        str(out_path),
        str(audio_path),
    ]

    # Attach tags if available
    tag_args: List[str] = []
    if track.title:
        tag_args.append(f"--tag=TITLE={track.title}")
    else:
        tag_args.append(f"--tag=TITLE=Track {track.number:02d}")
    if track.artist:
        tag_args.append(f"--tag=ARTIST={track.artist}")
    if track.album:
        tag_args.append(f"--tag=ALBUM={track.album}")
    if track.isrc:
        tag_args.append(f"--tag=ISRC={track.isrc}")

    # Insert tag arguments just before "-o"
    cmd[8:8] = tag_args

    return cmd


def extract_track(audio_path: Path, track: TrackInfo, label: str) -> None:
    """Run FLAC to encode one track from the raw DDP audio image.

    Filenames follow: <label>_Track_NN.flac, with label derived from
    ZIP/DDP context, and uniqueness enforced.
    """

    label_component = safe_filename_component(label or track.album or "Unknown")
    filename = f"{label_component}_Track_{track.number:02d}.flac"
    out_path = unique_out_path(OUTPUT_ROOT, filename)

    title_part = track.title or f"Track {track.number:02d}"

    cmd = build_flac_command(audio_path, track, out_path)

    print(f"Encoding track {track.number:02d}: {title_part}")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# High-level processing for one DDP
# ---------------------------------------------------------------------------

def process_ddp_set(ddp_dir: Path, label: str) -> None:
    """Process a single DDP directory: parse PQ, then extract all tracks."""

    print(f"Processing DDP: {ddp_dir} (label: {label})")

    audio_path = locate_audio_file(ddp_dir)
    tracks = parse_pq(ddp_dir)

    if not tracks:
        print(f"  No tracks found in PQ for {ddp_dir}")
        return

    # Enrich tracks with CD-Text disc-level metadata, if present
    apply_cdtext_metadata(ddp_dir, tracks)

    for track in tracks:
        extract_track(audio_path, track, label)


# ---------------------------------------------------------------------------
# Recursive traversal of directories and ZIPs
# ---------------------------------------------------------------------------

def walk_and_process(root: Path, label_prefix: Optional[str] = None) -> None:
    """Recursively scan `root` for DDP sets and ZIPs.

    * If a directory looks like a DDP, process it with the current label prefix
      (or the directory name if no prefix).
    * If a ZIP file is found, extract it under EXTRACT_ROOT and recurse into it
      with an updated label prefix incorporating the ZIP's stem.
    """

    for entry in root.iterdir():
        # Skip our own output and extraction directories to avoid loops
        if entry.name in {OUTPUT_ROOT.name, EXTRACT_ROOT.name}:
            continue

        if entry.is_dir():
            if looks_like_ddp(entry):
                ddp_label = label_prefix or entry.name
                process_ddp_set(entry, ddp_label)
            else:
                walk_and_process(entry, label_prefix)

        elif entry.is_file() and entry.suffix.lower() == ".zip":
            zip_label = entry.stem
            new_label_prefix = f"{label_prefix}_{zip_label}" if label_prefix else zip_label

            # Choose a unique extraction directory for this ZIP
            base_extract_dir = EXTRACT_ROOT / zip_label
            extract_dir = base_extract_dir
            counter = 2
            while extract_dir.exists():
                extract_dir = EXTRACT_ROOT / f"{zip_label}_{counter}"
                counter += 1

            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting ZIP {entry} -> {extract_dir}")

            with zipfile.ZipFile(entry, "r") as zf:
                zf.extractall(extract_dir)

            walk_and_process(extract_dir, new_label_prefix)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    if not DDP_ROOT.exists():
        raise SystemExit(f"DDP_ROOT does not exist: {DDP_ROOT}")

    print(f"Scanning for DDPs and ZIPs under {DDP_ROOT.resolve()}")
    walk_and_process(DDP_ROOT)
    print("Done.")


if __name__ == "__main__":
    main()

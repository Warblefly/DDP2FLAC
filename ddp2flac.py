#!/usr/bin/env python3
"""Recursive DDP-to-FLAC extractor with ZIP traversal.

Looks in the current directory for DDP sets, including those inside ZIP
archives (with support for nested ZIPs). For every DDP found, splits the
raw CD image into individual FLAC files, using PQDESCR to get track
boundaries and CDTEXT.BIN (if present) to get disc-level album/artist.

All FLACs are written into a single ./flacs directory next to the script.
Filenames follow the pattern:

    <label>_Track_NN.flac

where <label> comes from the nearest enclosing ZIP name (or the DDP
folder name if there is no ZIP). If a filename already exists, _2, _3,
... are appended to avoid overwriting.

Typical usage:
    - Put this script in a folder that contains student ZIPs and/or
      unpacked DDP folders.
    - Run:  python3 ddp2flacs.py
    - Look in ./flacs for the resulting FLAC files.

Key assumptions:
    * DDP directories contain:
        - a file named "DDPID" (case-insensitive), and
        - an audio image named AUDIO.DAT or IMAGE.DAT (16-bit, 44.1 kHz,
          stereo, little-endian, CD sector layout 2352 bytes/frame).
    * Track boundaries are defined in PQDESCR / PQDESCR.DAT as fixed
      64-byte records in the format you provided.
    * CDTEXT.BIN, if present, contains standard CD-Text pack data; here
      we only read disc TITLE (album) and PERFORMER (artist).

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

# Root folder to scan. Changing this lets you point the script somewhere
# else, but in typical use you just run the script in the folder that
# contains ZIPs / DDPs and leave this as "./".
DDP_ROOT = Path("./")

# Where extracted FLACs will be written.
# NOTE: For this recursive version we use a *single* shared folder rather
# than one subfolder per DDP. Filenames are prefixed with a label derived
# from ZIP/DDP context, and uniqueness is enforced.
OUTPUT_ROOT = Path("./flacs")

# Where ZIP contents will be extracted. If you want to clean up after
# processing, you can safely delete this folder.
EXTRACT_ROOT = Path("./_extracted")

# Path to the FLAC encoder binary (can be just "flac" if on PATH).
# On Windows, you might use "flac.exe"; on macOS/Linux just "flac".
FLAC_BIN = "flac"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    """Represents one CD track as defined by the DDP PQ / CD-Text.

    All positions are expressed relative to the start of the disc.
    `start_frames` and `end_frames` are in CD frames (75 frames/second).

    Fields:
        number        : Track number (1-based, as on the CD).
        start_frames  : INDEX 01 position from disc start, in frames.
        end_frames    : Start of the next track, or lead-out, in frames.
        title         : Track title (unused for now, we set generic "Track nn").
        artist        : Track artist/performer (disc-level from CD-Text).
        album         : Album title (disc-level from CD-Text).
        isrc          : ISRC code if available (not populated here).
    """

    number: int
    start_frames: int
    end_frames: int

    title: Optional[str] = None
    artist: Optional[str] = None
    album: Optional[str] = None
    isrc: Optional[str] = None

    def duration_frames(self) -> int:
        """Return track duration in CD frames (end_frames - start_frames)."""

        return self.end_frames - self.start_frames


# ---------------------------------------------------------------------------
# CD time helpers (MSF <-> frames / seconds)
# ---------------------------------------------------------------------------

# Constants for standard audio CD format.
CD_FRAMES_PER_SECOND = 75
BYTES_PER_CD_FRAME = 2352  # 588 stereo samples * 4 bytes/sample
SAMPLE_RATE = 44100
CHANNELS = 2
BITS_PER_SAMPLE = 16


def msf_to_frames(m: int, s: int, f: int) -> int:
    """Convert MM:SS:FF (75 fps) to absolute CD frames.

    frames = ((M * 60) + S) * 75 + F

    Used when parsing PQDESCR MMSSFF timestamps.
    """

    return (m * 60 + s) * CD_FRAMES_PER_SECOND + f


def frames_to_seconds(frames: int) -> float:
    """Convert CD frames to seconds (float).

    Not used directly for FLAC timing here (we work in frames -> centiseconds),
    but kept for completeness.
    """

    return frames / float(CD_FRAMES_PER_SECOND)


def frames_to_byte_offset(frames: int) -> int:
    """Byte offset into the raw audio image for a given CD frame index.

    Not used in this script because we let FLAC do the slicing, but useful
    if you ever want to read/write raw PCM slices yourself.
    """

    return frames * BYTES_PER_CD_FRAME


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def safe_filename_component(text: str) -> str:
    """Sanitise a string for use as part of a filename.

    - Removes NULs.
    - Collapses whitespace to single spaces.
    - Replaces characters outside [letters, digits, space, _, -, ., ()]
      with underscores.

    Use this for labels (e.g. ZIP/DDP names) and album titles before
    embedding them in filenames.
    """

    text = text.replace("\x00", "")
    text = " ".join(text.split())
    return re.sub(r"[^\w\s\-\.\(\)]", "_", text)


def unique_out_path(directory: Path, filename: str) -> Path:
    """Return a unique output path in `directory` for `filename`.

    If `directory/filename` exists, append "_2", "_3", ... before the
    suffix to avoid overwriting.
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
    """Heuristic: does `ddp_dir` look like a DDP set?

    We consider a directory to be a DDP set if:
        - It is a directory; and
        - It contains a file named "DDPID" (any case); and
        - It contains an audio image named "AUDIO.DAT" or "IMAGE.DAT".

    This matches the typical structure of DDP masters produced by common
    authoring tools.
    """

    if not ddp_dir.is_dir():
        return False

    files_upper = {p.name.upper() for p in ddp_dir.iterdir() if p.is_file()}
    has_ddpid = "DDPID" in files_upper
    has_audio = any(name in files_upper for name in ("AUDIO.DAT", "IMAGE.DAT"))
    return has_ddpid and has_audio


def locate_audio_file(ddp_dir: Path) -> Path:
    """Find the raw 16-bit 44.1 kHz stereo audio file inside a DDP directory.

    We look for AUDIO.DAT first, then IMAGE.DAT. Adjust this if your
    authoring software uses different names.
    """

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

      0-3   : b"VVVS"                       (magic)
      4-5   : track code (ASCII)   e.g. "00", "01", ..., "99", "AA" (lead-out)
      6-7   : index code (ASCII)   e.g. "00", "01"
      8-9   : b"  " (two spaces)
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
    artist/album/isrc are left for CD-Text or other metadata sources.
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

    # Some implementations terminate with CRLF. If stripping CRLF leaves an
    # exact multiple of 64-byte records, drop it so we parse whole records.
    if len(data) >= 2 and data[-2:] == b"\r\n" and (len(data) - 2) % 64 == 0:
        data = data[:-2]

    RECORD_SIZE = 64
    # If the file is not an exact multiple of 64, we still parse all full
    # records and silently ignore trailing bytes.

    track_entries: List[tuple[int, int]] = []  # (track_number, start_frames)
    lead_out_frames: Optional[int] = None

    # Walk records in 64-byte chunks
    num_records = len(data) // RECORD_SIZE
    for i in range(num_records):
        offset = i * RECORD_SIZE
        rec = data[offset : offset + RECORD_SIZE]

        if len(rec) < 18:
            # Too short to contain required fields; skip
            continue

        if rec[0:4] != b"VVVS":
            # Not a PQ timing record
            continue

        try:
            track_code = rec[4:6].decode("ascii", errors="strict")
            index_code = rec[6:8].decode("ascii", errors="strict")
            time_str = rec[10:16].decode("ascii", errors="strict")
        except UnicodeDecodeError:
            # Skip malformed records
            continue

        if len(time_str) != 6 or not time_str.isdigit():
            # Unexpected time format
            continue

        mm = int(time_str[0:2])
        ss = int(time_str[2:4])
        ff = int(time_str[4:6])

        frames = msf_to_frames(mm, ss, ff)

        # Lead-out: track AA, index 01
        if track_code == "AA" and index_code == "01":
            # Use the earliest AA01 we see as lead-out
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

    # Sort entries by frame position (start time), not just track number,
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
    """Read CDTEXT.BIN and return a list of raw 18-byte CD-Text packs.

    We do not validate CRCs here; for our purposes it's enough to split
    the file into packs and use the payload bytes.
    """

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

    This is enough to retrieve the disc TITLE and PERFORMER cleanly in
    your example, but not robust enough for all CD-Text encodings.
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

    All tracks from the same DDP are given the same album and artist.
    Track titles remain generic ("Track nn") until/unless a more
    sophisticated CD-Text track-title parser is implemented.
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

    We work in integer centiseconds and floor, so the resulting time is
    always <= the true position, avoiding 'after end of input' errors
    for the last track.
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

    Input is raw PCM:
        - 16-bit
        - 44.1 kHz
        - stereo
        - little-endian

    We call flac with:
        --force-raw-format --endian=little --sign=signed
        --channels=2 --bps=16 --sample-rate=44100
        --skip=mm:ss.ss --until=+mm:ss.ss

    `--skip` is the offset from the start of the raw image;
    `--until=+...` is the duration relative to that offset.
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

    # Attach tags if available. If no title, use a generic "Track NN".
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

    # Insert tag arguments just before "-o" so they are applied to this file.
    cmd[8:8] = tag_args

    return cmd


def extract_track(audio_path: Path, track: TrackInfo, label: str) -> None:
    """Run FLAC to encode one track from the raw DDP audio image.

    Filenames follow: <label>_Track_NN.flac, with `label` derived from
    ZIP/DDP context (or album name if label is empty), and uniqueness
    enforced by `unique_out_path`.
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
    """Process a single DDP directory: parse PQ, then extract all tracks.

    `label` is used as the filename prefix for all tracks from this DDP.
    It is typically derived from the nearest enclosing ZIP name, or falls
    back to the DDP directory name.
    """

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

    Behaviour:
      * If a directory looks like a DDP, process it with the current
        `label_prefix` (or the directory name if `label_prefix` is None).
      * If a ZIP file is found, extract it under EXTRACT_ROOT and recurse
        into it with an updated label prefix incorporating the ZIP's stem.
      * For ordinary directories that are not DDPs, recurse into them with
        the same `label_prefix`.

    This effectively walks nested ZIP trees and applies `process_ddp_set`
    to every DDP it finds, building a label that reflects the ZIP nesting
    (e.g. OuterZip_InnerZip_Track_01.flac).
    """

    for entry in root.iterdir():
        # Skip our own output and extraction directories to avoid loops
        if entry.name in {OUTPUT_ROOT.name, EXTRACT_ROOT.name}:
            continue

        if entry.is_dir():
            if looks_like_ddp(entry):
                # Use the label prefix if available, otherwise the directory name
                ddp_label = label_prefix or entry.name
                process_ddp_set(entry, ddp_label)
            else:
                # Recurse into non-DDP directories without changing the label
                walk_and_process(entry, label_prefix)

        elif entry.is_file() and entry.suffix.lower() == ".zip":
            # Found a ZIP: build a new label prefix and extract it
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

            # Now scan inside the extracted ZIP, carrying the new label
            walk_and_process(extract_dir, new_label_prefix)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: prepare folders and kick off recursive scan.

    You normally just run this script from a shell:

        python3 ddp2flacs.py

    It will:
      * Create ./flacs and ./_extracted if needed.
      * Scan DDP_ROOT (default: current directory) recursively.
      * Extract nested ZIPs as needed.
      * Convert every DDP it finds into FLAC tracks.
    """

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

    if not DDP_ROOT.exists():
        raise SystemExit(f"DDP_ROOT does not exist: {DDP_ROOT}")

    print(f"Scanning for DDPs and ZIPs under {DDP_ROOT.resolve()}")
    walk_and_process(DDP_ROOT)
    print("Done.")


if __name__ == "__main__":
    main()

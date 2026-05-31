#!/usr/bin/env python3
"""
Looks in current directory for a DDP, which is the file folder format
used to master audio CDs; then splits the audio correctly into separate
FLAC files.

(C) John Warburton 2026, GNU Public Licence v. 3 applies.

No command line parameters yet!
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Root folder that contains one subfolder per DDP set
DDP_ROOT = Path("./")

# Where extracted FLACs will be written. Each DDP gets its own subfolder.
OUTPUT_ROOT = Path("./flacs")

# Path to the FLAC encoder binary (can be just "flac" if on PATH)
FLAC_BIN = "flac.exe"

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrackInfo:
    """Represents one CD track as defined by the DDP PQ / CD‑Text.

    All positions are expressed relative to the start of the disc.
    `start_frames` and `end_frames` are in CD frames (75 frames per second).
    """

    number: int
    start_frames: int  # CD frames from disc start (INDEX 01 position)
    end_frames: int    # CD frames from disc start (start of next track or lead‑out)

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

    F = ((M * 60) + S) * 75 + F
    """

    return (m * 60 + s) * CD_FRAMES_PER_SECOND + f


def frames_to_seconds(frames: int) -> float:
    """Convert CD frames to seconds (float)."""

    return frames / float(CD_FRAMES_PER_SECOND)


def frames_to_byte_offset(frames: int) -> int:
    """Byte offset into the raw audio image for a given CD frame index."""

    return frames * BYTES_PER_CD_FRAME


# ---------------------------------------------------------------------------
# DDP discovery and basic structure
# ---------------------------------------------------------------------------

def find_ddp_sets(root: Path) -> List[Path]:
    """Return a list of directories that look like DDP sets.

    Heuristic: a DDP directory usually contains a file named "DDPID" and
    some form of raw audio image (AUDIO.DAT or IMAGE.DAT).
    """

    ddp_dirs: List[Path] = []

    for dirpath, _dirnames, filenames in os.walk(root):
        files_upper = {fn.upper() for fn in filenames}
        has_ddpid = "DDPID" in files_upper
        has_audio = any(name in files_upper for name in ("AUDIO.DAT", "IMAGE.DAT"))

        if has_ddpid and has_audio:
            ddp_dirs.append(Path(dirpath))

    return ddp_dirs


def locate_audio_file(ddp_dir: Path) -> Path:
    """Find the raw 16‑bit 44.1 kHz stereo audio file inside a DDP directory.
    """

    candidates = ["AUDIO.DAT", "IMAGE.DAT"]

    for name in candidates:
        p = ddp_dir / name
        if p.exists():
            return p

    raise FileNotFoundError(f"No audio image (AUDIO.DAT / IMAGE.DAT) found in {ddp_dir}")


# ---------------------------------------------------------------------------
# PQ / CD‑Text parsing (project‑specific)
# ---------------------------------------------------------------------------

def parse_pq(ddp_dir: Path) -> List["TrackInfo"]:
    """Parse PQDESCR in a DDP directory and return a list of TrackInfo.

    The PQDESCR file is treated as a sequence of fixed-size 64-byte records.
    Each relevant record has the form (byte offsets, 0-based):

        0-3   : b"VVVS"               (magic)
        4-5   : track code (ASCII)    e.g. b"00", b"01", ..., b"99", b"AA" (lead-out)
        6-7   : index code (ASCII)    e.g. b"00", b"01"
        8-9   : two spaces            b"  "
        10-15 : time MMSSFF (ASCII)   minutes / seconds / frames @ 75 fps
        16-17 : flag/constant (e.g. b"01") – ignored
        18-63 : padding – ignored

    We use only records with magic "VVVS". Semantics:
      * track "00"        : disc start / index 0 – ignored for track boundaries.
      * track "01"-"99"   : audio tracks.
      * track "AA", index "01" : lead-out time for the disc.
      * index "01"        : INDEX 01 of that track – used as track start.
      * index "00"        : INDEX 00 (pregap) – currently ignored.

    For each track with an INDEX 01 entry, we compute its start frame.
    The end frame of track N is the start frame of track N+1, or the
    lead-out frame for the last track.

    This returns TrackInfo objects with only number/start/end populated;
    title/artist/album/isrc are left as None for now.
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

    # Sort entries by their frame position (start time), not just track number,
    # in case of non-sequential or hidden tracks.
    track_entries.sort(key=lambda tf: tf[1])

    if lead_out_frames is None:
        raise ValueError(f"No lead-out (AA01) record found in {pq_path}")

    # Now build TrackInfo objects
    
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
# Parse CD-TEXT from CDTEXT.BIN
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
      * Ignore the character-position inheritance and multi-string-per-pack
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

        # Skip empty payloads
        if not payload:
            continue

        key = (pack_type, track_no)
        chunks.setdefault(key, []).append(payload)

    strings: Dict[tuple[int, int], str] = {}

    for key, payload_list in chunks.items():
        combined = b"".join(payload_list)

        # Strip outer spaces
        combined = combined.strip()

        text = combined.decode("latin-1", errors="ignore")
        strings[key] = text

    return strings


def apply_cdtext_metadata(ddp_dir: Path, tracks: List["TrackInfo"]) -> None:
    cdtext_candidates = ["CDTEXT.BIN", "cdtext.bin"]
    cdtext_path = None
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

    # Debug: show what we picked up
    print("CD-Text disc title:", disc_title)
    print("CD-Text disc performer:", disc_performer)

    for t in tracks:
        if disc_title:
            t.album = disc_title
        if disc_performer:
            t.artist = disc_performer

# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def ensure_output_dir_for_ddp(ddp_dir: Path) -> Path:
    """Create (if needed) an output subdirectory for this DDP set."""

    out_dir = OUTPUT_ROOT / ddp_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def safe_filename_component(text: str) -> str:
    """Make a string safe for use as part of a filename."""
    # Remove NULs just in case
    text = text.replace("\x00", "")
    # Collapse whitespace
    text = " ".join(text.split())
    # Replace problematic chars (keep letters, digits, space, _, -, ., parentheses)
    text = re.sub(r"[^\w\s\-\.\(\)]", "_", text)
    return text

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
    # centiseconds = floor(frames * (100 / 75))
    total_centis = (frames * 100) // CD_FRAMES_PER_SECOND  # CD_FRAMES_PER_SECOND = 75

    minutes = total_centis // (60 * 100)
    rem_centis = total_centis - minutes * 60 * 100
    seconds = rem_centis // 100
    centis = rem_centis % 100

    # e.g. "0:02.00", "4:10.47"
    return f"{minutes:d}:{seconds:02d}.{centis:02d}"


def build_flac_command(audio_path: Path, track: TrackInfo, out_path: Path) -> List[str]:
    """Construct the command‑line for FLAC to encode a single track.

    Input: raw PCM, 16‑bit, 44.1 kHz, stereo, little‑endian.
    We use --skip (absolute, from disc start) and --until=+... (relative
    to skip), with times derived directly from CD frames.
    """

    start_frames = track.start_frames
    end_frames = track.end_frames
    duration_frames = end_frames - start_frames

    if duration_frames <= 0:
        raise ValueError(
            f"Track {track.number} has non‑positive duration in frames: "
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
    if track.artist:
        tag_args.append(f"--tag=ARTIST={track.artist}")
    if track.album:
        tag_args.append(f"--tag=ALBUM={track.album}")
    if track.isrc:
        tag_args.append(f"--tag=ISRC={track.isrc}")

    # Insert tag arguments just before "-o"
    # (index 8 here, after the sample‑format options)
    cmd[8:8] = tag_args

    return cmd


def extract_track(audio_path: Path, track: TrackInfo, out_dir: Path) -> None:
    """Run FLAC to encode one track from the raw DDP audio image."""

    # Generic title tag for now
    title_part = track.title or f"Track {track.number:02d}"

    # Album title for filename prefix
    album_name = track.album or "Unknown Album"
    album_component = safe_filename_component(album_name)

    # Desired pattern: %ALBUM_TITLE%_Track_nn.flac
    filename = f"{album_component}_Track_{track.number:02d}.flac"
    out_path = out_dir / filename

    cmd = build_flac_command(audio_path, track, out_path)

    print(f"Encoding track {track.number:02d}: {title_part}")
    print(" ", " ".join(cmd))

    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# High‑level processing
# ---------------------------------------------------------------------------

def process_ddp_set(ddp_dir: Path) -> None:
    """Process a single DDP directory: parse PQ, then extract all tracks."""

    print(f"Processing DDP: {ddp_dir}")

    audio_path = locate_audio_file(ddp_dir)
    tracks = parse_pq(ddp_dir)

    if not tracks:
        print(f"  No tracks found in PQ for {ddp_dir}")
        return
    
    # enrich tracks with CD-TEXT metadata, if present
    apply_cdtext_metadata(ddp_dir, tracks)

    out_dir = ensure_output_dir_for_ddp(ddp_dir)

    for track in tracks:
        extract_track(audio_path, track, out_dir)


def main() -> None:
    if not DDP_ROOT.exists():
        raise SystemExit(f"DDP_ROOT does not exist: {DDP_ROOT}")

    ddp_dirs = find_ddp_sets(DDP_ROOT)

    if not ddp_dirs:
        print(f"No DDP sets found under {DDP_ROOT}")
        return

    print(f"Found {len(ddp_dirs)} DDP set(s) under {DDP_ROOT}")

    for ddp_dir in ddp_dirs:
        try:
            process_ddp_set(ddp_dir)
        except NotImplementedError as e:
            # Expected until parse_pq is implemented
            print(f"Skipping {ddp_dir}: {e}")
        except Exception as e:
            print(f"Error processing {ddp_dir}: {e}")


if __name__ == "__main__":
    main()

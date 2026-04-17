#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def last_line_contains_cancelled(file_path: Path) -> bool:
	"""Return True if the file's last non-empty line contains 'CANCELLED'."""
	with file_path.open("r", encoding="utf-8", errors="replace") as f:
		lines = f.read().splitlines()

	if not lines:
		return False

	# Check the final line as requested; trim surrounding whitespace.
	return not ("metrics saved to" in lines[-1].strip())


def find_cancelled_indices(base_dir: Path, start: int, end: int) -> list[int]:
	cancelled = []
	for i in range(start, end + 1):
		file_name = f"slurm-59453213_{i}.out"
		file_path = base_dir / file_name
		if file_path.is_file() and last_line_contains_cancelled(file_path):
			cancelled.append(i)
	return cancelled


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Find i values where slurm-59453213_{i}.out has 'out of range' in the last line."
		)
	)
	parser.add_argument(
		"--dir",
		default=".",
		help="Directory containing the slurm output files (default: current directory).",
	)
	parser.add_argument("--start", type=int, default=1, help="Start index (default: 1).")
	parser.add_argument("--end", type=int, default=720, help="End index (default: 720).")
	args = parser.parse_args()

	base_dir = Path(args.dir)
	cancelled = find_cancelled_indices(base_dir=base_dir, start=args.start, end=args.end)
	print(",".join(str(i) for i in cancelled))


if __name__ == "__main__":
	main()

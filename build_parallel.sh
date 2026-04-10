#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

usage() {
	cat <<'EOF'
Usage: ./build_parallel.sh [--clean] [--help]

Options:
  --clean    Remove the existing build directory before configuring and building.
  --help     Show this help message.
EOF
}

clean_build=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--clean)
			clean_build=true
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		*)
			echo "Unknown option: $1" >&2
			usage >&2
			exit 1
			;;
	esac
done

cd "$SCRIPT_DIR"

if [[ "$clean_build" == true ]]; then
	rm -rf "$BUILD_DIR"
fi

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --parallel
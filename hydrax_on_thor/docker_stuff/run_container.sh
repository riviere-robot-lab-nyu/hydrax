#!/usr/bin/env bash

set -euo pipefail
IMAGE="hydrax-thor:base"
WS="$(cd "$(dirname "$0")" && pwd)"

docker run --rm -it \
	--runtime nvidia \
	--shm-size=1g \
	--network host \
	-v "${WS}/hydrax:/workspace/hydrax" \
	"${IMAGE}" \
	"${@:-bash}"

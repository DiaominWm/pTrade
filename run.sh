#!/bin/bash
cd "$(dirname "$0")"
if [ -z "$1" ]; then
    echo "用法: ./run.sh <脚本路径> [参数...]"
    exit 1
fi
SCRIPT=$1
shift
docker run --rm -v "$(pwd):/app" ptrade-py312 python "$SCRIPT" "$@"
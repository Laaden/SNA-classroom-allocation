#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
tmpfile=$(mktemp "$SCRIPT_DIR/compile_trace_XXXXXX.jl")
trap 'rm -f "$tmpfile"' EXIT

CMD='using GNNProject; GNNProject.julia_main()'

# Generate precompile trace
# This makes the produced executable not require JIT
# compilation = faster CLI executable
julia --project="$SCRIPT_DIR/.." \
  --trace-compile="$tmpfile" \
  -e "$CMD" -- --stdin < "$SCRIPT_DIR/test_input.json" > /dev/null

# Compile the GNNProject into an executable
julia --project="$SCRIPT_DIR/.." <<EOF
using PackageCompiler
create_app(
    "$SCRIPT_DIR/..",
    "$SCRIPT_DIR/../build/GNNWorker";
    force=true,
    precompile_statements_file=["$tmpfile"],
    sysimage_build_args = \`--strip-metadata\`,
    include_transitive_dependencies=false
)
EOF

find "$SCRIPT_DIR/../build/GNNWorker" -type f -executable -exec strip --strip-unneeded {} \;  >/dev/null 2>&1

## Copy the trained model into the assets folder

ASSETS_DIR="$SCRIPT_DIR/../build/GNNWorker/assets"
MODEL_SRC="$SCRIPT_DIR/../output/models/model.bson"
MODEL_DEST="$ASSETS_DIR/model.bson"
mkdir -p "$ASSETS_DIR"

if [ ! -f "$MODEL_SRC" ]; then
  echo "❌ Error: Model file not found at $MODEL_SRC"
  exit 1
fi

cp "$MODEL_SRC" "$MODEL_DEST"
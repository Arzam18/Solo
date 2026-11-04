#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

NDK_ROOT="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}"
if [[ -z "$NDK_ROOT" && -n "${ANDROID_SDK_ROOT:-}" ]]; then
    NDK_ROOT="$ANDROID_SDK_ROOT/ndk/27.2.12479018"
fi
if [[ -z "$NDK_ROOT" && -n "${ANDROID_HOME:-}" ]]; then
    NDK_ROOT="$ANDROID_HOME/ndk/27.2.12479018"
fi
if [[ -z "$NDK_ROOT" || ! -d "$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin" ]]; then
    echo "ERROR: Android NDK r27c not found. Set ANDROID_NDK_HOME to its installation directory." >&2
    exit 1
fi

TOOLCHAIN="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin"
CXX="$TOOLCHAIN/aarch64-linux-android24-clang++"
READELF="$TOOLCHAIN/llvm-readelf"
OBJDUMP="$TOOLCHAIN/llvm-objdump"
NM="$TOOLCHAIN/llvm-nm"
STRIP="$TOOLCHAIN/llvm-strip"

for tool in "$CXX" "$READELF" "$OBJDUMP" "$NM" "$STRIP"; do
    if [[ ! -x "$tool" ]]; then
        echo "ERROR: Required NDK tool is missing or not executable: $tool" >&2
        exit 1
    fi
done

BUILD_DIR="$ROOT/build/android-arm64"
OBJ_DIR="$BUILD_DIR/obj"
DIST_DIR="$ROOT/dist"
OUT="$DIST_DIR/Solo-Android-ARM64-NEON"
mkdir -p "$OBJ_DIR" "$DIST_DIR"
rm -f "$OBJ_DIR"/*.o "$OUT" "$BUILD_DIR"/final.disasm "$BUILD_DIR"/symbols.txt

COMMON_FLAGS=(
    -std=c++23
    -O3
    -DNDEBUG
    -ffast-math
    -flto
    -march=armv8-a+simd
    -fno-strict-aliasing
    -pthread
)

SOURCES=(
    main.cpp
    uci.cpp
    board.cpp
    movegen.cpp
    movepicker.cpp
    search.cpp
    evaluation.cpp
    bitboard.cpp
    history.cpp
    nnue.cpp
    datagen.cpp
)
OBJECTS=()

printf '\n=== Building Solo Android ARM64 NEON (API 24, NDK r27c) ===\n'
printf 'Source: %s\n' "$ROOT"
printf 'Compiler: %s\n' "$CXX"
printf 'Network: embedded in nnue_data.h\n'

for source in "${SOURCES[@]}"; do
    object="$OBJ_DIR/${source%.cpp}.o"
    printf '\n=== Compile %s ===\n' "$source"
    "$CXX" "${COMMON_FLAGS[@]}" -c "$source" -o "$object"
    OBJECTS+=("$object")
done

printf '\n=== Link Android ARM64 executable ===\n'
"$CXX" "${COMMON_FLAGS[@]}" -fuse-ld=lld -static-libstdc++ -Wl,--no-undefined \
    "${OBJECTS[@]}" -lm -o "$OUT"

printf '\n=== Verify ELF architecture ===\n'
"$READELF" -h "$OUT" > "$BUILD_DIR/elf-header.txt"
if ! grep -Eq 'Machine:[[:space:]]+AArch64' "$BUILD_DIR/elf-header.txt"; then
    cat "$BUILD_DIR/elf-header.txt"
    echo "ERROR: Output is not an AArch64 ELF executable." >&2
    exit 1
fi
echo "PASS: ELF machine is AArch64."

printf '\n=== Verify embedded Solo NNUE payload ===\n'
"$NM" -S "$OUT" > "$BUILD_DIR/symbols.txt"
python3 - "$BUILD_DIR/symbols.txt" <<'PY'
import sys
from pathlib import Path

symbol_file = Path(sys.argv[1])
required = 768 * 512 * 2 + 512 * 2 + 512 * 2 * 2 + 2
found = None
for line in symbol_file.read_text(errors="replace").splitlines():
    fields = line.split()
    if len(fields) >= 4 and "nnue_data" in fields[-1]:
        try:
            size = int(fields[1], 16)
        except ValueError:
            continue
        found = (fields[-1], size)
        break
if found is None:
    raise SystemExit("ERROR: Embedded nnue_data symbol was not found in the linked executable.")
name, size = found
if size < required:
    raise SystemExit(f"ERROR: Embedded NNUE payload is too short: {size} bytes; need at least {required}.")
print(f"PASS: {name} is embedded ({size} bytes; {required} bytes of model parameters required).")
PY

printf '\n=== Verify NEON 8x16-bit accumulator instructions ===\n'
"$OBJDUMP" -d "$OUT" > "$BUILD_DIR/final.disasm"
if ! grep -Eq '\badd[[:space:]]+v[0-9]+\.8h' "$BUILD_DIR/final.disasm"; then
    echo "ERROR: No 8x16-bit NEON ADD instruction found in the executable." >&2
    exit 1
fi
if ! grep -Eq '\bsub[[:space:]]+v[0-9]+\.8h' "$BUILD_DIR/final.disasm"; then
    echo "ERROR: No 8x16-bit NEON SUB instruction found in the executable." >&2
    exit 1
fi
echo "PASS: final executable contains both 8x16-bit NEON ADD and SUB instructions."

printf '\n=== Check runtime dependencies ===\n'
DYNAMIC="$BUILD_DIR/dynamic-section.txt"
"$READELF" -d "$OUT" > "$DYNAMIC" || true
if grep -q 'libc++_shared.so' "$DYNAMIC"; then
    echo "ERROR: Executable unexpectedly depends on libc++_shared.so." >&2
    exit 1
fi
echo "PASS: no libc++_shared.so dependency."

"$STRIP" --strip-unneeded "$OUT"
chmod 755 "$OUT"
printf '\n=== Build result ===\n'
ls -lh "$OUT"
sha256sum "$OUT" | tee "$DIST_DIR/Solo-Android-ARM64-NEON.sha256"
echo "SUCCESS: Solo Android ARM64 NEON executable is ready."

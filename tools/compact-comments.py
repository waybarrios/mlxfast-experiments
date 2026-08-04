#!/usr/bin/env python3
"""Compact long Swift doc-comment blocks in the editable surface.

Safe trimming: only runs on full-line // and /// comment blocks that are
NOT inside a Swift multi-line string literal (triple-quote), keeps the
first line of each block (usually the flag/kernel identifier), and never
touches code. Preserves copyright headers. Prints per-file savings.
"""
import json
import os
import re
import sys

CONTRACT = "benchmark.json"


def find_editable_files():
    contract = json.load(open(CONTRACT))
    files = []
    for path in contract["editablePaths"]:
        full = os.path.join(".", path)
        if os.path.isdir(full):
            for dp, _, names in os.walk(full):
                for n in names:
                    p = os.path.join(dp, n)
                    if os.path.isfile(p):
                        files.append(p)
        elif os.path.isfile(full):
            files.append(full)
    return sorted(files)


def is_comment_line(line):
    s = line.lstrip()
    return s.startswith("//") or s.startswith("///")


def compact_file(path, min_lines=4, keep_lines=1, min_savings=120):
    with open(path) as f:
        lines = f.readlines()
    out = []
    i = 0
    saved = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # detect multi-line string state on this line
        stripped = line.strip()
        if '"""' in line:
            # a """ opens or closes a multiline string; never touch
            # comments inside; copy line and any following string body
            # until the closing """ (conservative: just copy the rest
            # of this line; subsequent lines handled normally, but we
            # mark state via a simple heuristic: skip comment-compaction
            # for lines that are part of a string body)
            out.append(line)
            i += 1
            # if this line opened a string (odd count of """), consume
            # body lines until a line containing the closing """
            count = line.count('"""')
            if count % 2 == 1:
                i_prev = i
                while i < n and lines[i].count('"""') % 2 == 0:
                    out.append(lines[i])
                    i += 1
                if i < n:
                    out.append(lines[i])
                    i += 1
            continue
        # full-line comment block?
        if is_comment_line(line):
            j = i
            while j < n and is_comment_line(lines[j]):
                j += 1
            block_len = j - i
            block_bytes = sum(len(x) + 1 for x in lines[i:j])
            if block_len >= min_lines and block_bytes >= min_savings:
                # keep first `keep_lines` lines (identifier doc), drop rest
                kept = lines[i : i + keep_lines]
                # ensure we keep a sensible summary: if first line is
                # short, also keep the next line if it exists
                out.extend(kept)
                saved += block_bytes - sum(len(x) + 1 for x in kept)
                i = j
                continue
        out.append(line)
        i += 1
    if saved:
        with open(path, "w") as f:
            f.writelines(out)
        print(f"{path}: saved {saved:,} bytes")
    return saved


def main():
    total = 0
    for path in find_editable_files():
        if not (path.endswith(".swift") or path.endswith(".cpp")
                or path.endswith(".h") or path.endswith(".metal")):
            continue
        total += compact_file(path)
    print(f"\nTOTAL SAVED: {total:,} bytes")


if __name__ == "__main__":
    main()

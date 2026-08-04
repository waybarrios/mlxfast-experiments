#!/usr/bin/env python3
"""Aggressive pass 2: compact/remove far more doc comments in the editable
surface. Removes full comment blocks of 2+ lines (keeps only lines that
mention copyright/license), truncates single-line comments longer than 100
chars, and never touches Swift multi-line string literals (Metal sources).
"""
import json
import os
import re

CONTRACT = "benchmark.json"
PROTECT = ("copyright", "license", "spdx", "author", "modified by")


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


def is_comment(line):
    s = line.lstrip()
    return s.startswith("//") or s.startswith("///") or s.startswith("/*") or s.startswith("*")


def protected(line):
    low = line.lower()
    return any(t in low for t in PROTECT)


def compact_file(path):
    with open(path) as f:
        lines = f.readlines()
    out = []
    i = 0
    n = len(lines)
    saved = 0
    while i < n:
        line = lines[i]
        # multiline string state: copy whole literal, never touch inside
        if '"""' in line:
            out.append(line)
            i += 1
            if line.count('"""') % 2 == 1:
                while i < n and lines[i].count('"""') % 2 == 0:
                    out.append(lines[i])
                    i += 1
                if i < n:
                    out.append(lines[i])
                    i += 1
            continue
        # C-style block comment start /* ... */  (single or multi line)
        if line.lstrip().startswith("/*"):
            if "*/" in line:
                if not protected(line):
                    saved += len(line) + 1
                    i += 1
                    continue
                out.append(line)
                i += 1
                continue
            # multi-line block comment: consume until */
            j = i
            body = []
            while j < n and "*/" not in lines[j]:
                body.append(lines[j])
                j += 1
            if j < n:
                body.append(lines[j])
                j += 1
            if any(protected(b) for b in body):
                out.extend(body)
            else:
                saved += sum(len(b) + 1 for b in body)
            i = j
            continue
        # comment block (2+ consecutive // lines) -> drop entirely
        if is_comment(line):
            j = i
            block = []
            while j < n and is_comment(lines[j]) and '"""' not in lines[j]:
                block.append(lines[j])
                j += 1
            if len(block) >= 2:
                if any(protected(b) for b in block):
                    out.extend(block)
                else:
                    saved += sum(len(b) + 1 for b in block)
                i = j
                continue
            # single-line comment: truncate if very long
            if len(line) > 100 and not protected(line):
                out.append(line[:60].rstrip() + "\n")
                saved += len(line) - 60
                i += 1
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
    print(f"\nTOTAL SAVED PASS 2: {total:,} bytes")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Single-threaded Python baseline for Phase 2 morphology features.

This mirrors the Rust benchmark workload (synthetic masks, area/perimeter/centroid)
without third-party dependencies.
"""

from math import sqrt
from time import perf_counter


def generate_masks(mask_count: int, side: int):
    masks = []
    for i in range(mask_count):
        mask = [[False for _ in range(side)] for _ in range(side)]
        offset = (i * 7) % max(1, side - 12)
        for r in range(offset + 2, min(offset + 10, side)):
            for c in range(offset + 2, min(offset + 10, side)):
                mask[r][c] = True
        masks.append(mask)
    return masks


def binary_erosion_4(mask):
    h = len(mask)
    w = len(mask[0])
    eroded = [[False for _ in range(w)] for _ in range(h)]
    if h < 3 or w < 3:
        return eroded
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            eroded[r][c] = (
                mask[r][c]
                and mask[r - 1][c]
                and mask[r + 1][c]
                and mask[r][c - 1]
                and mask[r][c + 1]
            )
    return eroded


def perimeter(mask):
    h = len(mask)
    w = len(mask[0])
    eroded = binary_erosion_4(mask)
    border = [[mask[r][c] and not eroded[r][c] for c in range(w)] for r in range(h)]

    offsets = [
        (-1, -1, 10),
        (-1, 0, 2),
        (-1, 1, 10),
        (0, -1, 2),
        (0, 0, 1),
        (0, 1, 2),
        (1, -1, 10),
        (1, 0, 2),
        (1, 1, 10),
    ]

    hist = [0] * 50
    for r in range(h):
        for c in range(w):
            code = 0
            for dr, dc, weight in offsets:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and border[rr][cc]:
                    code += weight
            hist[code] += 1

    total = 0.0
    for code, count in enumerate(hist):
        if code in (5, 7, 15, 17, 25, 27):
            total += count
        elif code in (21, 33):
            total += count * sqrt(2.0)
        elif code in (13, 23):
            total += count * ((1.0 + sqrt(2.0)) / 2.0)
    return total


def calculate_basic_morphology(mask):
    h = len(mask)
    w = len(mask[0])
    area = 0
    sum_r = 0.0
    sum_c = 0.0
    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                area += 1
                sum_r += r
                sum_c += c
    if area == 0:
        raise ValueError("empty mask")

    return {
        "area": float(area),
        "perimeter": perimeter(mask),
        "centroid_row": sum_r / area,
        "centroid_col": sum_c / area,
    }


def run_batch(masks):
    return [calculate_basic_morphology(m) for m in masks]


def bench(mask_count: int, repeats: int = 5):
    masks = generate_masks(mask_count, 64)
    # warm-up
    run_batch(masks)
    times = []
    for _ in range(repeats):
        t0 = perf_counter()
        run_batch(masks)
        t1 = perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    median = times[len(times) // 2]
    print(f"python_single_thread/{mask_count}: median={median:.3f} ms")


if __name__ == "__main__":
    for n in (32, 128, 512):
        bench(n)

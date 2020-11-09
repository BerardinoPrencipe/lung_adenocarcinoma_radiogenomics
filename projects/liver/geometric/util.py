import numpy as np
from projects.liver.geometric.points_segments import extract_contours


def find_lines(volume):
    slices, h, w = volume.shape
    for slice in range(slices):
        a = 0
        s = volume[slice]
        un = np.unique(s)
        pairs = []
        if len(un) == 1 and bool(un == 0):
            continue
        else: #pairs creation
            if 4 in un:
                if 2 in un:
                    pairs.append((2, 4))
                elif 3 in un:
                    pairs.append((3, 4))
                if 8 in un:
                    pairs.append((4, 8))
                    if 7 in un:
                        pairs.append((7, 8))
                elif 5 in un:
                    pairs.append((4, 5))
                    if 6 in un:
                        pairs.append((5, 6))
            elif 8 in un:
                pairs.append((7, 8))
            elif 5 in un:
                pairs.append((5, 6))
        if not pairs:
            continue

        for pair in pairs:
            thickness = 2
            intersect = extract_contours(pair, s, thickness)
            data = np.where(intersect == 1)
            while len(data[0]) < 10 and thickness <= 12:
                thickness += 1
                intersect = extract_contours(pair, s, thickness)
                data = np.where(intersect == 1)
            if thickness > 12:
                continue
            a += intersect

    return a

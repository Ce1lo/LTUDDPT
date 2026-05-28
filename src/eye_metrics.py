# eye_metrics.py - Tinh cac chi so hinh hoc cua mat.
import math


def calculate_distance(p1, p2):
    """Tinh khoang cach Euclid giua hai diem toa do."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def calculate_ear(eye_points):
    """Tinh ty le EAR tu 6 diem moc cua mot mat."""
    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]

    # EAR = (hai khoang cach doc giua mi tren/duoi)
    # / (2 * khoang cach ngang giua hai khoe mat). Khi mat nham,
    # khoang cach doc giam manh nen EAR cung giam theo.
    vertical_1 = calculate_distance(p2, p6)
    vertical_2 = calculate_distance(p3, p5)
    horizontal = calculate_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)

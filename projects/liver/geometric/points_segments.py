import cv2
import time
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib import pyplot as plt


def extract_segments_plane(planes, volume):
    lpv = planes["left_pv"]
    rpv = planes["right_pv"]
    points_plane = plane_to_lines(planes, volume)

    rebuilt = np.zeros(volume.shape)

    for slice in range(volume.shape[0]):
        l = [(-511, 0, 0)]
        l.extend([p[slice] for p in points_plane.values()])
        l.append((0, 511, 0))
        coeffs = [(l[i], l[i + 1]) for i in range(len(l) - 1)]
        values = [2, 4, 8, 7]
        if slice < lpv:
            values[0] = 3
        if slice < rpv:
            values[2] = 5
            values[3] = 6
        # print("values: ", values)
        segments = []
        for coeff, value in zip(coeffs, values):
            r = point_to_segments({"down": coeff[1], "up": coeff[0]}, volume[slice][np.newaxis])
            segment = value * r
            segments.append(segment)
            volume[slice][np.newaxis] -= r

        rebuilt[slice] = np.sum(np.concatenate(segments), axis=0)
        # print("slice: ", slice)
        # print("unique: ", np.unique(rebuilt[slice]))
        # plt.imshow(rebuilt[slice])
        # plt.show()

    return rebuilt


def plane_to_lines(planes, volume):  #volume == parenchyma

    couples = np.array([[i, j] for i in range(volume.shape[0]) for j in range(volume.shape[2])])
    points_plane = {}
    for key, plane in zip(planes["planes"], planes["planes"].values()):
        plane_volume = np.zeros(volume.shape)
        reg = plane["plane"]
        yp = np.array([couples[:, 0] * reg[0] + couples[:, 1] * reg[1] + reg[2]])
        res = np.concatenate((couples, np.transpose(yp)), 1)
        res = res.astype(np.int32)
        res = np.array([[i, j, k] for i, j, k in res if 0 <= k < volume.shape[1]])
        plane_volume[res[:, 0], res[:, 2], res[:, 1]] = 1

        points = []
        for slice in range(plane_volume.shape[0]):
            ones = np.transpose(np.array(np.where(plane_volume[slice] == 1)))
            i = random.sample(range(ones.shape[0]), k=2)
            pts = (tuple(ones[i[0]]), tuple(ones[i[1]]))
            points.append(calculate_coefficent(points=pts))

        points_plane[key] = points

    return points_plane


def find_lines(volume):
    start = time.time()
    slices, h, w = volume.shape
    left_pv = np.max(np.where(volume == 3)[0]) + 1
    right_pv = np.max(np.where(volume == 5)[0]) + 1
    data2_4 = []
    data4_8 = []
    data7_8 = []
    datas = {}
    for slice in range(slices):
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
            z_axis = np.array(([slice]*len(data[0])), ndmin=2)
            data = np.transpose(np.concatenate((z_axis, data)))
            if pair == (2,4) or pair == (3,4):
                data2_4.append(data)
            if pair == (4,5) or pair == (4,8):
                data4_8.append(data)
            if pair == (5,6) or pair == (7,8):
                data7_8.append(data)
    datas["2,4"] = np.concatenate(data2_4,0)
    datas["4,5"] = np.concatenate(data4_8,0)
    datas["7,8"] = np.concatenate(data7_8,0)

    # Linear Regression
    results = {}
    res = {}
    for key, val in zip(datas, datas.values()):
        reg = LinearRegression()
        result = {}
        y = val[:, 1]
        X = val[:, [0, 2]]
        reg.fit(y=y, X=X)
        yp = reg.predict(X)
        r2 = r2_score(y, yp)
        mse = mean_squared_error(y,yp)
        result["plane"] = (reg.coef_[0], reg.coef_[1], reg.intercept_)
        result["r2"] = r2
        result["mse"] = mse
        res[key] = result


    results["planes"] = res
    results["left_pv"] = left_pv
    results["right_pv"] = right_pv

    print("time: ", time.time() - start)
    return results
    #return (X, y, yp, reg)


def extract_contours(pair, a, thickness):

    seg0 = pair[0] * (a == pair[0])
    seg0 = seg0.astype(np.uint8)
    seg1 = pair[1] * (a == pair[1])
    seg1 = seg1.astype(np.uint8)

    cont0, _ = cv2.findContours(seg0, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cont1, _ = cv2.findContours(seg1, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    c0 = seg0.copy()
    c1 = seg1.copy()
    cv2.drawContours(c0, cont0, -1, 1, thickness)
    c0 = np.subtract(seg0, c0)
    cv2.drawContours(c1, cont1, -1, 1, thickness)
    c1 = np.subtract(seg1, c1)

    intersect = 1 * np.logical_and(c0, c1)
    return intersect


def find_lines_one(volume):
    SUM = np.sum(volume > 0, axis=(1,2))
    slice = np.argmax(SUM)
    s = volume[slice]
    un = np.unique(s)
    left_pv = np.max(np.where(volume == 3)[0]) + 1
    right_pv = np.max(np.where(volume == 5)[0]) + 1
    results = {}
    results["left_pv"] = left_pv
    results["right_pv"] = right_pv
    reg = LinearRegression()

    if np.sum(np.isin(un, 2)) > 0:
        pairs = [(2, 4), (4, 8), (8, 7)]
    elif np.sum(np.isin(un, 3)) > 0 and np.sum(np.isin(un, 8)) > 0:
        pairs = [(3, 4), (4, 8), (8, 7)]
    else:
        pairs = [(3, 4), (4, 5), (5, 6)]
    ps = []
    for pair in pairs:

        thickness = 2
        intersect = extract_contours(pair, s, thickness)
        data = np.where(intersect == 1)

        while len(data[0]) < 20:
            thickness += 1
            intersect = extract_contours(pair, s, thickness)
            data = np.where(intersect == 1)

        y = data[0]
        X = np.expand_dims(data[1], 1)
        reg.fit(X, y)
        yp = reg.predict(X)

        ps.append({"line": (-reg.coef_[0], 1, -reg.intercept_), "r2": r2_score(y, yp),})

    results["lines"] = ps
    return results


def extract_segments(lines, volume, lpv, rpv):
    assert type(lines) == list, "lines must be a list"
    print("volume unique:", np.unique(volume))
    l = [(-511, 0, 0)]
    l.extend(lines)
    l.append((0, 511, 0))
    coeffs = [(l[i], l[i+1]) for i in range(len(l)-1)]
    values = [2,4,8,7]
    segments = []
    for coeff,value in zip(coeffs, values):
        result = point_to_segments({"down": coeff[1], "up": coeff[0]}, volume)
        segment = value*result
        if value == 2:
            temp = np.where(segment[:lpv] == 2, 3, 0)
            segment = np.concatenate((temp, segment[lpv:]))
        elif value == 8:
            temp = np.where(segment[:rpv] == 8, 5, 0)
            segment = np.concatenate((temp, segment[rpv:]))
        elif value == 7:
            temp = np.where(segment[:rpv] == 7, 6, 0)
            segment = np.concatenate((temp, segment[rpv:]))
        segments.append(segment)
        volume -=result
    return sum(segments)


def point_to_segments(points, image):
    """
    convert a couple of straight line to a semi plane
    :param points: dict with a "up" element and a "down" element. Up element selects the upper right of the image,
                    down element selects the downard right of the image
                    point["up"]   = ((y1,x1),(y2,x2))
                    point["down"] = ((y1,x1),(y2,x2))
                    or
                    point["up"]   = (a1, b1, c1)
                    point["down"] = (a2, b2, c2)

    :param image: ct volume of liver paremchyma
    :param value: value to assign to the segment
    :return: the segments of the volume
    """

    assert len(points["up"]) == 2 or len(points["up"]) == 3, "Points is uncorrect. Check that its a couple of points " \
                                                             "((y1,x1),(y2,x2)) or two tuples of coefficents (a,b,c)"

    coeff1, coeff2 = None, None
    if len(points["up"]) == 2:
        coeff1 = calculate_coefficent(points["up"])
        coeff2 = calculate_coefficent(points["down"])
    elif len(points["up"]) == 3:
        coeff1 = points["up"]
        coeff2 = points["down"]
    start = time.time()
    up_img = draw_semi_plane(line=coeff1, shape=image.shape, mode="up", value=1)
    down_img = draw_semi_plane(line=coeff2, shape=image.shape, mode="down", value=1)
    print("Exection time:", time.time()-start)
    return np.multiply(image,np.logical_and(up_img, down_img))


def draw_semi_plane(line, shape, mode, value):
    a, b, c = line
    p = []
    img = np.zeros((shape[1],shape[2]), dtype=np.int32)
    if mode == "up":
        if b != 0:
            m = -a / b
            q = -c / b
            print("m , q:", m, q)
            if m > 0:
                if 0 < q < shape[1]:
                    p.extend([(0, 0), (0, round(q))])
                elif q >= shape[1]:
                    p = [(0, 0), (0, shape[1]-1), (shape[2]-1, shape[1]-1), (shape[2]-1, 0)]
                    cv2.fillConvexPoly(img=img,points=np.array(p, dtype=np.int32), color=value)
                    return img
                else:
                    p.append((round(-q / m), 0))
                if m * (shape[2] - 1) + q > shape[1] - 1:
                    p.extend([(round(((shape[2] - 1) - q) / m), shape[1] - 1), (shape[2] - 1, shape[1] - 1)])
                else:
                    p.append((shape[1] - 1, round(m * (shape[2] - 1) + q)))
                p.append((shape[2] - 1, 0))
            elif m == 0:
                if 0 <= q < shape[1]:
                    p.extend([(0, 0), (0, round(q)), (shape[2] - 1, round(q)), (shape[2] - 1, 0)])
                elif q >= shape[1]:
                    p.extend([(0, 0), (0, shape[1]-1), (shape[2] - 1, shape[1]-1), (shape[2] - 1, 0)])
            else:

                if m * shape[2] - 1 + q == shape[1] - 1:
                    p = [(0, 0), (0, shape[1] - 1), (shape[2] - 1, shape[1] - 1), (shape[2] - 1, 0)]
                    cv2.fillConvexPoly(img=img, points=np.array(p, dtype=np.int32), color=value)
                    return img
                if 0 < q < shape[1]:
                    p.extend([(0, 0), (0, q)])
                elif q >= shape[1]:
                    p.extend([(0, 0), (0, shape[1] - 1), (round(((shape[1] - 1) - q) / m), shape[1] - 1)])

                if m * (shape[2] - 1) + q <= 0:
                    p.append((-round(q / m), 0))
                else:
                    p.extend([(shape[2] - 1, round(shape[2] - 1) * m + q), (shape[2] - 1, 0)])
        else: # b == 0
            q = -c / a
            print("q:", q)
            if 0 <= q < shape[2]:
                p.extend([(round(q), 0), (round(q), shape[1]-1), (shape[2]-1, shape[1]-1), (shape[1]-1, 0)])
            elif q < 0:
                p = [(0, 0), (0, shape[1] - 1), (shape[2] - 1, shape[1] - 1), (shape[2] - 1, 0)]
                cv2.fillConvexPoly(img=img, points=np.array(p, dtype=np.int32), color=value)
                return img
    if mode == "down":
        if b != 0:
            m = -a / b
            q = -c / b
            print("m , q:", m, q)
            if m > 0:
                if m*(shape[2]-1) + q == 0:
                    p = [(0, 0), (0, shape[1] - 1), (shape[2] - 1, shape[1] - 1), (shape[2] - 1, 0)]
                    cv2.fillConvexPoly(img=img, points=np.array(p, dtype=np.int32), color=value)
                    return img
                if q < 0:
                    p.extend([(-round(q / m), 0), (0, 0)])
                else:
                    p.append((0, round(q)))
                p.append((0, shape[1] - 1))
                if m * (shape[2] - 1) + q < shape[1]:
                    p.extend([(shape[2] - 1, shape[1] - 1), (shape[2] - 1, round((shape[2] - 1) * m + q))])
                else:
                    p.append(((round((shape[1]-1-q)/m), shape[1]-1)))
            elif m == 0:
                if 0 <= q < shape[1]:
                    p.extend([(0, round(q)), (0, shape[1]-1), (shape[2] - 1, shape[1] - 1), (shape[2] - 1, round(q))])
            else:
                if 0 <= q < shape[1]:
                    p.extend([(0, round(q)), (0, shape[1] - 1)])
                elif q < 0:
                    p = [(0, 0), (0, shape[1] - 1), (shape[2] - 1, shape[1] - 1), (shape[2] - 1, 0)]
                    cv2.fillConvexPoly(img=img, points=np.array(p, dtype=np.int32), color=value)
                    return img
                else:
                    p.append((round(((shape[1]-1)-q)/ m), shape[1]-1))

                p.append((shape[2] - 1, shape[1] - 1))

                if m * (shape[2] - 1) + q < 0:
                    p.extend([(shape[2] - 1, 0), (-round(q / m), 0)])
                elif m * (shape[2] - 1) + q == 0:
                    p.append((shape[2] - 1, 0))
                else:
                    p.append((shape[2] - 1, round((shape[2] - 1) * m + q)))

        else: # b == 0
            q = -c / a
            print("q:", q)
            if 0 <= q < shape[2]:
                p.extend([(0, 0), (round(q), 0), (round(q), shape[1] - 1), (0, shape[1] - 1)])
            elif q >= shape[2]:
                p = [(0, 0), (0, shape[1] - 1), (shape[2] - 1, shape[1] - 1), (shape[2] - 1, 0)]
                cv2.fillConvexPoly(img=img, points=np.array(p, dtype=np.int32), color=value)
                return img
    cv2.fillConvexPoly(img, points=np.array(p, dtype=np.int32), color=value)
    return np.repeat(img[np.newaxis,:,:], repeats=shape[0], axis=0)


def calculate_coefficent(points):
    """
    Calculate line coefficents

    a = y2-y1
    b = x2-x1
    d = y1x2 - x1y2
    :param points: points where the line passes
    :return: coefficent of a line in the form ax+by+c = 0
    """

    a = -(points[1][0] - points[0][0])
    b = -(points[0][1] - points[1][1])
    d = points[0][1] * points[1][0] - points[0][0] * points[1][1]
    return a, b, d


def parse_json(file):
    pass


def evaluate_segments(gt, pred):
    pred[gt == 1] = 0
    values = [2, 3, 4, 5, 6, 7, 8]
    DSC = {}
    for value in values:
        l = gt.copy()
        lr = pred.copy()
        l[gt != value] = 0
        lr[pred != value] = 0
        a = (np.sum(np.logical_and(l, lr)))
        b = (np.sum(l > 0) + np.sum(lr > 0))
        DSC[value] = 2 * a / b
    return DSC

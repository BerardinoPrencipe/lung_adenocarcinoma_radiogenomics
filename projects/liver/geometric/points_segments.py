import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

#
# def points_to_segments_opt(points, shape):
#     temp = np.zeros(shape)
#     a, b, d = calculate_coefficent(points)
#     xx, yy = np.meshgrid(range(shape[0]), range(shape[1]))
#     for x,y in zip(xx,yy):
#         xy1 = np.stack((y,x,np.ones(shape[0],)))
#         xy1 = xy1.transpose()
#         # 512 x 3
#         abc = np.array([b, a, d])
#         # 3 x 1
#         temp = np.where(np.matmul(xy1,abc) > 0)[0]
#
# def points_to_segments(points, shape):
#     start = time.time()
#     temp = np.zeros(shape)
#     a, b, d = calculate_coefficent(points)
#     print("a,b,d: ",a,b,d)
#     for idx_y in range(shape[0]):
#         for idx_x in reversed(range(shape[1])):
#             temp[idx_y, idx_x] = 1 if b*idx_y+a*idx_x+d <= 0 else 0
#     print("Execution time: ", time.time()-start)
#     return temp
#
#
# def points_to_segments2(points, mask):
#     start = time.time()
#     a, b, d = calculate_coefficent(points)
#     print("a,b,d: ", a, b, d)
#     X = range(mask.shape[0])
#     Y = range(mask.shape[1])
#     result = []
#     for x in X:
#         for y in Y:
#             result.append(int(a*x+b*y+d > 0))
#     temp = np.resize(np.array(result),new_shape=mask.shape)
#     print("Execution time: ", time.time()-start)
#     return temp
#
#
# def points_to_segments3(points,mask):
#     start = time.time()
#     x = np.linspace(0, mask.shape[0]-1,mask.shape[0])
#     y = np.linspace(0, mask.shape[1]-1,mask.shape[1])
#     temp = np.zeros(mask.shape)
#     prod = np.array(np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))]),dtype=np.int)
#     a, b, d = calculate_coefficent(points)
#     for p in prod:
#         temp[p[0],p[1]] = 1 if p[0]*b+p[1]*a+d == 0 else 0
#     print("Execution time: ", time.time() - start)
#     return temp
#
#
# def calculate_coefficent(points):
#     """
#     Calculate line coefficents
#
#     a = y2-y1
#     b = x2-x1
#     d = y1x2 - x1y2
#     :param points: points where the line passes
#     :return: coefficent of a line in the form ax+by+d
#     """
#
#     a = -(points[1][0] - points[0][0])
#     b = -(points[0][1] - points[1][1])
#     d = points[0][1] * points[1][0] - points[0][0] * points[1][1]
#     return a/b, 1, d/b
#
#
# a = points_to_segments(((511,0),(9,9)),np.zeros((512,512)))
# b = points_to_segments(((0,0),(9,9)),np.zeros((512,512)))
# c = points_to_segments(((15,30),(4,20)),np.zeros((512,512)))
# d = points_to_segments(((15,30),(511,20)),np.zeros((512,512)))
# plt.imshow(a); plt.show()
# plt.imshow(b); plt.show()
# plt.imshow(c); plt.show()
# plt.imshow(d); plt.show()
#
# points = ((0,0),(9,9))


def points_to_segment(points, image):

    a, b, d = calculate_coefficent(points)
    print("a,b,d: ",a,b,d)
    p = []
    if abs(a) > 1:
        idx_x_s = round(-(d/a))
        idx_x_e = round(((image.shape[0]-1)-d)/-a)
        p.extend([(idx_x_s,0),(idx_x_e,511),(511,511),(511,0)])
    elif abs(a) < 1:
        pass
    else:
        p.extend([(0,0),(image.shape[1]-1,image.shape[0]-1),(511,0)])

    print(p)
    I = cv2.fillConvexPoly(image, np.array(p), color=(255,0,0))
    return I


def points_to_segment2(points,image):

    image = image.copy()
    a, b, d = calculate_coefficent(points)
    m = -a
    q = -d
    print("a,b,d: ", a, b, d)
    p = []
    # idx_x_s = max(0, round(-(d/a)))
    # idx_x_e = min(image.shape[0]-1, round(((image.shape[0]-1)-d)/-a))
    # print("q/m: ", round(-(d/a)))
    # print("idx_x_s: ",idx_x_s)
    # print(round(((image.shape[0]-1)-d)/-a))
    # print("idx_x_e: ",idx_x_e)
    # if idx_x_s > 0:
    #     p.extend([(0,0),(idx_x_s,0)])
    # else:
    #     p.append((idx_x_s,0))
    # if idx_x_e != image.shape[0]-1:
    #     p.extend([(idx_x_e,image.shape[0]-1),(image.shape[1]-1, image.shape[0]-1)])
    # else:
    #     p.append((idx_x_e,image.shape[0]-1))
    #
    # p.append((image.shape[1]-1,0))
    # print(p)

    if q > 0:
        p.extend([(0,0),(0,round(q))])
    else:
        p.append((round(-q/m),0))
    if m*(image.shape[1]-1)+q > image.shape[0]-1:
        p.extend([(round(((image.shape[1]-1)-q)/m), image.shape[0]-1), (image.shape[1]-1, image.shape[0]-1)])
    else:
        p.append((image.shape[0]-1,round(m*(image.shape[1]-1)+q)))

    p.append((image.shape[1] - 1, 0))
    print(p)
    I = cv2.fillConvexPoly(img=image, pts=np.array(p), color=(255, 0, 0))
    plt.imshow(I)
    plt.show()
    return I


def points_to_segments3(points, image):

    a1, _ ,d1 = calculate_coefficent(points[0])
    m1 = -a1
    q1 = -d1
    a2, _, d2 = calculate_coefficent(points[1])
    m2 = -a2
    q2 = -d2

    if m1 > 0 and m2>0:
        pass
    elif m1 > 0 and m2 < 0:
        pass
    elif m1 <0 and m2 > 0:
        pass
    elif m1 < 0 and m2 < 0:
        pass

    # print("m1, q1 :", m1, q1)
    # print("m2, q2 :", m2, q2)
    # h, w = image.shape
    # x = range(w)
    # y1 = np.array(np.round(np.array(x) * m1 + q1)).astype(np.int32)
    # y2 = np.array(np.round(np.array(x) * m2 + q2)).astype(np.int32)
    # idx_cross = round(-((q1-q2)/(m1-m2)))
    # p = []
    # if m1 > 0:
    #     if q1 <= 0:
    #         p.append((0, q1))
    #     else:
    #         p.extend([(0, 0), (0, q1)])
    # elif m1 < 0:
    #     if 0 < q1 < h-1:
    #         p.append((0, q1))
    #     else:
    #         p.extend([(0, 0), (0, q1)])
    # if 0 <= idx_cross < w:
    #     p.append((idx_cross, y1[idx_cross]))
    # else:
    #     p.extend([(round(((h-1) - q1) /m1), h), (round(((h-1) - q2)/m2), h)])
    # if m2*(w-1)+q2 <= 0:
    #     p.append((w-1, m2*(w-1)+q2))
    # else:
    #     p.extend([(w-1, round(m2*(w-1)+q2)), (w-1, 0)])
    # p1 = np.array([[[xi, yi]] for xi,yi in zip(x,y1) if 0 <= xi < w and 0 <= yi <h ]).astype(np.int32)
    # p2 = np.array([[[xi, yi]] for xi,yi in zip(x,y2) if 0 <= xi < w and 0 <= yi < h]).astype(np.int32)
    # # p2 = np.flipud(p2)
    # p = np.concatenate((p1,p2), axis=0)
    # # p = p1
    a = image.copy()
    print(p)
    cv2.fillConvexPoly(a, np.array(p, dtype=np.int32), color=(255,0,0))
    plt.imshow(a)
    plt.show()
    return a,y1,y2,p


def draw_line(points, image):
    a = image.copy()
    cv2.line(a, points[0], points[1], color=(255,0,0))
    plt.imshow(a)
    plt.show()
    return a

def calculate_coefficent(points):
    """
    Calculate line coefficents

    a = y2-y1
    b = x2-x1
    d = y1x2 - x1y2
    :param points: points where the line passes
    :return: coefficent of a line in the form y-mx-q = 0
    """

    a = -(points[1][0] - points[0][0])
    b = -(points[0][1] - points[1][1])
    d = points[0][1] * points[1][0] - points[0][0] * points[1][1]
    return a/b, 1, d/b

# I = np.zeros((512,512),dtype=np.uint8)
#
#
# a = points_to_segment2(((0,0),(120,60)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((0,50),(120,60)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((0,50),(60,120)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((30,0),(60,120)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((30,0),(120,60)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((0,0),(60,60)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((0,511),(60,60)),np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segment2(((0,0),(60,60)),I)
#
# a = points_to_segment2(((0,511),(256,256)),I)

# a = points_to_segments3([[(0,0),(511,170)],[(0,511),(511,400)]],np.zeros((512,512),dtype=np.uint8))
#
# a = points_to_segments3([[(0,511),(511,400)],[(0,0),(511,170)]],np.zeros((512,512),dtype=np.uint8))

# a = points_to_segments3([[(0,100),(511,0)],[(0,511),(511,400)]],np.zeros((512,512),dtype=np.uint8))

g = draw_line([(100,0),(0,511)],np.zeros((512,512),dtype=np.uint8))

# plt.imshow(a); plt.show()


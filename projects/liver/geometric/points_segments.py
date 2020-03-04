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


def calculate_coefficent(points):
    """
    Calculate line coefficents

    a = y2-y1
    b = x2-x1
    d = y1x2 - x1y2
    :param points: points where the line passes
    :return: coefficent of a line in the form ax+by+d
    """

    a = -(points[1][0] - points[0][0])
    b = -(points[0][1] - points[1][1])
    d = points[0][1] * points[1][0] - points[0][0] * points[1][1]
    return a/b, 1, d/b

I = np.zeros((512,512),dtype=np.uint8)
# start = time.time()
# cv2.fillConvexPoly( I , np.array([(0,0),(511,511),(0,511)]) , color=(255,0,0))
# print(time.time()-start)
# plt.imshow(I); plt.show()

a = points_to_segment(((0,0),(9,9)),I)
plt.imshow(a); plt.show()
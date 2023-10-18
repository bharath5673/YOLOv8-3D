import os
import cv2
import numpy as np
from enum import Enum
import itertools

from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

"""
Script for handling calibration file
"""

import numpy as np

def get_P(calib_file):
    """
    Get matrix P_rect_02 (camera 2 RGB)
    and transform to 3 x 4 matrix
    """
    for line in open(calib_file):
        if 'P_rect_02' in line:
            cam_P = line.strip().split(' ')
            cam_P = np.asarray([float(cam_P) for cam_P in cam_P[1:]])
            matrix = np.zeros((3, 4))
            matrix = cam_P.reshape((3, 4))
            return matrix

# TODO: understand this

def get_calibration_cam_to_image(cab_f):
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

    file_not_found(cab_f)

def get_R0(cab_f):
    for line in open(cab_f):
        if 'R0_rect:' in line:
            R0 = line.strip().split(' ')
            R0 = np.asarray([float(number) for number in R0[1:]])
            R0 = np.reshape(R0, (3, 3))

            R0_rect = np.zeros([4,4])
            R0_rect[3,3] = 1
            R0_rect[:3,:3] = R0

            return R0_rect

def get_tr_to_velo(cab_f):
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            Tr = line.strip().split(' ')
            Tr = np.asarray([float(number) for number in Tr[1:]])
            Tr = np.reshape(Tr, (3, 4))

            Tr_to_velo = np.zeros([4,4])
            Tr_to_velo[3,3] = 1
            Tr_to_velo[:3,:4] = Tr

            return Tr_to_velo

def file_not_found(filename):
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    exit()


import numpy as np

# using this math: https://en.wikipedia.org/wiki/Rotation_matrix
def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])


    return Ry.reshape([3,3])
    # return np.dot(np.dot(Rz,Ry), Rx)

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in
    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners

# this is based on the paper. Math!
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray):
    #global orientation
    orient = alpha + theta_ray
    R = rotation_matrix(orient)

    # format 2d corners
    xmin = box_2d[0]
    ymin = box_2d[1]
    xmax = box_2d[2]
    ymax = box_2d[3]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    # below is very much based on trial and error

    # based on the relative angle, a different configuration occurs
    # negative is back of car, positive is front
    left_mult = 1
    right_mult = -1

    # about straight on but opposite way
    if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
        left_mult = 1
        right_mult = 1
    # about straight on and same way
    elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
        left_mult = -1
        right_mult = -1
    # this works but doesnt make much sense
    elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
        left_mult = -1
        right_mult = 1

    # if the car is facing the oppositeway, switch left and right
    switch_mult = -1
    if alpha > 0:
        switch_mult = 1

    # left and right could either be the front of the car ot the back of the car
    # careful to use left and right based on image, no of actual car's left and right
    for i in (-1,1):
        left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
    for i in (-1,1):
        right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

    # top and bottom are easy, just the top and bottom of car
    for i in (-1,1):
        for j in (-1,1):
            top_constraints.append([i*dx, -dy, j*dz])
    for i in (-1,1):
        for j in (-1,1):
            bottom_constraints.append([i*dx, dy, j*dz])

    # now, 64 combinations
    for left in left_constraints:
        for top in top_constraints:
            for right in right_constraints:
                for bottom in bottom_constraints:
                    constraints.append([left, top, right, bottom])

    # filter out the ones with repeats
    constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

    # create pre M (the term with I and the R*X)
    pre_M = np.zeros([4,4])
    # 1's down diagonal
    for i in range(0,4):
        pre_M[i][i] = 1

    best_loc = None
    best_error = [1e09]
    best_X = None

    # loop through each possible constraint, hold on to the best guess
    # constraint will be 64 sets of 4 corners
    count = 0
    for constraint in constraints:
        # each corner
        Xa = constraint[0]
        Xb = constraint[1]
        Xc = constraint[2]
        Xd = constraint[3]

        X_array = [Xa, Xb, Xc, Xd]

        # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
        Ma = np.copy(pre_M)
        Mb = np.copy(pre_M)
        Mc = np.copy(pre_M)
        Md = np.copy(pre_M)

        M_array = [Ma, Mb, Mc, Md]

        # create A, b
        A = np.zeros([4,3])
        b = np.zeros([4,1])

        indicies = [0,1,0,1]
        for row, index in enumerate(indicies):
            X = X_array[row]
            M = M_array[row]

            # create M for corner Xx
            RX = np.dot(R, X)
            M[:3,3] = RX.reshape(3)

            M = np.dot(proj_matrix, M)

            A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
            b[row] = box_corners[row] * M[2,3] - M[index,3]

        # solve here with least squares, since over fit will get some error
        loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # found a better estimation
        if error < best_error:
            count += 1 # for debugging
            best_loc = loc
            best_error = error
            best_X = X_array

    # return best_loc, [left_constraints, right_constraints] # for debugging
    best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
    return best_loc, best_X




##########################




def get_new_alpha(alpha):
    """
    change the range of orientation from [-pi, pi] to [0, 2pi]
    :param alpha: original orientation in KITTI
    :return: new alpha
    """
    new_alpha = float(alpha) + np.pi / 2.
    if new_alpha < 0:
        new_alpha = new_alpha + 2. * np.pi
        # make sure angle lies in [0, 2pi]
    new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)

    return new_alpha

def recover_angle(bin_anchor, bin_confidence, bin_num):
    # select anchor from bins
    max_anc = np.argmax(bin_confidence)
    anchors = bin_anchor[max_anc]
    # compute the angle offset
    if anchors[1] > 0:
        angle_offset = np.arccos(anchors[0])
    else:
        angle_offset = -np.arccos(anchors[0])

    # add the angle offset to the center ray of each bin to obtain the local orientation
    wedge = 2 * np.pi / bin_num
    angle = angle_offset + max_anc * wedge

    # angle - 2pi, if exceed 2pi
    angle_l = angle % (2 * np.pi)

    # change to ray back to [-pi, pi]
    angle = angle_l - np.pi / 2
    if angle > np.pi:
        angle -= 2 * np.pi
    angle = round(angle, 2)
    return angle


def compute_orientaion(P2, obj):
    x = (obj.xmax + obj.xmin) / 2
    # compute camera orientation
    u_distance = x - P2[0, 2]
    focal_length = P2[0, 0]
    rot_ray = np.arctan(u_distance / focal_length)
    # global = alpha + ray
    rot_global = obj.alpha + rot_ray

    # local orientation, [0, 2 * pi]
    # rot_local = obj.alpha + np.pi / 2
    rot_local = get_new_alpha(obj.alpha)

    rot_global = round(rot_global, 2)
    return rot_global, rot_local


def translation_constraints(P2, obj, rot_local):
    bbox = [obj.xmin, obj.ymin, obj.xmax, obj.ymax]
    # rotation matrix
    R = np.array([[ np.cos(obj.rot_global), 0,  np.sin(obj.rot_global)],
                  [          0,             1,             0          ],
                  [-np.sin(obj.rot_global), 0,  np.cos(obj.rot_global)]])
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    I = np.identity(3)

    xmin_candi, xmax_candi, ymin_candi, ymax_candi = obj.box3d_candidate(rot_local, soft_range=8)

    X  = np.bmat([xmin_candi, xmax_candi,
                  ymin_candi, ymax_candi])
    # X: [x, y, z] in object coordinate
    X = X.reshape(4,3).T

    # construct equation (4, 3)
    for i in range(4):
        matrice = np.bmat([[I, np.matmul(R, X[:,i])], [np.zeros((1,3)), np.ones((1,1))]])
        M = np.matmul(P2, matrice)

        if i % 2 == 0:
            A[i, :] = M[0, 0:3] - bbox[i] * M[2, 0:3]
            b[i, :] = M[2, 3] * bbox[i] - M[0, 3]

        else:
            A[i, :] = M[1, 0:3] - bbox[i] * M[2, 0:3]
            b[i, :] = M[2, 3] * bbox[i] - M[1, 3]
    # solve x, y, z, using method of least square
    Tran = np.matmul(np.linalg.pinv(A), b)

    tx, ty, tz = [float(np.around(tran, 2)) for tran in Tran]
    return tx, ty, tz








line_thickness = 3


class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img, calib_file=None):
    if calib_file is not None:
        print(calib_file)
        cam_to_img = get_calibration_cam_to_image(calib_file)
        R0_rect = get_R0(calib_file)
        Tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point



# take in 3d points and plot them on image as red circles
def plot_3d_pts(img, pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None):
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for pt in pts:
        if relative:
            pt = [i + center[j] for j,i in enumerate(pt)] # more pythonic

        point = project_3d_pt(pt, cam_to_img)

        color = cv_colors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 5, color, thickness=-1)





def plot_3d_box(img, cam_to_img, ry, dimension, center):

    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    color = (20, 255, 20)
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    #LINE
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), color, line_thickness)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), color, line_thickness)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), color, line_thickness)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), color, line_thickness)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), color, line_thickness)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), color, line_thickness)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), color, line_thickness)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), color, line_thickness)


    cv2.line(img, tuple(box_3d[0]), tuple(box_3d[1]), color, line_thickness)
    cv2.line(img, tuple(box_3d[2]), tuple(box_3d[3]), color, line_thickness)
    cv2.line(img, tuple(box_3d[4]), tuple(box_3d[5]), color, line_thickness)
    cv2.line(img, tuple(box_3d[6]), tuple(box_3d[7]), color, line_thickness)

    cv2.line(img, tuple(box_3d[0]), tuple(box_3d[4]), color, line_thickness)
    cv2.line(img, tuple(box_3d[1]), tuple(box_3d[5]), color, line_thickness)
    cv2.line(img, tuple(box_3d[2]), tuple(box_3d[6]), color, line_thickness)
    cv2.line(img, tuple(box_3d[3]), tuple(box_3d[7]), color, line_thickness)



    # cv2.line(img, tuple(box_3d[1]), tuple(box_3d[2]), (0, 255, 0), line_thickness)
    # cv2.line(img, tuple(box_3d[3]), tuple(box_3d[0]), (0, 255, 0), line_thickness)
    # cv2.line(img, tuple(box_3d[5]), tuple(box_3d[6]), (0, 255, 0), line_thickness)
    # cv2.line(img, tuple(box_3d[7]), tuple(box_3d[4]), (0, 255, 0), line_thickness)

    # cv2.line(img, tuple(box_3d[0]), tuple(box_3d[6]), (0, 255, 0), line_thickness)
    # cv2.line(img, tuple(box_3d[1]), tuple(box_3d[7]), (0, 255, 0), line_thickness)
    # cv2.line(img, tuple(box_3d[2]), tuple(box_3d[4]), (0, 255, 0), line_thickness)
    # cv2.line(img, tuple(box_3d[3]), tuple(box_3d[5]), (0, 255, 0), line_thickness)



    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.GREEN.value, 1)

    # frame to drawing polygon
    frame = np.zeros_like(img, np.uint8)

    # front side
    cv2.fillPoly(frame, np.array([[[box_3d[0]], [box_3d[1]], [box_3d[3]], [box_3d[2]]]], dtype=np.int32), cv_colors.BLUE.value)
    # center_x = (box_3d[0][0] + box_3d[1][0] + box_3d[2][0] + box_3d[3][0]) / 4
    # center_y = (box_3d[0][1] + box_3d[1][1] + box_3d[2][1] + box_3d[3][1]) / 4
    # cv2.circle(frame, (int(center_x), int(center_y)), 8, (0, 255, 255), -1)


    alpha = 0.6
    mask = frame.astype(bool)
    img[mask] = cv2.addWeighted(img, alpha, frame, 1 - alpha, 0)[mask]






def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, line_thickness)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, line_thickness)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, line_thickness)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, line_thickness)




class Plot3DBoxBev:
    """Plot 3D bounding box and bird eye view"""
    def __init__(
        self,
        proj_matrix = None, # projection matrix P2
        object_list = ["car", "pedestrian", "truck", "cyclist", "motorcycle", "bus"],
        
    ) -> None:

        self.proj_matrix = proj_matrix
        self.object_list = object_list

        self.fig = plt.figure(figsize=(20.00, 5.12), dpi=100)
        gs = GridSpec(1, 4)
        gs.update(wspace=0)
        self.ax = self.fig.add_subplot(gs[0, :3])
        self.ax2 = self.fig.add_subplot(gs[0, 3:])   


        self.fig2 = plt.figure(figsize=(8.00, 8.0), dpi=100)
        self.ax3 = self.fig2.add_subplot(111)
        # Set background color to black
        self.ax3.set_facecolor('black')

        self.shape = 900
        self.scale = 12

        self.COLOR = {
            "car": "blue",
            "pedestrian": "green",
            "truck": "yellow",
            "cyclist": "red",
            "motorcycle": "cyan",
            "bus": "magenta",
        }

    def compute_bev(self, dim, loc, rot_y):
        """compute bev"""
        # convert dimension, location and rotation
        h = dim[0] * self.scale
        w = dim[1] * self.scale
        l = dim[2] * self.scale
        x = loc[0] * self.scale
        y = loc[1] * self.scale
        z = loc[2] * self.scale
        rot_y = np.float64(rot_y)

        R = np.array([[-np.cos(rot_y), np.sin(rot_y)], [np.sin(rot_y), np.cos(rot_y)]])
        t = np.array([x, z]).reshape(1, 2).T
        x_corners = [0, l, l, 0]  # -l/2
        z_corners = [w, w, 0, 0]  # -w/2
        x_corners += -w / 2
        z_corners += -l / 2
        # bounding box in object coordinate
        corners_2D = np.array([x_corners, z_corners])
        # rotate
        corners_2D = R.dot(corners_2D)
        # translation
        corners_2D = t - corners_2D
        # in camera coordinate
        corners_2D[0] += int(self.shape / 2)
        corners_2D = (corners_2D).astype(np.int16)
        corners_2D = corners_2D.T

        return np.vstack((corners_2D, corners_2D[0, :]))

    def draw_bev(self, dim, loc, rot_y, class_object):
        color = self.COLOR[class_object]
        """draw bev"""

        # gt_corners_2d = self.compute_bev(self.gt_dim, self.gt_loc, self.gt_rot_y)
        pred_corners_2d = self.compute_bev(dim, loc, rot_y)

        codes = [Path.LINETO] * pred_corners_2d.shape[0]
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        pth = Path(pred_corners_2d, codes)
        patch = patches.PathPatch(pth, fill=False, color=color, label="prediction")
        patch2 = patches.PathPatch(pth, fill=False, color=color, label="prediction")
        self.ax2.add_patch(patch)

        # draw z location of object
        self.ax2.text(
            pred_corners_2d[0, 0],
            pred_corners_2d[0, 1],
            f"z: {loc[2]:.2f}",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="green", alpha=0.4, pad=0.5),
        )

        self.ax3.add_patch(patch2)

        # draw z location of object
        self.ax3.text(
            pred_corners_2d[0, 0],
            pred_corners_2d[0, 1],
            f"z: {loc[2]:.2f}",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="green", alpha=0.4, pad=0.5),
        )


    def compute_3dbox(self, bbox, dim, loc, rot_y):
        """compute 3d box"""
        # 2d bounding box
        xmin, ymin = int(bbox[0]), int(bbox[1])
        xmax, ymax = int(bbox[2]), int(bbox[3])

        # convert dimension, location
        h, w, l = dim[0], dim[1], dim[2]
        x, y, z = loc[0], loc[1], loc[2]

        R = np.array([[np.cos(rot_y), 0, np.sin(rot_y)], [0, 1, 0], [-np.sin(rot_y), 0, np.cos(rot_y)]])
        x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
        y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
        z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

        x_corners += -l / 2
        y_corners += -h
        z_corners += -w / 2

        corners_3D = np.array([x_corners, y_corners, z_corners])
        corners_3D = R.dot(corners_3D)
        corners_3D += np.array([x, y, z]).reshape(3, 1)

        corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
        corners_2D = self.proj_matrix.dot(corners_3D_1)
        corners_2D = corners_2D / corners_2D[2]
        corners_2D = corners_2D[:2]

        return corners_2D

    def draw_3dbox(self, class_object, bbox, dim, loc, rot_y):
        """draw 3d box"""
        color = self.COLOR[class_object]
        corners_2D = self.compute_3dbox(bbox, dim, loc, rot_y)

        # draw all lines through path
        # https://matplotlib.org/users/path_tutorial.html
        bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]
        bb3d_on_2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]
        verts = bb3d_on_2d_lines_verts.T
        codes = [Path.LINETO] * verts.shape[0]
        codes[0] = Path.MOVETO
        pth = Path(verts, codes)
        patch = patches.PathPatch(pth, fill=False, color=color, linewidth=2)

        width = corners_2D[:, 3][0] - corners_2D[:, 1][0]
        height = corners_2D[:, 2][1] - corners_2D[:, 1][1]
        # put a mask on the front
        front_fill = patches.Rectangle((corners_2D[:, 1]), width, height, fill=True, color=color, alpha=0.4)
        self.ax.add_patch(patch)
        self.ax.add_patch(front_fill)

        # draw text of location, dimension, and rotation
        self.ax.text(
            corners_2D[:, 1][0],
            corners_2D[:, 1][1],
            f"Loc: ({loc[0]:.2f}, {loc[1]:.2f}, {loc[2]:.2f})\nDim: ({dim[0]:.2f}, {dim[1]:.2f}, {dim[2]:.2f})\nYaw: {rot_y:.2f}",
            fontsize=8,
            color="white",
            bbox=dict(facecolor=color, alpha=0.4, pad=0.5),
        )

    def plot(
        self,
        img = None,
        class_object: str = None,
        bbox = None, # bbox 2d [xmin, ymin, xmax, ymax]
        dim = None, # dimension of the box (l, w, h)
        loc = None, # location of the box (x, y, z)
        rot_y = None, # rotation of the box around y-axis
    ):
        """plot 3d bbox and bev"""
        # initialize bev image
        bev_img = np.zeros((self.shape, self.shape, 3), np.uint8)

        # loop through all detections
        if class_object in self.object_list:
            # self.draw_3dbox(class_object, bbox, dim, loc, rot_y)
            self.draw_bev(dim, loc, rot_y, class_object)

        # visualize 3D bounding box
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img_rgb)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # plot camera view range
        x1 = np.linspace(0, self.shape / 2)
        x2 = np.linspace(self.shape / 2, self.shape)
        self.ax2.plot(x1, self.shape / 2 - x1, ls="--", color="grey", linewidth=1, alpha=0.5)
        self.ax2.plot(x2, x2 - self.shape / 2, ls="--", color="grey", linewidth=1, alpha=0.5)
        self.ax2.plot(self.shape / 2, 0, marker="+", markersize=16, markeredgecolor="red")

        # visualize bird eye view (bev)
        self.ax2.imshow(bev_img, origin="lower")
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        # Define colors and labels for legend
        legend_colors = [self.COLOR[class_name] for class_name in self.COLOR]
        legend_labels = list(self.COLOR.keys())

        # Create custom legend entries with colored rectangles
        legend_entries = [mpatches.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none', label=label) 
                          for color, label in zip(legend_colors, legend_labels)]

        # Add the custom legend to ax2
        self.ax2.legend(handles=legend_entries, loc='lower right', fontsize='x-small', framealpha=0.5)

        # Define circle parameters
        num_circles = 8
        radius_increment = 0.1
        center = (0.5, 0.01)  # Adjusted y-coordinate

        for i in range(num_circles):
            radius = (i + 1) * radius_increment
            circle = plt.Circle(center, radius, fill=False, color='blue')
            self.ax3.add_patch(circle)
            
        # Draw the circles on bev_img
        for i in range(num_circles):
            radius = (i + 1) * radius_increment
            cv2.circle(bev_img, (int(center[0] * self.shape), int(center[1] * self.shape)), int(radius * self.shape), (0, 0, 255), 1)

            # Add text above the circle
            text_content = f'{round((i + 1) * 9.6, 2)} m'
            text_position = ( 30 , center[1] * self.shape + radius * self.shape + 10)  # Adjusted position
            # print(text_position)
            self.ax3.text(text_position[0], text_position[1], text_content, color='white', fontsize=6, va='center', ha='center')

        # Draw the circles on bev_img
        for i in range(num_circles):
            radius = (i + 1) * radius_increment
            cv2.circle(bev_img, (int(center[0] * self.shape), int(center[1] * self.shape)), int(radius * self.shape), (0, 0, 255), 1)


        # plot camera view range
        x1 = np.linspace(0, self.shape / 2)
        x2 = np.linspace(self.shape / 2, self.shape)
        # self.ax3.plot(x1, self.shape / 2 - x1, ls="--", color="grey", linewidth=1, alpha=0.5)
        # self.ax3.plot(x2, x2 - self.shape / 2, ls="--", color="grey", linewidth=1, alpha=0.5)
        self.ax3.plot(self.shape / 2, 0, marker="+", markersize=2, markeredgecolor="red")

        # Display the image
        self.ax3.imshow(bev_img, origin="lower")
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.ax3.set_aspect('equal', adjustable='box')
        self.ax3.legend(handles=legend_entries, loc='lower right', fontsize='x-small', framealpha=0.5)

        # self.fig2.savefig("testing_bev.png") ##testing
        # bevimg_ = cv2.imread('testing_bev.png')
        # cv2.imshow('testing_bev', bevimg_)
        # cv2.waitKey(1)



    def save_plot(self, path1, path2, name):
        self.fig.savefig(
            os.path.join(path1, f"{name}.png"),
            dpi=self.fig.dpi,
            bbox_inches="tight",
            pad_inches=0.0,
        )
            # Close the figure
        plt.close(self.fig)

        self.fig2.savefig(os.path.join(path2, f"{name}.png"))
        # Close the figure
        plt.close(self.fig2)

    def show_result(self, path, name):
        # Draw the Matplotlib figure
        self.fig.canvas.draw()

        # Convert the figure to an image array
        image = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # Display the image using OpenCV
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # self.fig.close()
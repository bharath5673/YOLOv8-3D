import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} to control the verbosity
import numpy as np
import csv
import tensorflow as tf
from .Plotting import *

class KITTILoader():
    def __init__(self, subset='training'):
        super(KITTILoader, self).__init__()

        self.base_dir = '/home/bharath/Downloads/test_codes/3Dbbox/kitti/'
        self.KITTI_cat = ['Car', 'Cyclist', 'Pedestrian']

        label_dir = os.path.join(self.base_dir, subset, 'label_2')
        image_dir = os.path.join(self.base_dir, subset, 'image_2')

        self.image_data = []
        self.images = []

        for i, fn in enumerate(os.listdir(label_dir)):
            label_full_path = os.path.join(label_dir, fn)
            image_full_path = os.path.join(image_dir, fn.replace('.txt', '.png'))

            self.images.append(image_full_path)
            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw', 'dl',
                          'lx', 'ly', 'lz', 'ry']
            with open(label_full_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):

                    if row['type'] in self.KITTI_cat:
                        if subset == 'training':
                            new_alpha = get_new_alpha(row['alpha'])
                            dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                            annotation = {'name': row['type'], 'image': image_full_path,
                                          'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                          'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                          'dims': dimensions, 'new_alpha': new_alpha}

                        elif subset == 'eval':
                            dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                            translations = np.array([float(row['lx']), float(row['ly']), float(row['lz'])])
                            annotation = {'name': row['type'], 'image': image_full_path,
                                          'alpha': float(row['alpha']),
                                          'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                          'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                          'dims': dimensions, 'trans': translations, 'rot_y': float(row['ry'])}


                        self.image_data.append(annotation)

    def get_average_dimension(self):
        dims_avg = {key: np.array([0, 0, 0]) for key in self.KITTI_cat}
        dims_cnt = {key: 0 for key in self.KITTI_cat}

        for i in range(len(self.image_data)):
            current_data = self.image_data[i]
            if current_data['name'] in self.KITTI_cat:
                dims_avg[current_data['name']] = dims_cnt[current_data['name']] * dims_avg[current_data['name']] + \
                                                 current_data['dims']
                dims_cnt[current_data['name']] += 1
                dims_avg[current_data['name']] /= dims_cnt[current_data['name']]
        return dims_avg, dims_cnt

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


class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi




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


class detectionInfo(object):
    def __init__(self, line):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi




###########################




class KITTILoader():
    def __init__(self, subset='training'):
        super(KITTILoader, self).__init__()

        self.base_dir = '/home/bharath/Downloads/test_codes/3Dbbox/kitti/'
        self.KITTI_cat = ['Car', 'Cyclist', 'Pedestrian']

        label_dir = os.path.join(self.base_dir, subset, 'label_2')
        image_dir = os.path.join(self.base_dir, subset, 'image_2')

        self.image_data = []
        self.images = []

        for i, fn in enumerate(os.listdir(label_dir)):
            label_full_path = os.path.join(label_dir, fn)
            image_full_path = os.path.join(image_dir, fn.replace('.txt', '.png'))

            self.images.append(image_full_path)
            fieldnames = ['type', 'truncated', 'occluded', 'alpha', 'xmin', 'ymin', 'xmax', 'ymax', 'dh', 'dw', 'dl',
                          'lx', 'ly', 'lz', 'ry']
            with open(label_full_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file, delimiter=' ', fieldnames=fieldnames)

                for line, row in enumerate(reader):

                    if row['type'] in self.KITTI_cat:
                        if subset == 'training':
                            new_alpha = get_new_alpha(row['alpha'])
                            dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                            annotation = {'name': row['type'], 'image': image_full_path,
                                          'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                          'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                          'dims': dimensions, 'new_alpha': new_alpha}

                        elif subset == 'eval':
                            dimensions = np.array([float(row['dh']), float(row['dw']), float(row['dl'])])
                            translations = np.array([float(row['lx']), float(row['ly']), float(row['lz'])])
                            annotation = {'name': row['type'], 'image': image_full_path,
                                          'alpha': float(row['alpha']),
                                          'xmin': int(float(row['xmin'])), 'ymin': int(float(row['ymin'])),
                                          'xmax': int(float(row['xmax'])), 'ymax': int(float(row['ymax'])),
                                          'dims': dimensions, 'trans': translations, 'rot_y': float(row['ry'])}


                        self.image_data.append(annotation)

    def get_average_dimension(self):
        dims_avg = {key: np.array([0, 0, 0]) for key in self.KITTI_cat}
        dims_cnt = {key: 0 for key in self.KITTI_cat}

        for i in range(len(self.image_data)):
            current_data = self.image_data[i]
            if current_data['name'] in self.KITTI_cat:
                dims_avg[current_data['name']] = dims_cnt[current_data['name']] * dims_avg[current_data['name']] + \
                                                 current_data['dims']
                dims_cnt[current_data['name']] += 1
                dims_avg[current_data['name']] /= dims_cnt[current_data['name']]
        return dims_avg, dims_cnt

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

def compute_orientaion_(P2, xmax, ymin, alpha):
    x = (xmax + ymin) / 2
    # compute camera orientation
    u_distance = x - P2[0, 2]
    focal_length = P2[0, 0]
    rot_ray = np.arctan(u_distance / focal_length)
    # global = alpha + ray
    rot_global = alpha + rot_ray

    # local orientation, [0, 2 * pi]
    # rot_local = alpha + np.pi / 2
    rot_local = get_new_alpha(alpha)

    rot_local = tf.round(rot_local)
    rot_global = tf.round(rot_global)
    return rot_global, rot_local

def translation_constraints_(P2, bbox, rot_local, rot_global, h, w, l):
    [xmin, ymin, xmax, ymax] = bbox
    # rotation matrix
    R = np.array([[ np.cos(rot_global), 0,  np.sin(rot_global)],
                  [          0,             1,             0          ],
                  [-np.sin(rot_global), 0,  np.cos(rot_global)]])
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    I = np.identity(3)

    xmin_candi, xmax_candi, ymin_candi, ymax_candi = box3d_candidate_(rot_local=rot_local, soft_range=8, h=h, w=w, l=l)
    # print(xmin_candi, xmax_candi, ymin_candi, ymax_candi)

    if xmin_candi != 0 or xmax_candi != 0 or ymin_candi != 0 or ymax_candi != 0:
      X  = np.bmat([xmin_candi, xmax_candi,
                    ymin_candi, ymax_candi])
      # X: [x, y, z] in featuresect coordinate
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
      tx, ty, tz = [float(np.around(tran[0], 2)) for tran in Tran]
    else:
      return 0, 0, 0  
    return tx, ty, tz


def box3d_candidate_(rot_local, soft_range, h, w, l):

    x_corners = [l, l, l, l, 0, 0, 0, 0]
    y_corners = [h, 0, h, 0, h, 0, h, 0]
    z_corners = [0, 0, w, w, w, w, 0, 0]

    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
    point1 = corners_3d[0, :]
    point2 = corners_3d[1, :]
    point3 = corners_3d[2, :]
    point4 = corners_3d[3, :]
    point5 = corners_3d[6, :]
    point6 = corners_3d[7, :]
    point7 = corners_3d[4, :]
    point8 = corners_3d[5, :]

    # set up projection relation based on local orientation
    xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

    if 0 < rot_local < np.pi / 2:
        xmin_candi = point8
        xmax_candi = point2
        ymin_candi = point2
        ymax_candi = point5

    if np.pi / 2 <= rot_local <= np.pi:
        xmin_candi = point6
        xmax_candi = point4
        ymin_candi = point4
        ymax_candi = point1

    if np.pi < rot_local <= 3 / 2 * np.pi:
        xmin_candi = point2
        xmax_candi = point8
        ymin_candi = point8
        ymax_candi = point1

    if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
        xmin_candi = point4
        xmax_candi = point6
        ymin_candi = point6
        ymax_candi = point5

    # soft constraint
    div = soft_range * np.pi / 180
    if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
        xmin_candi = point8
        xmax_candi = point6
        ymin_candi = point6
        ymax_candi = point5

    if np.pi - div < rot_local < np.pi + div:
        xmin_candi = point2
        xmax_candi = point4
        ymin_candi = point8
        ymax_candi = point1

    return xmin_candi, xmax_candi, ymin_candi, ymax_candi




def calc_theta_ray(img, box_2d, proj_matrix):
    """
    Calculate global angle of object, see paper
    """
    width = img.shape[1]
    # Angle of View: fovx (rad) => 3.14
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box_2d[1] + box_2d[0]) / 2
    dx = center - (width/2)

    mult = 1
    if dx < 0:
        mult = -1
    dx = abs(dx)
    angle = np.arctan((2*dx*np.tan(fovx/2)) / width)
    angle = angle * mult

    return angle


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
def calc_location_(dimension, proj_matrix, box_2d, alpha, theta_ray):
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
        A = np.zeros([4,3], dtype=float)
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



def plot3d(img, proj_matrix, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location_(dimensions, proj_matrix, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray
    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)
    plot_3d_box(img, proj_matrix, orient, dimensions, location) # 3d boxes

    return location


class KITTIObject():
    """
    utils for YOLO3D
    detectionInfo is a class that contains information about the detection
    """
    def __init__(self, line = np.zeros(16)):
        self.name = line[0]

        self.truncation = float(line[1])
        self.occlusion = int(line[2])

        # local orientation = alpha + pi/2
        self.alpha = float(line[3])

        # in pixel coordinate
        self.xmin = float(line[4])
        self.ymin = float(line[5])
        self.xmax = float(line[6])
        self.ymax = float(line[7])

        # height, weigh, length in object coordinate, meter
        self.h = float(line[8])
        self.w = float(line[9])
        self.l = float(line[10])

        # x, y, z in camera coordinate, meter
        self.tx = float(line[11])
        self.ty = float(line[12])
        self.tz = float(line[13])

        # global orientation [-pi, pi]
        self.rot_global = float(line[14])

        # score
        self.score = float(line[15])

    def member_to_list(self):
        output_line = []
        for name, value in vars(self).items():
            output_line.append(value)
        return output_line

    def box3d_candidate(self, rot_local, soft_range):
        x_corners = [self.l, self.l, self.l, self.l, 0, 0, 0, 0]
        y_corners = [self.h, 0, self.h, 0, self.h, 0, self.h, 0]
        z_corners = [0, 0, self.w, self.w, self.w, self.w, 0, 0]

        x_corners = [i - self.l / 2 for i in x_corners]
        y_corners = [i - self.h for i in y_corners]
        z_corners = [i - self.w / 2 for i in z_corners]

        corners_3d = np.transpose(np.array([x_corners, y_corners, z_corners]))
        point1 = corners_3d[0, :]
        point2 = corners_3d[1, :]
        point3 = corners_3d[2, :]
        point4 = corners_3d[3, :]
        point5 = corners_3d[6, :]
        point6 = corners_3d[7, :]
        point7 = corners_3d[4, :]
        point8 = corners_3d[5, :]

        # set up projection relation based on local orientation
        xmin_candi = xmax_candi = ymin_candi = ymax_candi = 0

        if 0 < rot_local < np.pi / 2:
            xmin_candi = point8
            xmax_candi = point2
            ymin_candi = point2
            ymax_candi = point5

        if np.pi / 2 <= rot_local <= np.pi:
            xmin_candi = point6
            xmax_candi = point4
            ymin_candi = point4
            ymax_candi = point1

        if np.pi < rot_local <= 3 / 2 * np.pi:
            xmin_candi = point2
            xmax_candi = point8
            ymin_candi = point8
            ymax_candi = point1

        if 3 * np.pi / 2 <= rot_local <= 2 * np.pi:
            xmin_candi = point4
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        # soft constraint
        div = soft_range * np.pi / 180
        if 0 < rot_local < div or 2*np.pi-div < rot_local < 2*np.pi:
            xmin_candi = point8
            xmax_candi = point6
            ymin_candi = point6
            ymax_candi = point5

        if np.pi - div < rot_local < np.pi + div:
            xmin_candi = point2
            xmax_candi = point4
            ymin_candi = point8
            ymax_candi = point1

        return xmin_candi, xmax_candi, ymin_candi, ymax_candi

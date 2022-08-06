from cmath import rect
from cv2 import cvtColor, waitKey
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from depth_map import dense_map

# Class for the calibration matrices for KITTI data
class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P = calibs['K']
        self.P = np.reshape(self.P, [3,3])

        self.L2W = calibs['Tr_velo_to_world']
        self.L2W = np.reshape(self.L2W, [3,4])

        self.W2C = calibs['Tr_world_to_cam']
        self.W2C = np.reshape(self.W2C,[3,4])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    # From LiDAR coordinate system to Camera Coordinate system
    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        # print(pts_3d_lidar.shape)
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))
        # print(pts_3d_hom.shape)
        pts_3d_cam_ref = np.dot(self.L2W, np.transpose(pts_3d_hom))
        # print(pts_3d_cam_ref.shape)
        # print(pts_3d_cam_ref.shape)
        pts_3d_cam_ref = np.transpose(pts_3d_cam_ref)
        # print(pts_3d_cam_ref.shape)
        pts_3d_cam_ref_hom = np.hstack((pts_3d_cam_ref, np.ones((n,1))))
        pts_3d_cam_rec = np.dot(self.W2C, np.transpose(pts_3d_cam_ref_hom))
        print(pts_3d_cam_rec.shape)
        # pts_3d_cam_rec = pts_3d_cam_ref
        pts_3d_cam_rec = np.transpose(pts_3d_cam_rec)
        print(pts_3d_cam_rec.shape)
        return pts_3d_cam_rec
    
    # From Camera Coordinate system to Image frame
    def rect2Img(self, rect_pts):
        n = rect_pts.shape[0]
        # points_hom = np.hstack((rect_pts, np.ones((n,1))))
        points_2d = np.dot(rect_pts, np.transpose(self.P)) # nx3
        points_2d[:,0] /= points_2d[:,2]
        points_2d[:,1] /= points_2d[:,2]
        
        mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= 2040) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= 1086)
        mask = mask & (rect_pts[:,2] > 2)
        return points_2d[mask,0:2], mask
    

if __name__ == "__main__":
    root = "data/"
    image_dir = os.path.join(root, "undist_1/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "1point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("1",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p1/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1
    
    image_dir = os.path.join(root, "undist_2/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "2point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("2",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p2/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1

    image_dir = os.path.join(root, "undist_3/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "3point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("3",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p3/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1
    image_dir = os.path.join(root, "undist_4/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "4point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("4",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p4/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1
    image_dir = os.path.join(root, "undist_5/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "5point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("5",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p5/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1
    image_dir = os.path.join(root, "undist_6/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "6point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("6",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p6/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1
    image_dir = os.path.join(root, "undist_7/")
    img_list = os.listdir(image_dir)
    # print(len(img_list))
    velodyne_dir = os.path.join(root, "7point/")
    calib_dir = os.path.join(root, "calib/")
    len_list = os.listdir(velodyne_dir)
    # print(len(len_list))
    # Data id
    cur_id = 0
    for cur_id in range(len(len_list)):
        print("7",cur_id)
        # Loading the image
        img = cv2.imread(os.path.join(image_dir, "%006d.png" % cur_id), cv2.IMREAD_COLOR)
        # img = cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        # Loading the LiDAR data
        lidar = np.fromfile(os.path.join(velodyne_dir, "%010d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)
        # Loading Calibration
        calib = Calibration(os.path.join(calib_dir, "000001.txt"))
        # From LiDAR coordinate system to Camera Coordinate system
        lidar_rect = calib.lidar2cam(lidar[:,0:3])
        # From Camera Coordinate system to Image frame
        lidarOnImage, mask = calib.rect2Img(lidar_rect)
        # print(lidarOnImage.shape[0])
        # print((lidar_rect[mask,2].reshape(-1,1)).shape)
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        # print(lidarOnImage.shape)
        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # print(np.max(out))
        # print(np.min(out))
        # print(out.shape)
        # dst = cv2.addWeighted(img, 0.5, out, 0.5, 0)
        # plt.figure(figsize=(1086,2040))
        # plt.imshow(out); plt.show()
        plt.imsave("p7/depth_map_%06d.png" % cur_id, out)
        # cv2.imwrite("gangnam2_2/depth_map_%06d.png" % cur_id, out)
        cur_id += 1

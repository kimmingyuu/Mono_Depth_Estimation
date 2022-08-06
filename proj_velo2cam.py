import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os

path = "undist_2/"
dir_list = os.listdir(path)
dir_list = sorted(dir_list)
velo_path = "/home/kmg/Desktop/ACELAB_Project/DenseMap/data/2point/"
velo_list = os.listdir(velo_path)
velo_list = sorted(velo_list)

 # P2 (3 x 4) for left eye
#P2 = np.matrix([1046.688720703125,0.0,1033.3313677806436,0.0,0.0,1277.919921875,460.2549448068021,0.0,0.0,0.0,1.0,0.0],dtype=np.float64).reshape(3,4)
projection_matrix = np.matrix([1046.688720703125,0.0,1033.3313677806436,0.0,0.0,1277.919921875,460.2549448068021,0.0,0.0,0.0,1.0,0.0],dtype=np.float64).reshape(3,4)
#print(projection_matrix)
RT_world_to_cam = np.matrix([0.0122169643644543,-0.9999223846340135,0.0024434585213449255,0.19247596939426734,0.08370134546701578,-0.0014124149699195523,-0.9964898844699649,1.2632069552043956,0.996415992720274,0.012378602173938212,0.0836775935330984,-2.8884357686348134],dtype=np.float64).reshape(3,4)
#print(RT_world_to_cam)
#RT_world_to_cam = np.insert(RT_world_to_cam,3,values=[0,0,0,1],axis=0)
RT_vel_to_world = np.matrix([0.9999999553735484,-3.4906585032797835e-05,0.000296705968304882,1.5,3.4906583496307106e-05,0.9999999993907651,1.0356992118682821e-08,0.03,-0.0002967059684856456,8.271806125530277e-25,0.9999999559827831,2]).reshape(3,4)
RT_vel_to_world = np.insert(RT_vel_to_world,3,values=[0,0,0,1],axis=0)
k = np.matrix([1365.4887468866116,0.0,1026.5997744850633,0.0,1366.2954658193316,468.9522311262687,0.0,0.0,1.0]).reshape(3,3)
#k = np.matrix([1046.688720703125,0.0,1033.3313677806436,0.0,1277.919921875,460.2549448068021,0.0,0.0,1.0]).reshape(3,3)
def vel2cam(binary,name,img):


        # read raw data from binary
    global projection_matrix,k,RT_world_to_cam,RT_vel_to_world
    scan = np.fromfile(binary, dtype=np.float32).reshape((-1,4))
    points = scan[:, 0:3] # lidar xyz (front, left, up)
    # TODO: use fov filter? 
    velo = np.insert(points,3,1,axis=1).T
    velo = np.delete(velo,np.where(velo[0,:]<0),axis=1)
    print("velo:",velo)
    print("R0_rect:",RT_world_to_cam)
    print("Tr_velo_to_cam:",RT_vel_to_world)
    cam = k * RT_world_to_cam * RT_vel_to_world * velo
    cam = np.delete(cam,np.where(cam[2,:]<0)[1],axis=1)
    # get u,v,z
    cam[:2] /= cam[2,:]
    # do projection staff
    plt.figure(figsize=(12,5),dpi=96,tight_layout=True)
    png = mpimg.imread(img)
    IMG_H,IMG_W,_ = png.shape
    # restrict canvas in range
    plt.axis([0,IMG_W,IMG_H,0])
    # plt.imshow(png)
    # filter point out of canvas
    u,v,z = cam
    u_out = np.logical_or(u<0, u>IMG_W)
    v_out = np.logical_or(v<0, v>IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam,np.where(outlier),axis=1)
    # generate color map from depth
    u,v,z = cam
    plt.scatter([u],[v],c=[z],alpha=0.5,s=0.5)
    plt.title(name)
    plt.savefig(f'K_test2/{name}',bbox_inches='tight')

for sn, ns in zip(dir_list, velo_list):
    name = sn
    img = str(path) + sn
    binary = str(velo_path) + ns
    #with open(f'./testing/calib/{name}.txt','r') as f:
    #    calib = f.readlines()
    print(img)
    print(binary)
    vel2cam(binary,name,img)

#plt.show()

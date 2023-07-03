import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
import torch
import torch.nn.functional as nn

# Removing off-label heat maps
def miss_red(img,label):
    [x_l,y_l]=label.shape
    for i in range(x_l):
        for j in range(y_l):
            for k in range(3):
                if label[i,j]==0:
                    img[i,j,k]=0
                if img[i,j,0]==0 and img[i,j,1]==0:
                    img[i,j,k]=0
    return img

# Calculating IoU
def iou(gray,label):
    [x,y]=label.shape
    image= np.resize(gray,(x,y))
    maps=np.zeros((x,y))
    count=0
    for ii in range(x):
        for jj in range(y):
            if image[ii,jj]!=0 and label[ii,jj]!=0:
                count=count+1
            if image[ii,jj]!=0:
                maps[ii,jj]=1
            if label[ii,jj]!=0:
                maps[ii,jj]=1
    return count/np.sum(maps)

# Image Slicing
def slice(cam, label, image, Thresh, path_image, path_label, line=None):
    RoIs = 0
    slice_K=0
    [x,y]=label.shape
    x_small = x // 2
    y_small = y // 2
    x_step = x//10
    y_step = y//10
    all_slice_K = []
    small_img_names = []
    for i in range(0, (x-x_small)//x_step):
        for j in range(0, (y-y_small)//y_step):
            small = label[i*x_step:x_small+i*x_step, j*y_step:y_small+j*y_step]
            small_ROI = cam[i*x_step:x_small+i*x_step, j*y_step:y_small+j*y_step]
            small_IoU = iou(small, small_ROI)
            print(line+"-"+str(i)+"-"+str(j)+"-"+str(small_IoU))
            if small_IoU < Thresh and small_IoU > 0:
                all_slice_K.append(slice_K)
                small_image=image[i*x_step:x_small+i*x_step,j*y_step:y_small+j*y_step,:]
                small_label=small

                cv2.imwrite("F:/code/deeplab_v3/up_test/small_gray/"+str(RoIs)+".jpg",small_image)#test
                cv2.imwrite("F:/code/deeplab_v3/up_test/small_label/"+str(RoIs)+".png",small_label)#test
                cv2.imwrite(path_image + line + "_" + str(RoIs)+".jpg",small_image)#image
                cv2.imwrite(path_label + line + "_" + str(RoIs)+".png",small_label)#label
                small_img_names.append(line+"_"+str(RoIs)+'\n')
                RoIs=RoIs+1

            slice_K=slice_K+1
    return all_slice_K, small_img_names 

# Slice the image at position indexes
def new_slice(image, indexes):
    x, y = image.shape #All images sliced at position K
    x_small = x // 2
    y_small = y // 2
    x_step = x//10
    y_step = y//10
    t = (x-x_small)//x_step
    for i in range(0, (x-x_small)//x_step):
        for j in range(0, (y-y_small)//y_step):
            index = i * t + j
            if index==indexes:
                small_image = image[i*x_step:x_small+i*x_step,j*y_step:y_small+j*y_step] # (166, 250)
                return small_image

# eq9 and eq10 in paper
def condition(H, h, mapH, maph):
    delta = 1e-8
    x, y = H.shape
    map = np.zeros((x, y))
    order = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if h[i,j]>H[i, j] and h[i, j]/(maph[i,j]+delta) > H[i, j]/(mapH[i,j]+delta) and (mapH[i,j]-H[i,j]!=0): #eq10
                map[i,j] = (H[i,j] / mapH[i,j]) / ((H[i,j] / mapH[i,j]) + (h[i,j] / maph[i,j])) #eq9
                order[i,j] = h[i,j] / maph[i, j]
    return map, order

# Screening information ratio and (1-n)*f maximum pixel position of the overlapping part
def ratio(large_H, all_small_H, all_indexes, small_f, k):
    n = []
    f_map = []

    sH = new_slice(large_H[0], all_indexes[0][0])# (166, 250)
    x_sH, y_sH = sH.shape
    for i in range(k): # Replace all small slices in the form of large sizes with small sizes
        for j in range(5):           
            all_small_H[i][j] = np.resize(all_small_H[i][j], [x_sH, y_sH])
    
    f_maps = []
    f_ratios = []
    f_orders = []
    for j in range(4):
        x_sH, y_sH = large_H[j].shape
        map = np.ones((x_sH, y_sH))
        order = np.zeros((x_sH, y_sH))

        maps = []
        ratios = []
        orders = []

        for i  in range(k):
            sH = new_slice(large_H[j], all_indexes[i][0])
            mapH = new_slice(large_H[4], all_indexes[i][0])
            ma, orde = condition(sH, all_small_H[i][j], mapH, all_small_H[i][4]) # Small size's map、order
           
            x, y = large_H[j].shape
            x_small = x // 2
            y_small = y // 2
            x_step = x//10
            y_step = y//10
            t = (x-x_small)//x_step
            for ii in range(0, (x-x_small)//x_step):
                for jj in range(0, (y-y_small)//y_step):
                    index = ii * t + jj
                    if index==all_indexes[i][0]:
                        map[ii*x_step:x_small+ii*x_step,jj*y_step:y_small+jj*y_step] = ma # Get the map in large size
                        order[ii*x_step:x_small+ii*x_step,jj*y_step:y_small+jj*y_step] = orde # Get the order in large size                        

            each_f = small_f[i][j]
            h = each_f.shape[2]
            w = each_f.shape[3]
            tensor_map = torch.from_numpy(map[None,None,:,:])
            down_tensor_map = nn.interpolate(tensor_map,(h,w))
            down_tensor_map = down_tensor_map.squeeze(0).squeeze(0)
            down_map = down_tensor_map.numpy()
    
            # down_map = np.resize(map, (h,w))
            down_ratio=(1 - down_map)[None, None,:,:] * each_f #1*256*72*72

            tensor_order = torch.from_numpy(order[None,None,:,:])
            down_tensor_order = nn.interpolate(tensor_order,(h,w))
            down_tensor_order = down_tensor_order.squeeze(0).squeeze(0)
            down_order = down_tensor_order.numpy()

            ratios.append(down_ratio)
            maps.append(down_map)
            orders.append(down_order)
        f_maps.append(maps)
        f_ratios.append(ratios)
        f_orders.append(orders)

    for j in range(4):
        x, y = f_maps[j][0].shape
        chananl = f_ratios[j][0].shape[1]
        for d in range(k):
            an_map = np.expand_dims(f_maps[j][d], axis=0)
            an_order = np.expand_dims(f_orders[j][d], axis=0)
            concate_map = an_map if d==0 else np.concatenate((concate_order, an_map), axis=0)
            concate_order = an_order if d==0 else np.concatenate((concate_order, an_order), axis=0)
            concate_ratio = f_ratios[j][d] if d==0 else np.concatenate((concate_ratio, f_ratios[j][d]), axis=0)
        max_order = np.max(concate_order, axis=0)
        max_order_index = np.argmax(concate_order, axis=0)
        n_map = np.zeros((x,y))
        f_ratio = np.zeros((1,chananl,x,y))
        for m in range(x):
            for p in range(y):
                idx = max_order_index[m,p]
                n_map[m,p] = concate_map[idx,m,p]
                f_ratio[:,:,m,p] = concate_ratio[idx,:,m,p]

        f_map.append(f_ratio) #(1-n)*f
        n.append(n_map)  #n  
    return n, f_map

# The final information rate used for replacement and (1-n)*f are obtained
def caculate_n_f_map(all_large_H, all_small_H, large_f, small_f, all_indexes):
    all_f_map = []
    all_n = []
    for i in range(len(all_large_H)):
        roi_small_H = []
        k = 0
        for j in range(len(all_small_H)):
            if all_indexes[j][1] == i:
                roi_small_H.append(all_small_H[j])
                k = k + 1
        if len(roi_small_H)==0:            
            f_map = [0, 0, 0, 0]
            n = 1
            all_f_map.append(f_map)
            all_n.append(n)            
        else:
            bbb=all_large_H[i]
            n, f_map = ratio(all_large_H[i], roi_small_H, all_indexes, small_f, k)
            all_f_map.append(f_map)
            all_n.append(n) 
    return all_f_map, all_n #4 images
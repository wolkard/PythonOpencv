import cv2 
import numpy as np
import math
#获取最优tant集合，以便求取旋转角度
def get_cluster(arr):
    new_arr = np.sort(arr)
    if len(new_arr)!=1:
        
        diff = new_arr[1:]-new_arr[:-1]#值的差
        thd = (max(diff)-min(diff))/30+min(diff)#选取值差 的阈值

        diff_idx = np.where(diff>=thd)[0]#选取符合阈值的 差的下标
        if diff_idx[0]==0 and diff_idx[-1]!=len(diff):
            new_diff_idx = np.zeros(len(diff_idx)+1)
            new_diff_idx[-1] = len(diff)
            new_diff_idx[:-1]=diff_idx
        elif diff_idx[0]!=0 and diff_idx[-1]==len(diff):
            new_diff_idx = np.zeros(len(diff_idx)+1)
            new_diff_idx[-1] = len(diff)
            new_diff_idx[:-1]=diff_idx
        elif diff_idx[0]!=0 and diff_idx[-1]!=len(diff):
            new_diff_idx = np.zeros(len(diff_idx)+2)
            new_diff_idx[0] = 0
            new_diff_idx[-1] = len(diff)
            new_diff_idx[1:-1]=diff_idx
        
        diff_idx_diff = new_diff_idx[1:]-new_diff_idx[:-1]
        diff_idx_diff_idx = np.where(diff_idx_diff==max(diff_idx_diff))#选出所有集合中，元素最多的集合下标
        diff_idx_diff_idx = diff_idx_diff_idx[len(diff_idx_diff_idx)//2][0]
        
        left = int(new_diff_idx[diff_idx_diff_idx]+1)#开始下标
        right = int(new_diff_idx[diff_idx_diff_idx+1]+1)#结束下标
        return new_arr[left:right]
    else:
        return new_arr

#获取到所有直线tant值，并且去掉 0和无穷大的值
def get_theta_arr(lines):
    theta_arr=np.zeros(len(lines))
    for num in range(len(lines)):
        x1, y1, x2, y2 = lines[num][0]
        if y1 - y2 == 0:
            theta=0.0
        else:
            if x2 - x1==0:
                theta=0.0
            else:
                theta = (y2 - y1) / (x2 - x1)
        theta_arr[num]=theta
    theta_arr = np.delete(theta_arr,np.where(theta_arr==0))#删除为0的数据
    return theta_arr

#图像处理
def get_good_img(img,close_y,close_x,open_y,open_x):
    kernel = np.ones((4,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    img=255-img
    
    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,101,8)
    img = cv2.medianBlur(img,3)
    return img
    
def run_cut_line(img):

    #1 图像处理
    height = int(img.shape[0]/img.shape[1]*1000)
    new_img = cv2.resize(img,(1000,height))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    good_img = get_good_img(gray,1,30,5,8)
    
    #霍夫直线变换
    lines = cv2.HoughLinesP(good_img,2, np.pi/180, 30, minLineLength=100, maxLineGap=10)

    #获取到所有直线tant值，并且去掉 0和无穷大的值
    theta_arr = get_theta_arr(lines)

    #获取最优tant集合，以便求取旋转角度
    all_num=10
    while all_num:
        theta_arr  = get_cluster(theta_arr)
        if len(theta_arr)==1:
            break
        all_num-=1
    atan_val = theta_arr[len(theta_arr)//2]

    #根据tant值求应旋转角度
    angle = math.atan(atan_val)
    angle = angle * (180 / np.pi)
    
    #对图像进行旋转
    h, w = img.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #new_save_img = np.zeros((img.shape[0],img.shape[1]*2+10,3),dtype=np.uint8)

    #new_save_img[:,:img.shape[1]]=img
    #new_save_img[:,img.shape[1]+10:]=rotated
    #cv2.imwrite("finish_imgs/"+str(4)+'.jpg',new_save_img)
    return rotated
    
#import time
#start_ = time.time()
#img = cv2.imread('new/'+str(4)+'.jpg')
#arg = run_cut_line(img)
#print(time.time()-start_)
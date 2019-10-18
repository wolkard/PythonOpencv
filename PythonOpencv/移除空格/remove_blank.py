import numpy as np
import cv2

#移除边缘
def remove_side(img):
    image=cv2.adaptiveThreshold(img.copy(),255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,21,30)
    
    image = 255-image
    img_index = np.where(image<150)
    side=3
    if min(img_index[1])-side<0:
        left=min(img_index[1])
    else:
        left = min(img_index[1])-side
    if max(img_index[1])+side>img.shape[1]:
        right = max(img_index[1])
    else:
        right = max(img_index[1])+side
        
    if min(img_index[0])-side<0:
        top=min(img_index[0])
    else:
        top = min(img_index[0])-side
    if max(img_index[0])+side>img.shape[0]:
        bottom = max(img_index[0])
    else:
        bottom = max(img_index[0])+side
    return top,bottom,left,right
#图像处理
def get_good_img(img,close_y,close_x,open_y,open_x):
    
    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY_INV,31,25)
    kernel = np.ones((2,2),np.uint8)
    img = cv2.dilate(img,kernel,iterations = 1)
    kernel = np.ones((close_y,close_x),np.uint8)
    close= cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((open_y,open_x),np.uint8)
    open_img = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    return open_img

def get_index_c(arr,th):
    top = np.where(arr>th)[0]
    index_dic = top[1:]-top[:-1]
    top_dic_index = np.where(index_dic>100)[0]
    part_right_index = top[top_dic_index]
    right_index = np.zeros(len(part_right_index)+1)
    right_index[-1] = len(arr)
    right_index[:-1] = part_right_index
    right_index = np.int64(right_index+1)

    part_left_index = top[top_dic_index]+index_dic[top_dic_index]
    left_index = np.zeros(len(part_left_index)+1)
    left_index[1:] = part_left_index
    left_index = np.int64(left_index)
    bottom_area = np.zeros(len(part_left_index))
    for num in range(len(part_left_index)):
        area = arr[part_right_index[num]:part_left_index[num]]
        if len(area)>0:
            min_index = np.where(area==np.min(area))[0]+part_right_index[num]
            bottom_area[num] = min_index[int(len(min_index)/2)]
    return bottom_area

def get_clo(img):
    img_mean_y = img.mean(axis=0)
    bottom_area = get_index_c(img_mean_y,5)
    fir_row = np.where(img_mean_y>0)[0]
    edg = 5
    if len(fir_row) >0:
        fir_row = np.where(img_mean_y>0)[0][0]
    else:
        fir_row=0
    end_row = np.where(img_mean_y>0)[0]
    if len(end_row) >0:
        end_row = np.where(img_mean_y>0)[0][-1]
    else:
        end_row=len(img_mean_y)-1
    if fir_row >edg:
        fir_row-=edg
    else:
        fir_row=0
    if len(img_mean_y)-1-end_row>edg:
        end_row+=edg
    else:
        end_row=len(img_mean_y)-1
    clo_area = np.zeros(len(bottom_area)+2,int)
    clo_area[0]=fir_row
    clo_area[1:-1]=bottom_area
    clo_area[-1]=end_row
    return clo_area
def run(img):
    img_gary = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    good_img = get_good_img(img_gary,100,5,5,5)
    clo_area = get_clo(good_img)
    clear_clo_imgs = []
    for num_clo in range(len(clo_area)-1):
        clo_img = img_gary[:,clo_area[num_clo]:clo_area[num_clo+1]]
        clo_img_save = img[:,clo_area[num_clo]:clo_area[num_clo+1]]
        top,bottom,left_clo,right_clo = remove_side(clo_img)
    
        clear_clo_img_save = clo_img_save[:,left_clo:right_clo]
        if num_clo == 0:
            new_clo_img = clear_clo_img_save
        else:
            new_clo_img = np.concatenate((new_clo_img,clear_clo_img_save), axis=1)
    return new_clo_img
if __name__=="__main__":
    """
    传入单行彩色图像
    传出去除空格的，拼接好的单行彩色图像
    """
    path="new_clo_img/1560770740.9662056.jpg"
    img = cv2.imread(path)
    clear_img = run(img)
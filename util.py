from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes 
    
    
    """
    #Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    #get the corrdinates of the intersection rectangle
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    #Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    #Union Area
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    Chuyển đổi từ dạng thô (output của network) sang dạng bounding box output (5+C). ngoài công thức bx,by,bw,bh đã biết, các obj_conf và class_conf đều cho qua hàm sigmoid
    :param prediction:  (batchsize x ((5+C)*anchorNum) x gridSize x gridSize) trả gái trị bounding box tương ứng của từng anchor box, trong từng grid tại từng image trong batch
    :param inp_dim: resolution input
    :param anchors: list các anchor (mỗi anchor gồm chiều cao và chiều rộng)
    :param num_classes: só class C
    :param CUDA:
    :return:
    '''
    
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)         #chiều cao và rộng của 1 cell
    grid_size = inp_dim // stride           #sô cell theo một chiều (chính là gridSize luôn, mục đích chỉ làm tròn)
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]  #đổi kích thước anchor box sang theo tỉ lệ với cell size

    #thực hiện sigmoid với bx, by, obj_conf
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #tạo grid index để làm offset trong việc tính toán bx, by (grid tức là index tăng dần tương ứng với vị trí cell trông hình)
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    #cộng offset cho bx, by
    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    #trải đều các anchor ra tất cả các grid, vì anchor ở các grid là như nhau, sau đó dùng để tính bw, bh
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0) #=> anchor (gridsize*gridsize*anchorNum x 2)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors  #exponention sau đó nhân với kích thước anchor để tính bw, bh

    #các class conf cũng cho qua hàm sigmoid
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    #chuyển tọa độ từ dạng tỉ lệ dang kích thước thật bằng cách nhân với kích thước 1 cell
    prediction[:,:,:4] *= stride
    
    return prediction

def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    đưa các bb predict được từ model (không còn dạng thô) và thực hiện non-max suppression
    Các bước thực hiện với mỗi image trong batch:
    1. Tìm các bb còn obj_conf lớn hơn ngưỡng confidence đầu vào
    2. với mỗi bb còn lại, tìm các class có class_conf lớn nhất của mỗi bb trong image đang xét
    3. Duyệt qua từng loại class có trong danh sách này
    4. Thực hiện Non-max suppression với các bb có cùng loại class lớn nhất là class đang xét
    5. Non-max suppress bằng các sắp xếp theo thứ tự giảm dẫn của obj_conf, tính iou của thằng cao nhất với các thằng phía sau, loại bỏ các thằng phía sau nào nếu có iou tính được nhỏ hơn ngưỡng nms_conf
    Parameters
    ----------
    prediction : output mạng network (batchsize x #bounding box x (5+C))
        5 bao gồm 4 cho hệ tọa độ của box (center và width, height), 1 cho confidence có object
    confidence : ngưỡng confidence chấp nhận là có object
        DESCRIPTION.
    num_classes : Số class C
        DESCRIPTION.
    nms_conf : IOU threshold để thực hiện Non-max suppressing
        DESCRIPTION. The default is 0.4.

    Returns
    -------
    None.

    """
    #lấy các prediction (bounding box có confidence > giá trị cho trước)
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask           #có bb có obj_conf < ngưỡng thì  cho bằng 0 hết
    
    #chuyển tọa độ bounding box theo corner
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]                         #gán lại 4 giá trị tọa độ cho biến prediction sau khi đã chuyển đổi
    
    batch_size = prediction.size(0)                 # lấy số batch

    write = False
    


    for ind in range(batch_size):           #vòng lặp chạy mỗi bức ảnh trong 1 batch 
        image_pred = prediction[ind]          #image Tensor
       #confidence threshholding 
       #NMS

        #với tất cả các bounding box trong image ind, lấy giá trị xác suất loại class lớn nhất và index của nó trong số class
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)  # tìm max confidence class (max_conf), và index (max_conf_score) trong từng bounding box (từng hàng 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)  # tuple 3 tensor, tensor ví trí bb và obj_conf của các bb, ma trận index_max_class_conf của các bb, ma trận max_class_conf_score  của các bb  , pc (max conf ) và class (index của class có confidence cao nhất tại mỗi bb)
        image_pred = torch.cat(seq, 1)              #nối tuple thành tensor có chiều (số bb x (5+2)) với 5 là obj_conf và 4 tọa độ, 2 là max_class_conf và index của nó (loại class)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))  # tạo tensor chứa index các bb có obj_conf khac 0, return ma trận mask (số bb x 1)
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
        # lấy index các class khác nhau (unique) trong số các bb có obj_conf khác 0
        #Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1])  # -1 index holds the class index
        
        #Với mỗi class trong các class unique vừa lọc ra, thực hiện non-max suppressing
        for cls in img_classes:
            # NMS

            #lấy các bb đã lọc (bb có obj_conf lớn hơn ngưỡng) của đúng loại class và class_conf != 0
            #get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            #sap xep theo chiều giảm obj_conf
            #sort the detections such that the entry with the maximum objectness
            #confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   #Number of detections
            
            for i in range(idx):
                #tính iou của các box này với các box sau box này
                #Get the IOUs of all boxes that come after the one we are looking at 
                #in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break
            
                except IndexError:
                    break
            
                #Xoa cac bb co Iou > threshold bằng cách đặt mask và nhân với mask
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
                # sau đó loại các bb có obj_conf = 0 (vì nhân với mask)
                #Remove the non-zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            #lấy id của batch và các giá trị của các bb đủ điều kiện
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      #Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_ind, image_pred_class

            #thực hiện gắn vào output để return output(số bb x 8) =>bao gồm là 4 tọa độ, 1 obj_conf, 1 class và 1 class_conf và 1 batch index
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0
    
def letterbox_image(img, inp_dim):
    '''resize image mà vẫn giữ nguyên tỉ lệ w/h'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))      #
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    thực hiện đảo phần tử chiều kênh màu và đưa ra đầu, sau đấy thêm 1 dimension vào đầu
    Returns a Variable 
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:,:,::-1].transpose((2,0,1)).copy()       #img[:,:,::-1] chiều kênh đảo ngược vị trí (-1), transpose (2,0,1): đưa chiều kênh (2) ra đầu, 2 chiều kích thước (0,1) ra sau
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)     #div : normalize
                                                                    #unsqueeze(0) thêm 1 dimension vào đầu =>(1,3,416,416) 
    return img

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

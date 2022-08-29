from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse                  #thư viện parse argument từ command line
import os 
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
        
def arg_parse():
    """
    Parse arguements to the detect module
    
    Arguments
    ----------
    --images: hình ảnh cần detect
    --det : đường dẫn lưu ảnh sau detect
    --bs: kích thước batch
    --confidence : pc. xác suất dự đoán có object 
    --nms_thresh: ngưỡng non-max suppression (IOU)
    --cfg: file config chứa thông tin mạng neuron
    --weights: file weight chứa bộ weights đã được training
    --reso: độ phân giải của ảnh input
    
    """
    
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
   
    parser.add_argument("--images", dest = 'images', help = 
                        "Image / Directory containing images to perform detection upon",
                        default = "imgs", type = str)
    parser.add_argument("--det", dest = 'det', help = 
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    
    return parser.parse_args()
  
#parse arg từ dòng lệnh   
args = arg_parse()
images = args.images                #ảnh input
batch_size = int(args.bs)           #kích thước batch size
confidence = float(args.confidence) # pc
nms_thesh = float(args.nms_thresh)  #non-max threshold (IOU)
start = 0
CUDA = torch.cuda.is_available()


#số lượng class phân loại
num_classes = 80
classes = load_classes("data/coco.names")   #load tên class từ file coco.names



##########Set up the neural network#################
print("Loading network.....")
model = Darknet(args.cfgfile)           #load network dựa theo file config
model.load_weights(args.weightsfile)    #load weights từ file cho mạng neuron này
print("Network successfully loaded")
      
  
model.net_info["height"] = args.reso    #Load lại tham số height của input đầu vào. mặc định 416 (nếu ko gọi --reso)

#check lại input dimension xem đã đạt một số yêu cầu (chưa biết yêu cầu dùng làm gì)
inp_dim = int(model.net_info["height"]) #
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()


#Set the model in evaluation mode, chuyển về chế độ prediction => muc đích tắt dropout 
model.eval()

##############tạo batch input################
#bắt đầu đọc image và thực thi prediction

read_dir = time.time()          #lấy thông tin thời gian bắt đầu read path các image
    #Detection phase
try:
    #tạo list [] chứa path các image trong thư mục imgs mặc định 
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(images))
    exit()
    
    #Tạo thư mục chứa đầu ra 
if not os.path.exists(args.det):
    os.makedirs(args.det)

load_batch = time.time()        #lấy thông tin thời gian bắt đầu load batch images
loaded_ims = [cv2.imread(x) for x in imlist]    #load list các image từ list các path images

#tạo list argument để đưa vào hàm prep_image theo thứ tự, output ra sẽ được đưa vào list các image sau khi resize 
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]        #im_dim_list là list chứa list kích thước các bức ảnh trong loaded_ims
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)            #chuyển đổi list thành tensor
                                                                    #repeat gấp đôi số cột repeat(1,2) => x1 hàng, x2 cột
                                                                    #mục đích???
#thực hiện chia các batch (mặc định batch bằng 1), batch number = ceil(len_list /batch_size)
leftover = 0
if (len(im_dim_list) % batch_size):             #nếu chiều dài list image % batch size mà != 0 tức là có phần dư 
    leftover = 1

if batch_size != 1: 
    num_batches = len(imlist) // batch_size + leftover            #cộng 1 nếu có dư leftover
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,     #tạo list các batch image, dựa vào kích thước một batch
                        len(im_batches))]))  for i in range(num_batches)]           #và nối các image trong 1 batch lại với nhau theo chiều tạo mới(unsqueeze(0)) để tạo ra một phần tử trong list batch

write = 0


if CUDA:
    im_dim_list = im_dim_list.cuda()
    
start_det_loop = time.time()
for i, batch in enumerate(im_batches):
#load the image 
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    with torch.no_grad():
        prediction = model(Variable(batch), CUDA)

    prediction = write_results(prediction, confidence, num_classes, nms_conf = nms_thesh)

    end = time.time()

    if type(prediction) == int:

        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", ""))
            print("----------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size    #transform the atribute from index in batch to index in imlist 

    if not write:                      #If we have't initialised output
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()       
try:
    output
except NameError:
    print ("No detections were made")
    exit()

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(416/im_dim_list,1)[0].view(-1,1)


output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2



output[:,1:5] /= scaling_factor

for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

draw = time.time()


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, (int(c1[0]),int(c1[1])), (int(c2[0]),int(c2[1])),color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]),int(c1[1])), (int(c2[0]),int(c2[1])),color, -1)
    cv2.putText(img, label, (int(c1[0]),  int(np.float32(c1[1] + t_size[1] + 4))), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


list(map(lambda x: write(x, loaded_ims), output))

det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format(args.det,x.split("/")[-1]))
print(det_names[0])
print (loaded_ims[0].shape)
print('AAAA')
#list(map(cv2.imwrite, det_names, loaded_ims))
cv2.imwrite('det/dog-cycle-car.png', loaded_ims[0])

end = time.time()

print("SUMMARY")
print("----------------------------------------------------------")
print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
print()
print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------")


torch.cuda.empty_cache()
    

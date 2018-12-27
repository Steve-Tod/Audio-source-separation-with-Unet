import os
import torch
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import math

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam
def hook_feature(module, input, output):
    global features_blobs
    features_blobs = output.data.cpu().numpy()
def load_model():
    global net
    global normalize
    global preprocess
    global features_blobs
    global classes
    global weight_softmax
    labels_path = 'labels.json'
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
    net.eval()
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(), normalize])
    classes = {
        int(key): value
        for (key, value) in json.load(open(labels_path, 'r')).items()
    }
    if torch.cuda.is_available():
        net = net.cuda()
def get_CAM(pic):
    idxs = [401, 402, 486, 513, 558, 642, 776, 889]
    
#    img_pil = Image.open(os.path.join(imdir, imname))
#    img_tensor = preprocess(img_pil)
    #we get np form pic, transfer into PIL form
    image_PIL = Image.fromarray(cv2.cvtColor(pic,cv2.COLOR_BGR2RGB)) 
    img_tensor = preprocess(image_PIL)
    img_variable = Variable(img_tensor.unsqueeze(0))
    
    if torch.cuda.is_available():
        img_variable = img_variable.cuda()
        
#    img = cv2.imread(os.path.join(imdir, imname))
    img = pic
    
    height, width, _ = img.shape
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()
    if torch.cuda.is_available():
        h_x = h_x.cpu()
    probs1 = h_x.numpy()
    probs = []
    x_axis = np.arange(0, 256)
    #    print x_axis
    heatmap_mean_pos = np.zeros([8])
    
    for i in range(0, 8):
        #print('{:.3f} -> {}'.format(probs1[idxs[i]], names[i]))
        CAMs = returnCAM(features_blobs, weight_softmax, [idxs[i]])
        #        print CAMs
        array = np.sum(CAMs, axis=0)  #get 256*256
        #	print array
        line = np.sum(array, axis=0)  #get 1*256
        #	print line
        pos = np.sum(line * x_axis / np.sum(line))
        #	print pos
        heatmap_mean_pos[i] = pos
        #if probs1[idxs[i]] < 0.01:
        #    probs1[idxs[i]] = 0
        probs.append(probs1[idxs[i]])
    
 ###############################  nature scence   
 #   probs = np.array(probs)
 #   probs = probs / np.sum(probs)
 #   indexProbs = probs.argsort()[-2:][::-1] 
 #   if probs[indexProbs[1]]<0.01:
 #       probs = np.zeros([8])
 #       heatmap_mean_pos = np.zeros([8])
 ###############################
#    print heatmap_mean_pos
#    print '-' * 10

#    probs = np.array(probs)
#    probs = probs / np.sum(probs)
#    indexProbs = probs.argsort()[-3:][::-1] 
#    if  probs[indexProbs[1]]> 5*probs[indexProbs[2]]:
#        probs = probs
#    else:
#        probs = np.zeros([8])
#        heatmap_mean_pos = np.zeros([8])
#    index = probs.argsort()[-2:][::-1]
#    add_judge = 0
#    if names[index[0]] == label1 or names[index[1]] == label1:
#        add_judge += 1
#    if names[index[0]] == label2 or names[index[1]] == label2:
#        add_judge += 1
    return probs, heatmap_mean_pos#, add_judge
names = [
    'accordion', 'acoustic_guitar', 'cello', 'trumpet', 'flute', 'xylophone',
    'saxophone', 'violin'
]
def getInstrumentPos(videoSrc):
    name = videoSrc.split('/')[-1]
    video = cv2.VideoCapture(videoSrc)
    print video.isOpened()
    FrameNum = video.get(7)
    pic_set = []
    # we get some frame from the video
    if video.isOpened():
        #os.mkdir('/home/yxm/test/pic/'+ name)
        pic_cnt = 0
        frame_cnt = 0
        step = int(FrameNum/50) 
        while True:
            success,frame = video.read()
            if success:
                frame_cnt += 1   #video frame indicator
                if frame_cnt % step == 0:
                    pic_set.append(frame)
                    #cv2.imwrite('/home/yxm/test/pic/' + name + '/' + str(frame_cnt)+'.jpg',frame)
            else:
                print "end to this video  " + name
                video.release()
                break      
    else:
        print "fail to open the video!"
    #begin to judge the instr
    load_model()
    finalProbs = np.zeros([8])
    position = []
    result = {}
    for item in pic_set:
        originProbs, mean_pos = get_CAM(item)
        finalProbs = finalProbs + originProbs
        position.append(mean_pos)
    position = np.mean(position, axis=0)
    print finalProbs
    finalProbs = finalProbs / np.sum(finalProbs)
    index = finalProbs.argsort()[-2:][::-1]
    print(finalProbs[index[1]])
    #redo
    if finalProbs[index[1]]< 0.03:
        finalProbsLeft1 = np.zeros([8])
        finalProbsRight1 = np.zeros([8])
        finalProbsLeft2 = np.zeros([8])
        finalProbsRight2 = np.zeros([8])
        finalProbsLeft = np.zeros([8])
        finalProbsRight = np.zeros([8])
        result = {}
        #kill = 0
        for item in pic_set:
            newimg = item[0,:,:]
            new_img = np.sum(newimg,axis = 1)
            max_gap = (0,0)
            for i in range(10,len(new_img)-10):
                if abs(int(new_img[i+1])-int(new_img[i]))>max_gap[0]:
                    max_gap = (abs(int(new_img[i+1])-int(new_img[i])),i)
            itemLeft = item[:,0:max_gap[1],:]
            itemRight = item[:,max_gap[1]:-1,:]
            h_left,w_left,_ = itemLeft.shape
            h_right,w_right,_ = itemRight.shape
            if h_left > w_left:
                itemLeft1 = itemLeft[0:w_left,:,:]
                itemLeft2 = itemLeft[h_left - w_left:-1,:,:]
            else:
                itemLeft1 = itemLeft[:,0:h_left,:]
                itemLeft2 = itemLeft[:,w_left - h_left:-1,:]
            if h_right > w_right:
                itemRight1 = itemRight[0:w_right,:,:]
                itemRight2 = itemRight[h_right - w_right:-1,:,:]
            else:
                itemRight1 = itemRight[:,0:h_right,:]
                itemRight2 = itemRight[:,w_right - h_right:-1,:]  
            
            #cv2.imwrite('/home/yxm/test/pic/' + name + '/*' + str(kill)+'.jpg',item)
            #kill+=1
            originProbsLeft1, _ = get_CAM(itemLeft1)
            originProbsRight1, _ = get_CAM(itemRight1)
            originProbsLeft2, _ = get_CAM(itemLeft2)
            originProbsRight2, _ = get_CAM(itemRight2)
            finalProbsLeft1 = finalProbsLeft1 + originProbsLeft1
            finalProbsRight1 = finalProbsRight1 + originProbsRight1
            finalProbsLeft2 = finalProbsLeft2 + originProbsLeft2
            finalProbsRight2 = finalProbsRight2 + originProbsRight2
        finalProbsLeft1 = finalProbsLeft1 / np.sum(finalProbsLeft1)
        finalProbsRight1 = finalProbsRight1 / np.sum(finalProbsRight1)
        finalProbsLeft2 = finalProbsLeft2 / np.sum(finalProbsLeft2)
        finalProbsRight2 = finalProbsRight2 / np.sum(finalProbsRight2)
        index_Left1 = finalProbsLeft1.argsort()[-1:][::-1] 
        index_Right1 = finalProbsRight1.argsort()[-1:][::-1]
        index_Left2 = finalProbsLeft1.argsort()[-1:][::-1] 
        index_Right2 = finalProbsRight1.argsort()[-1:][::-1]
        if finalProbsLeft1[index_Left1] > finalProbsLeft2[index_Left2]:
            finalProbsLeft = finalProbsLeft1
        else:
            finalProbsLeft = finalProbsLeft2
        if finalProbsRight1[index_Right1] > finalProbsRight2[index_Right2]:
            finalProbsRight = finalProbsRight1
        else:
            finalProbsRight = finalProbsRight2
        index_Left = finalProbsLeft.argsort()[-2:][::-1] 
        index_Right = finalProbsRight.argsort()[-2:][::-1]
    #    print index
    #    print probs
        if finalProbsLeft[index_Left[0]]>finalProbsRight[index_Right[0]]:
            result['left'] = names[index_Left[0]]
            if names[index_Right[0]] == names[index_Left[0]]:
                result['right'] = names[index_Right[1]]
            else:
                result['right'] = names[index_Right[0]]
        else:
            result['right'] = names[index_Right[0]]
            if names[index_Right[0]] == names[index_Left[0]]:
                result['left'] = names[index_Left[1]]
            else:
                result['left'] = names[index_Left[0]]
        pic_set = []
        result = (result['left'],result['right'])
    else:
        instrument = (names[index[0]], names[index[1]])
        if position[index[0]] > position[index[1]]:
            result['left'] = instrument[1]
            result['right'] = instrument[0]
        else:
            result['left'] = instrument[0]
            result['right'] = instrument[1]
        pic_set = []
        result = (result['left'],result['right'])
    return result
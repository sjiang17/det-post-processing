import os
import xml.etree.ElementTree as ET
import cPickle
import numpy as np

##########################################

def isOverlaped(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])

    if (x1 > x2) or (y1 > y2):
        iArea = 0
    else:
        iArea = (x2 - x1) * (y2 - y1)
    uArea = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1]) + (bb2[2] - bb2[0]) * (bb2[3] - bb2[1]) - iArea
    # print "areas:", iArea, uArea, iArea/uArea
    if iArea/uArea >= 0.5:
        return True
    else:
        return False

labelMap_gt = {"n02691156": "airplane", "n02419796": "antelope", "n02131653": "bear", "n02834778": "bicycle",
            "n01503061": "bird", "n02924116": "bus", "n02958343": "car", "n02402425": "cattle",
            "n02084071": "dog", "n02121808": "domestic cat", "n02503517": "elephant", "n02118333": "fox",
            "n02510455": "giant panda", "n02342885": "hamster", "n02374451": "horse", "n02129165": "lion",
            "n01674464": "lizard", "n02484322": "monkey", "n03790512": "motorcycle", "n02324045": "rabbit",
            "n02509815": "red panda", "n02411705": "sheep", "n01726692": "snake", "n02355227": "squirrel",
            "n02129604": "tiger", "n04468005": "train", "n01662784": "turtle", "n04530566": "watercraft",
            "n02062744": "whale", "n02391049": "zebra"}


labelMap_det = {0: "background", 1: "airplane", 2: "antelope", 3: "bear", 4: "bicycle", 5: "bird", 6: "bus", 7: "car",
             8: "cattle", 9: "dog", 10: "domestic cat", 11: "elephant", 12: "fox", 13: "giant panda", 14: "hamster",
             15: "horse", 16: "lion", 17: "lizard", 18: "monkey", 19: "motorcycle", 20: "rabbit", 21: "red panda",
             22: "sheep", 23: "snake", 24: "squirrel", 25: "tiger", 26: "train", 27: "turtle", 28: "watercraft",
             29: "whale", 30: "zebra"}

############################################

# dir_gt = 'D:\\Data\\ILSVRC2015_VID\\GroundTruth\\val'
# dir_det = 'D:\\Data\\ILSVRC2015_VID\\Detection\\val'

dir_gt = 'D:\\Data\\ILSVRC2015_VID\\GroundTruth\\train\\ILSVRC2015_VID_train_0001'
dir_det = 'D:\\Data\\ILSVRC2015_VID\\Detection\\train\\ILSVRC2015_VID_train_0001'

recallCount = []

videoNum = -1
objCount = 0
tp = 0
for parentdir, childdir, fileNames in os.walk(dir_gt): #iterate through every video: ILSVRC2015_val_00000000, ILSVRC2015_val_00000001 ...
    if videoNum == -1:
        videoNum += 1
        continue
    else:
        print "video number:", videoNum
        cpkPrntDir = os.path.split(parentdir)[1]
        #load the cPickle file (detections) for every video
        cpkDir = os.path.join(dir_det, cpkPrntDir, 'det_conf_result.pkl')
        det = cPickle.load(open(cpkDir))
        #print cpkDir
        # print "totalFrame:", len(det)

        frameNum = 0
        for fileName in fileNames: #iterate through every xml(frame)
            # print "frame:", frameNum
            # print os.path.join(parentdir, fileName)

            xmlDir = os.path.join(parentdir, fileName)
            annot = ET.parse(xmlDir).getroot()

            frameNumRead = int(annot.find('filename').text)
            if frameNum != frameNumRead:
                print "frame number not equal!"
                exit(-1)

            size = [int(annot.find('size').find('width').text), int(annot.find('size').find('height').text)]
            # print size[0], size[1]
            for obj in annot.iter('object'): # iterate every ground truth bounding box

                label_gt = obj.find("name").text

                # print "obj:", objCount
                xmin = float(obj.find('bndbox').find('xmin').text)
                xmax = float(obj.find('bndbox').find('xmax').text)
                ymin = float(obj.find('bndbox').find('ymin').text)
                ymax = float(obj.find('bndbox').find('ymax').text)

                bb_gt = np.array([xmin, ymin, xmax, ymax])
                # print bb_gt
                objCount += 1

                for j in range(det[frameNum].shape[2]):  # iterate every detected bounding box
                    label_det = int(det[frameNum][0, 0, j, 1])

                    xmin_det = det[frameNum][0, 0, j, 3] * size[0]
                    ymin_det = det[frameNum][0, 0, j, 4] * size[1]
                    xmax_det = det[frameNum][0, 0, j, 5] * size[0]
                    ymax_det = det[frameNum][0, 0, j, 6] * size[1]
                    bb_det = np.array([xmin_det, ymin_det, xmax_det, ymax_det])
                    # print bb_det
                    # see if they overlap
                    jud = isOverlaped(bb_gt, bb_det)
                    # print "det bb:", j, jud
                    if jud:  # and (labelMap_det[label_det] == labelMap_gt[label_gt]):
                        tp += 1
                        break
            # print "tp:", tp
            # print " "
            frameNum += 1
        recallCount.append(np.array([objCount, tp, float(tp) / float(objCount)]))
        print recallCount[videoNum]

        videoNum += 1

fsave = open('recall_train1.pkl', 'w')
cPickle.dump(recallCount, fsave)

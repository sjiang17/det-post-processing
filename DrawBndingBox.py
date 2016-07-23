import os
import cv2
import cPickle
import numpy as np
import xml.etree.ElementTree as ET
import datetime

time_start = datetime.datetime.now()

labelMap_gt = {"n02691156": "airplane", "n02419796": "antelope", "n02131653": "bear", "n02834778": "bicycle",
               "n01503061": "bird", "n02924116": "bus", "n02958343": "car", "n02402425": "cattle",
               "n02084071": "dog", "n02121808": "domestic cat", "n02503517": "elephant", "n02118333": "fox",
               "n02510455": "giant panda", "n02342885": "hamster", "n02374451": "horse", "n02129165": "lion",
               "n01674464": "lizard", "n02484322": "monkey", "n03790512": "motorcycle", "n02324045": "rabbit",
               "n02509815": "red panda", "n02411705": "sheep", "n01726692": "snake", "n02355227": "squirrel",
               "n02129604": "tiger", "n04468005": "train", "n01662784": "turtle", "n04530566": "watercraft",
               "n02062744": "whale", "n02391049": "zebra"}

labelMap_det = {0: "bg", 1: "airplane", 2: "antelope", 3: "bear", 4: "bicycle", 5: "bird", 6: "bus", 7: "car",
                8: "cattle", 9: "dog", 10: "domestic cat", 11: "elephant", 12: "fox", 13: "giant panda", 14: "hamster",
                15: "horse", 16: "lion", 17: "lizard", 18: "monkey", 19: "motorcycle", 20: "rabbit", 21: "red panda",
                22: "sheep", 23: "snake", 24: "squirrel", 25: "tiger", 26: "train", 27: "turtle", 28: "watercraft",
                29: "whale", 30: "zebra"}

dir_gt = 'D:\\PythonProjects\\EvalDetection\\sample3\\val_gt'
dir_det = 'D:\\PythonProjects\\EvalDetection\\sample3\\val_det'
dir_vid = 'D:\\PythonProjects\\EvalDetection\\VID\\val'
dir_save = 'D:\\PythonProjects\\EvalDetection\\ResultImg\\val'

color = np.random.randint(0, 255, (100, 3))
textThick = 2

videoNum = -1
for parentdir, childdir, fileNames in os.walk(dir_gt): #iterate through every video: ILSVRC2015_val_00000000, ILSVRC2015_val_00000001 ...
    if videoNum == -1 or videoNum == 238:
        videoNum += 1
        continue
    else:
        print "video number:", videoNum

        cpkPrntDir = os.path.split(parentdir)[1] # ILSVRC2015_val_00000000
        #load the cPickle file (detections) for every video
        cpkDir = os.path.join(dir_det, cpkPrntDir, 'det_conf_result.pkl')
        det = cPickle.load(open(cpkDir))
        #print cpkDir
        print "totalFrame:", len(det)

        frameNum = 0
        for fileName in fileNames:  # iterate through every xml(frame)

            # print os.path.join(parentdir, fileName)

            xmlDir = os.path.join(parentdir, fileName)
            annot = ET.parse(xmlDir).getroot()
            frameNumRead = int(annot.find('filename').text)
            if frameNum != frameNumRead:
                print "frame number not equal!"
                exit(-1)

            size = [int(annot.find('size').find('width').text), int(annot.find('size').find('height').text)]

            imgDir = os.path.join(dir_vid, cpkPrntDir, fileName.split('xml')[0] + 'JPEG')
            imgSaveDir = os.path.join(dir_save, cpkPrntDir)

            if not os.path.exists(imgSaveDir):
                os.makedirs(imgSaveDir)

            curImg = cv2.imread(imgDir)
            # cv2.imshow('tt', curImg)

            for obj in annot.iter('object'):  # iterate every ground truth bounding box
                label_gt = obj.find("name").text
                xmin = int(obj.find('bndbox').find('xmin').text)
                xmax = int(obj.find('bndbox').find('xmax').text)
                ymin = int(obj.find('bndbox').find('ymin').text)
                ymax = int(obj.find('bndbox').find('ymax').text)
                #bb_gt = np.array([xmin, ymin, xmax, ymax])

                cv2.rectangle(curImg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)  # draw gt bounding box in green
                textSize = cv2.getTextSize(labelMap_gt[label_gt], cv2.FONT_HERSHEY_SIMPLEX, 1, textThick)
                textOrg = (xmin, ymin + textSize[0][1])
                cv2.putText(curImg, " " + labelMap_gt[label_gt], textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), textThick)

            for j in range(det[frameNum].shape[2]):  # iterate every detected bounding box
                if det[frameNum][0, 0, j, 2] >= 0.3:  # only draw bounding boxes with a confidence >= threshold
                    label_det = int(det[frameNum][0, 0, j, 1])
                    xmin_det = int(det[frameNum][0, 0, j, 3] * size[0])
                    ymin_det = int(det[frameNum][0, 0, j, 4] * size[1])
                    xmax_det = int(det[frameNum][0, 0, j, 5] * size[0])
                    ymax_det = int(det[frameNum][0, 0, j, 6] * size[1])
                    #bb_det = np.array([xmin_det, ymin_det, xmax_det, ymax_det])

                    if (xmin_det != xmax_det) and (ymin_det != ymax_det):  # bbs with size of 0 are not drawn

                        cv2.rectangle(curImg, (xmin_det, ymin_det), (xmax_det, ymax_det), color[int(det[frameNum][0, 0, j, 1])], 3)
                        textSize = cv2.getTextSize(labelMap_det[label_det], cv2.FONT_HERSHEY_SIMPLEX, 1, textThick)
                        textOrg = (xmin_det, ymin_det + textSize[0][1])
                        textOrg2 = (xmin_det, ymin_det + 2 * textSize[0][1])
                        cv2.putText(curImg, " " + labelMap_det[label_det], textOrg,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color[int(det[frameNum][0, 0, j, 1])], textThick)
                        cv2.putText(curImg, " " + str(round(det[frameNum][0, 0, j, 2], 2)), textOrg2,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color[int(det[frameNum][0, 0, j, 1])], textThick)

            cv2.imwrite(str(imgSaveDir) + '\\' + str(frameNum) + '.JPEG', curImg)
            frameNum += 1

        videoNum += 1

print time_start
print datetime.datetime.now()
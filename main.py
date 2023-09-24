import cv2
import numpy as np
import os
import frameextractor as FrameExtractor
import handshape_feature_extractor as hfe
from scipy.spatial.distance import cosine

outputCsv = []
gestureDict = {'Num0': 0,
               'Num1': 1,
               'Num2': 2,
               'Num3': 3,
               'Num4': 4,
               'Num5': 5,
               'Num6': 6,
               'Num7': 7,
               'Num8': 8,
               'Num9': 9,
               'FanDown': 10,
               'FanOn': 11,
               'FanOff': 12,
               'FanUp': 13,
               'LightOff': 14,
               'LightOn': 15,
               'SetThermo': 16}

# Path variables
root = os.path.dirname(os.path.abspath(__file__))
testDirectory = root + "/" + "test"
trainDirectory = root + "/" + "traindata"
testFrameDirectory = testDirectory + "/frames"
trainFrameDirectory = trainDirectory + "/frames"
result = root + "/" + "Results.csv"


def fileNameList(path):
    filelist = [file for file in os.listdir(
        path) if os.path.isfile(os.path.join(path, file))]
    filelist.sort()
    return filelist


def extractFeatures(videoDir, framesDir):

    videolist = fileNameList(videoDir)
    features = []
    label = {}

    count = 0
    for file in videolist:
        if file.startswith('.'):
            continue
        FrameExtractor.frameExtractor(videoDir + "/" + file, framesDir, count)
        label[count+1] = file
        count += 1

    imglist = fileNameList(framesDir)
    for img in imglist:
        if img.startswith('.'):
            continue
        image = cv2.imread(framesDir + "/" + img, cv2.IMREAD_UNCHANGED)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        file = hfe.HandShapeFeatureExtractor.get_instance()
        featureVector = file.extract_feature(gray)
        features.append([img, featureVector])

    return features, label


trainX, trainlabel = extractFeatures(trainDirectory, trainFrameDirectory)
testX, testlabel = extractFeatures(testDirectory, testFrameDirectory)


for img in testX:

    currentFrame = int(img[0].split(".")[0])
    videoFrame = testlabel[currentFrame].split(".")[0]

    # Calculate loss
    loss = 0
    trainframe = 0
    testImg = img[1][0]

    for i in trainX:
        trainImg = i[1][0]
        similarity = 1-cosine(testImg, trainImg)

        if similarity > loss:
            loss = similarity
            trainframe = int(i[0].split(".")[0])

    gestureName = trainlabel[trainframe].split("_")[0]

    outputCsv.append(gestureDict[gestureName])

# Output to csv as a 51 x 1 matrix
np.savetxt(result, outputCsv, fmt='%i', delimiter=',')

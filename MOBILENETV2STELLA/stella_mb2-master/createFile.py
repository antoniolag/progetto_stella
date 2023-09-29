import os
import random
import sys
import cv2




allFiles = os.listdir("stella_dataset/images/")
print(allFiles)
size = len(allFiles)
testNum = int(size * 0.10)
valNum = int(size * 0.20)
testFileLst = []
valFileLst = []
while True:
    ap = random.choice(allFiles)
    if ap not in testFileLst and not len(testFileLst) == testNum :
        testFileLst.append(ap)
    if (len(testFileLst) == testNum):
            if ap not in valFileLst:
                valFileLst.append(ap)
                if len(valFileLst) == valNum:
                    break
print("finito val")
trainFileLst = list(set(testFileLst).union(set(valFileLst)).symmetric_difference(set(allFiles)))


f = open("stella_dataset/val.txt", 'w+')
for test in valFileLst:
    test = test.replace(".jpg","")
    f.write(test + "\n")
f.close()

f = open("stella_dataset/train.txt", 'w+')
for train in trainFileLst:
    train = train.replace(".jpg","")
    f.write(train + "\n")
f.close()

f = open("stella_dataset/test.txt", 'w+')
for train in testFileLst:
    train = train.replace(".jpg","")
    f.write(train + "\n")
f.close()






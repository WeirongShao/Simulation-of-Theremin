# coding=utf-8

import ctypes
import time
import re
import threading
from threading import Thread

from ctypes import windll
# import winsound

import cv2
import numpy as np
import copy
import math
import win32api
import win32con
import wave
import array
import pyaudio

print('loading param...')
# pyaudio
freq=450.0
data_size=2*10**20
frate = 8000.0
amp = 8000.0
target_freq = freq
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2), channels=2, rate=int(frate), output=True)
MAX_FREQ = 20000
MIN_FREQ = 450
print('pyaudio done!')

WM_APPCOMMAND = 0x319
APPCOMMAND_VOLUME_UP = 0x0a
APPCOMMAND_VOLUME_DOWN = 0x09
APPCOMMAND_VOLUME_MUTE = 0x08
hwnd=windll.user32.GetForegroundWindow()

player = ctypes.windll.kernel32

# beep蜂鸣器范围
MAX_FREQUENCY = 32767
MIN_FREQUENCY = 37

frequency = 100  #

# 静音
# windll.user32.PostMessageA(hwnd, WM_APPCOMMAND, 0, APPCOMMAND_VOLUME_MUTE*0x10000)
# windll.user32.PostMessageA(hwnd, WM_APPCOMMAND, 0, APPCOMMAND_VOLUME_DOWN*0x10000)


# notes = [0,523,587,659,698,784,880,988]

q = 1.06  # 每阶音的倍数
q2 = q * q
# 音阶
dolist = {'C': 523, 'D': 587, 'E': 659, 'F': 698, 'G': 784, 'A': 880, 'B': 988}
# 低中高 音
pitchs = {'l': 0.5, 'm': 1, 'h': 2}
# 每个音符有三个要素:音符、音调、节拍，
# 3m0.5 : 3表示mi，m表示中音，0.5表示一个八分音符


def BBPlayer(filename, dokey, speed):
    '''
    beep播放音乐
    :param filename: 
    :param dokey: 
    :param speed: 
    :return: 
    '''
    do = int(dolist[dokey])  # 获取对应调的do的频率
    re = int(do * q2)
    mi = int(re * q2)
    fa = int(mi * q)
    sol = int(fa * q2)
    la = int(sol * q2)
    si = int(la * q2)
    notes = [0, do, re, mi, fa, sol, la, si]
    beats = 60 / speed * 1000

    # print(beats)
    with open(filename) as fp:
        song = fp.read().replace('\n', '').split(',')
        # print(type(song))
        # print(song)

        for music in song:
            # print(type(music))
            # p = re.findall(r'[lmh]',music)[0] # 获取music中的字母
            p = music[1]
            p = float(pitchs[p])  # 高低音
            # music = re.split(r'[lmh]',music,maxsplit = 0,flags = 0)
            n = int(notes[int(music[0])])  # 音符
            b = float(music[2:])  # 节拍
            # print(music[0])
            print(n)
            # print(p)
            # print(music[2:])

            print('延迟：', b*beats, '节拍:', b, '频率：', n * p)
            if n == 0:
                time.sleep(b * beats / 1000)
                print('\t', b * beats / 1000)
            else:
                player.Beep(int(n * p), int(b * beats))


####################
# my beep sound
####################

def myBeepSound(sleep_time, frequency, volume):
    # frequency范围：37 - 32767
    winsound.Beep(frequency, 500) # frequency, duration
    time.sleep(sleep_time / 100)


def upVolume():
    windll.user32.PostMessageA(hwnd, WM_APPCOMMAND, 0, APPCOMMAND_VOLUME_UP * 0x10000)


def downVolume():
    windll.user32.PostMessageA(hwnd, WM_APPCOMMAND, 0, APPCOMMAND_VOLUME_DOWN * 0x10000)

def upVolumeN(n):
    for i in range(n):
        upVolume()

def downVolumnN(n):
    for i in range(n):
        downVolume()

def updateVolume(rate):
    if rate > 0.5:
        upVolumeN(int(100 * (rate - 0.5)))
    else:
        downVolumnN(int(100 * (0.5 - rate)))

def updateFrequency(frequency, rate):
    frequency = (MAX_FREQUENCY - MIN_FREQUENCY) * 10.0 * rate // 100
    return int(frequency) if frequency > 100 else 100

def updateFreq(freq, rate):
    freq = (MAX_FREQ - MIN_FREQ) * 100.0 * rate // 100 + MIN_FREQ
    return MAX_FREQ if freq > MAX_FREQ else int(freq)


def upFrequency(frequency):
    frequency += 10
    if frequency > MAX_FREQUENCY:
        frequency = MAX_FREQUENCY
    if frequency < MIN_FREQUENCY:
        frequency = MIN_FREQUENCY
    return frequency


def downFrequency(frequency):
    frequency -= 10
    if frequency < MIN_FREQUENCY:
        frequency = MIN_FREQUENCY
    if frequency > MAX_FREQUENCY:
        frequency = MAX_FREQUENCY
    return frequency


######################################################################
# 摄像头手势识别
######################################################################

# 参数
DEBUG = True  # 调试界面
img_x_size = 0.5  # 起点/总宽度
img_y_size = 0.8
threshold = 60  # 二值化阈值
blurValue = 41  # 高斯模糊参数
bgSubThreshold = 50
learningRate = 0

LowBound = 40  # 最低的计数限额
LowDist = 20  # 最低的距离限额

# 变量
isBgCaptured = 0  # 布尔类型, 背景是否被捕获


def printThreshold(thr):
    print("! Changed threshold to " + str(thr))


def removeBG(frame): #移除背景
    fgmask = bgModel.apply(frame, learningRate=learningRate) #计算前景掩膜
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1) #使用特定的结构元素来侵蚀图像。
    res = cv2.bitwise_and(frame, frame, mask=fgmask) #使用掩膜移除静态背景
    return res

def calculateFingers(res, drawing, DEBUG):  # -> finished bool, fingerNums: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)

    if len(hull) > 3:
        # convexhull为外边多边形，凸包
        # convexity defects是多边形与图轮廓之间的空白部分，比如两指之间的空隙
        defects = cv2.convexityDefects(res, hull)
        if type(defects) == type(None):  # avoid crashing.   (BUG not found)
            return False, 0
        fingerNums = 0
        for i in range(defects.shape[0]):  # calculate the angle
            # 返回：startPoint,endPoint, 距离 hull 的最远点farPoint，最远点距离depth

            s, e, f, d = defects[i][0]
            start = tuple(res[s][0])
            end = tuple(res[e][0])
            far = tuple(res[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            # 计算defects的夹角，也就是两指间隙的夹角
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                fingerNums += 1
                if DEBUG:
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
        return True, fingerNums
    return False, 0

def getCenterOfFinger(res, drawing, DEBUG=False):
    # moments用于计算物体质心、物体面积等
    moments = cv2.moments(res)  # 求最大区域轮廓的各阶矩
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    if DEBUG:
        cv2.circle(drawing, center, 8, (0, 0, 255), -1)  # 画出重心
    return center


def getMaxContours(contours):
    maxArea = max([cv2.contourArea(t) for t in contours])
    for i in range(len(contours)):
        if maxArea == cv2.contourArea(contours[i]):
            index = i
    return contours[index], maxArea

def myCalculateCenterFingers(res, drawing, DEBUG=False):

    # get center of finger
    center = getCenterOfFinger(res, drawing, DEBUG)

    finger_top_list = []  # 寻找指尖
    maxDist = 0
    count = 0
    fingerNums = 0

    for item in res:
        temp = item[0]
        dist = (temp[0] - center[0]) ** 2 + (temp[1] - center[1]) ** 2
        if dist > maxDist:
            maxDist = dist
            # 指尖候选
            finger_top = item[0]
        if dist < maxDist:
            count += 1
            if count > LowBound:
                count = 0
                maxDist = 0
                flag = False
                temp_list = [LowDist - abs(finger_top[0] - t[0]) for t in finger_top_list]
                for i in temp_list:
                    if i>0: flag = True

                if flag or center[1] < finger_top[1]:
                    continue
                # 保存指尖
                finger_top_list.append(finger_top)
                if DEBUG:
                    cv2.circle(drawing, tuple(finger_top), 8, (255, 0, 0), -1)  # 画出指尖
                    cv2.line(drawing, center, tuple(finger_top), (255, 0, 0), 2)
                fingerNums += 1
    return center, fingerNums

# 相机/摄像头
camera = cv2.VideoCapture(0)   #打开电脑自带摄像头，如果参数是1会打开外接摄像头
camera.set(10, 200)   #设置视频属性
cv2.namedWindow('trackbar') #设置窗口名字
cv2.resizeWindow("trackbar", 640, 200)  #重新设置窗口尺寸
cv2.createTrackbar('threshold', 'trackbar', threshold, 100, printThreshold)

print("running ... ")

# 取消按键触发，直接上电运行
bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
def startCV2():
    bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    isBgCaptured = 1
# time.sleep(3)

def startBeep(args):
    for x in range(data_size):
        # 播放beep音乐  
        data = array.array('h')

        # freq = updateFreq(freq, maxArea / (x_frame * y_frame))

        # if x % 10 == 0:
            # print('the x:', x, ' freq:', freq)
        print('the x:', x, ' freq:', freq)
        v = int(math.sin(2*math.pi*(freq)*(x/frate))*amp/2)
        data.append(v)
        data.append(v)
        stream.write(data.tostring()) 

t =threading.Thread(target=startBeep,args=(1,))
t.start()

x = 0
while camera.isOpened() and x < data_size:
# while x < data_size:


    # # 播放beep音乐  
    # data = array.array('h')

    # # freq = updateFreq(freq, maxArea / (x_frame * y_frame))

    # # if x % 10 == 0:
    #     # print('the x:', x, ' freq:', freq)
    # print('the x:', x, ' freq:', freq)
    # v = int(math.sin(2*math.pi*(freq)*(x/frate))*amp/2)
    # data.append(v)
    # data.append(v)
    # stream.write(data.tostring())  

    
    # beep 一直播放！
    # myBeepSound(0, frequency, 10)

    ret, frame = camera.read()
    threshold = cv2.getTrackbarPos('threshold', 'trackbar')
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # 双边滤波
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (int(img_x_size * frame.shape[1]), 0),
                  (frame.shape[1], int(img_y_size * frame.shape[0])),
                  (0, 0, 255), 2)

    # if DEBUG:
    cv2.imshow('original', frame)


    # if not isBgCaptured:
    #     time.sleep(3)
    #     bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
    #     isBgCaptured = 1
        # startCV2()

    #主要操作
    if isBgCaptured:  # 捕获背景
        img = removeBG(frame)
        img = img[0:int(img_y_size * frame.shape[0]), int(img_x_size * frame.shape[1]):frame.shape[1]]

        if DEBUG:
            cv2.imshow('mask', img)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        if DEBUG:
            cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
        if DEBUG:
            cv2.imshow('binary', thresh)

        # get the coutours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 轮廓特征：轮廓面积、周长、质心、边界框等

        if len(contours) > 0:

            res, maxArea = getMaxContours(contours)

            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            if DEBUG:
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            fingerNums1, fingerNums2 = 5, 5

            # 方法一：角度
            # isFinishCal, fingerNums1 = calculateFingers(res, drawing, DEBUG)

            # 方法二：距离
            center, fingerNums2 = myCalculateCenterFingers(res, drawing, DEBUG)

            fingerNums = min(fingerNums1, fingerNums2)
            if DEBUG:
                print('fingerNums1:', fingerNums1, ' fingerNums2:', fingerNums2, ' fingerNums:', fingerNums)


            # 中心点位置
            x_center, y_center = center[0], center[1]
            x_frame, y_frame = frame.shape[1] * img_x_size, frame.shape[0] * img_y_size  # 320，384
            print(fingerNums, 'center:', center, 'maxArea:', maxArea, maxArea / (x_frame * y_frame), frequency)

            # 处理逻辑
            if x_center > x_frame / 2:  # 往右移动
                upVolume()
            elif x_center < x_frame / 2:  # 往左移动
                downVolume()

            # 频率调节
            #frequency = updateFrequency(frequency, maxArea / (x_frame * y_frame))

            # 播放beep音乐  
            # data = array.array('h')

            freq = updateFreq(freq, maxArea / (x_frame * y_frame))

            # v = int(math.sin(2*math.pi*(freq)*(x/frate))*amp/2)
            # data.append(v)
            # data.append(v)
            # stream.write(data.tostring())


            # if maxArea / (x_frame * y_frame) > 0.5:  # 靠近手掌
            #     frequency = upFrequency(frequency)
            # else:                                    # 远离手掌
            #     frequency = downFrequency(frequency)

            cv2.imshow('output', drawing)

    x += 1  


    # 输入的键盘值
    k = cv2.waitKey(1)
    if k == 27:  # 按下ESC退出
        break
    elif k == ord('b'):  # 按下'b'会捕获背景
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        isBgCaptured = 1
        if DEBUG:
            print('!!!Background Captured!!!')
    elif k == ord('r'):  # 按下'r'会重置背景
        bgModel = None
        isBgCaptured = 0
        if DEBUG:
            print('!!!Reset BackGround!!!')

    

stream.stop_stream()
stream.close()

p.terminate()

# a, b = input().split(' ')
# a, b = 50, 1
# for i in range(532, 2500, 50):
# for i in [500, 1000, 1200, 1500, 1800, 2100, 2500, ]:
# while True:
    # if int(b):
    #     upVolume()
    # else:
    #     downVolume()
    #
    # myBeepSound(0, frequency, 10)



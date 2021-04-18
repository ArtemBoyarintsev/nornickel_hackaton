from collections import defaultdict
from PIL import Image

import os
import sys
import time
import json
import random
import cv2 as cv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


import nn_tools

video_file = './F2_1_2_2.ts'
clear = lambda: os.system('clear')


def get_centers(contours):
  centers = []
  for c in contours:
      M = cv.moments(c)
      if M["m00"] <= 1e-6:
          continue
      
      cX = int(M["m10"] / M["m00"])
      cY = int(M["m01"] / M["m00"])

      area = cv.contourArea(c)
      if area < 5000:
        centers.append((cX,cY,area))
  
  return centers



def calculate_square(seg_map, contours, div_coef=10):
  '''
  Calculates square in pixels. The main idea behind the method.
  Every flare belongs to a bubble. So let's take a segmentation map got from the neural network and find 
  for every pixel on it the closest flare. It makes sense as pixel on segmentation map is a bubble. So 
  the bubble square is amount of pixels for which the flare is the closest one

  Finding the closest flare for every pixels can be complicated calculation task. So let's reduce the size
  of a picture to calculate approximate square (which is still hasn't inaccuracy more then 5%)
  '''
  
  seg_map_smalled = np.array(Image.fromarray((seg_map*255).astype('uint8')).resize((80,60)))
  ones = np.where(seg_map_smalled)

  bubble2square = defaultdict(lambda :0)  # square of every bubble

  centers = get_centers(contours)
  for i in range(0, len(ones[0])):
      x = ones[0][i]*div_coef
      y = ones[1][i]*div_coef
      
      min_distance=10000
      c_min = None
      for c in centers:
          area = c[2]
          distance = np.sqrt((c[1]-x)**2 + (c[0]-y)**2)/(area**0.33)
          if distance < min_distance:
              min_distance = distance
              c_min = c
          
      bubble2square[c_min] += div_coef*div_coef
  
  #for c in bubble2square:
  #  if bubble2square[c] > 12000:
   #   bubble2square[c] = 0
  return bubble2square


def bubble_analyze(img, bubble2square):
  
  squares = [x[1] for x in bubble2square.items()]
  mean, std = np.mean(squares), np.std(squares)
  summa, cnt = np.sum(squares), len(squares)

  cv.putText(img, f"Mean = {mean:.3f}, std={std:.3f}", (10, 10), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.40, color=(240, 255, 200), thickness=1, lineType=cv.LINE_AA)
  cv.putText(img, f"Summa={summa:.3f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.40, color=(240, 255, 200), thickness=1, lineType=cv.LINE_AA)
  cv.putText(img, f"Count={cnt}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.40, color=(240, 255, 200), thickness=1, lineType=cv.LINE_AA)
  #plt.hist(squares)
  #plt.show()
  

def main():
  cap = cv.VideoCapture(video_file)

  fourcc = cv.VideoWriter_fourcc(*"mp4v")
  
  net = nn_tools.get_network() 

  frame = 0

  while(True):

    _, img = cap.read()

    if img is None:
      break
      
    #h, w, d = img.shape
    
    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    _, thresh = cv.threshold(hsv_img[:,:,2], 150, 255, cv.THRESH_BINARY_INV)  # Find flare

    kernel = np.ones((3,3))
    thresh = cv.dilate(thresh,kernel)  # Remove some noise 
    
    seg_map = nn_tools.pass_img(net, img)   # segmentation map
    thresh = (seg_map * thresh).astype('uint8')
    
    contours, _ = cv.findContours(thresh , cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE) 
    bubble2square = calculate_square(seg_map, contours)
    bubble_analyze(img,bubble2square)
    img_with_annotations = img/255*0.3 + img/255*seg_map[...,None]*0.7

    for bubble in bubble2square:
      square = bubble2square[bubble]
      x = bubble[0]
      y = bubble[1]

      cv.putText(img, f"{square}", (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.40, color=(240, 115, 200), thickness=1, lineType=cv.LINE_AA)
      cv.putText(img_with_annotations, f"{square}", (x, y), cv.FONT_HERSHEY_SIMPLEX, fontScale=0.40, color=(240, 115, 200), thickness=1, lineType=cv.LINE_AA)
    
    cv.imshow("Movie", img)
    cv.imshow("Movie2", img_with_annotations)


    try:
      button_press = cv.waitKey(10)

      # Нажатие клавиши [q] или [ESC] - завершение работы приложения
      if (button_press & 0xFF == 27) or (button_press & 0xFF == ord('q')) or (button_press & 0xFF == ord('Q')):
        print('q & Q')
        break

      # Нажатие клавиши [p] или [P] - передаем данные на обработку нейросети
      if (button_press & 0xFF == ord('p')) or (button_press & 0xFF == ord('P')):
        print('p & P')

      # Нажатие клавиши [r] или [R] - передаем данные на обработку нейросети
      if (button_press & 0xFF == ord('r')) or (button_press & 0xFF == ord('R')):
        print('r & R')

    except:
      print("Dummy for error's hook")

    frame = frame + 1
    
  cv.destroyAllWindows()
  cap.release()
  

if __name__ == '__main__':
  clear()
  main()

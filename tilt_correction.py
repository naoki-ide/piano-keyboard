#!/usr/bin/env python

import argparse

import cv2
import moviepy.editor as mp
import numpy as np
import time
import subprocess
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import embed


def transform(x, center, angle, scale):
    trans = cv2.getRotationMatrix2D(tuple(center), angle , scale)
    height = x.shape[0]                         
    width = x.shape[1]  
    y = cv2.warpAffine(x, trans, (width, height))
    return y

def ctrl_angle(x):
    c = 540
    s = np.mean(x[c-400:c+50,:480]) - np.mean(x[c-400:c+50,-480:])
    return s / 255

def ctrl_height(x, threshold):
    a = 300
    b = np.r_[x.T[:a],x.T[-a:]]
    return np.mean(np.where(b.T > threshold)[0])


def get_args():
    parser = argparse.ArgumentParser(description="quantized-sparse-modeling-evaluator")

    parser.add_argument("--input-filename", "-i", type=str)

    parser.add_argument("--output-filename", "-o", type=str)

    parser.add_argument("--zooming-up-scale", "-s", type=float, default="2")

    parser.add_argument("--horizontal-center", "-hc", type=int, default="960")

    parser.add_argument("--vertical-center", "-v", type=int, default="540")

    parser.add_argument("--initial-rotation-angular", "-a", type=float, default="0")

    parser.add_argument("--threshold-for-keyboard", "-t", type=int, default="100")

    parser.add_argument("--feedback-coefficient-for-rotation", "-r", type=float, default="20")

    parser.add_argument("--feedback-coefficient-for-vertical-shift", "-c", type=float, default="0.1")

    parser.add_argument("--audio-disable", action='store_true')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    print("Start Tilt Correction")
    args = get_args()

    input_file = args.input_filename
    output_file = args.output_filename

    scale = args.zooming_up_scale
    initial_angle = args.initial_rotation_angular
    threshold = args.threshold_for_keyboard

    vertical_center = args.vertical_center
    horizontal_center = args.horizontal_center

    fc_r = args.feedback_coefficient_for_rotation
    fc_v = args.feedback_coefficient_for_vertical_shift

    audio_disable = args.audio_disable

    print("- input file: %s" % input_file)
    print("- output file: %s" % output_file)
    print("- zoom up scale: %f" % scale)
    print("- initial angle: %f" % initial_angle)
    print("- vertical center: %d" % vertical_center)
    print("- horizontal center: %d" % horizontal_center)
    print("- audio disable: %s" % audio_disable)

    cap = cv2.VideoCapture(input_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("\nanalyze %s" % input_file)
    print("- frame per second: %f" % fps)
    print("- number of frames: %d" % num_frame)

    width = 1920
    height = 1080
    size = (width, height)
    
    fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    color_writer = cv2.VideoWriter(output_file, fmt, fps, size)
    #gray_writer = cv2.VideoWriter(output_file[:-4] + "_b.mp4", fmt, fps, size, 0)
    
    print("\nstart tilt correction.")
    bar = tqdm(total = num_frame)

    angle = initial_angle
    bias = 540 #vertical_center
    angles = []
    biases = []
    
    t0 = time.time()
    for i in range(num_frame):
        t1 = time.time() 
        _, frame = cap.read()
        x = np.zeros(frame.shape, dtype=np.uint8)
        a = max(0, vertical_center - 540)
        b = min(1080, vertical_center + 540)
        c = max(0, horizontal_center - 960)
        d = min(1920, horizontal_center + 960)
        x[0:b-a,0:d-c,:] = frame[a:b,c:d,:]
        
        t2 = time.time() 
        center = [960, bias]
        x = transform(x, center, angle, scale)
    
        t3 = time.time() 
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        _, b = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        
        t4 = time.time()     
        angle += 20 * ctrl_angle(b)
        bias += int(0.2 * ((ctrl_height(b, threshold) - bias) / scale))
        if angle < -45: angle = -45
        if 45 < angle : angle =  45
        if bias < 100: bias = 100
        if 980 < bias: bias = 980
    
        angles.append(angle)
        biases.append(bias - 540)
        
        t5 = time.time() 
        color_writer.write(x)
        
        t6 = time.time() 
        #gray_writer.write(b)
        
        t7 = time.time() 
        
        bar.update(1)
        #print("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % (t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6))
    cap.release()    
    color_writer.release()
    #gray_writer.release()

    #plt.plot(angles)
    #plt.savefig("_ra.png")
    #plt.plot(biases)
    #plt.savefig("_vc.png")
    #plt.imshow(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)) #cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
 
    if not audio_disable:

        print("\nstart audio extraction.")

        subprocess.call("mv " + output_file + " _tmp.mp4", shell=False)

        audio_clip = mp.VideoFileClip(input_file).subclip()
        audio_clip.audio.write_audiofile("_tmp.mp3", verbose=False, logger=None)

        print("\nmerge audio and vision.")
        
        clip = mp.VideoFileClip("_tmp.mp4").subclip()
        clip.write_videofile(output_file, fps=audio_clip.fps, audio="_tmp.mp3", verbose=False, logger=None)

        subprocess.call("rm _tmp.mp3", shell=False)
        subprocess.call("rm _tmp.mp4", shell=False)

    print("Total process time %.4f [sec]" % (time.time() - t0))    

    exit()
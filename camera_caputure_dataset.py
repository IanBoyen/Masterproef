import numpy as np
import pandas as pd
import cv2
import sys
import os
import keyboard
from vmbpy import *
import time

#from typing import Optional
#from queue import Queue
#import serial

opencv_display_format = PixelFormat.Bgr8 # All frames will either be recorded in this format, or transformed to it before being displayed
imgs_path = "C:/Users/ianbo/Documents/school/1E-ICT/1Ma/Masterproef/Code/dataset"

#Image counter
def img_counter(count: int = 0) -> int:
    return count + 1

#Abort program
def abort(reason: str, return_code: int = 1, usage: bool = False):
    print(reason + '\n')
    sys.exit(return_code)
        
#Get available camera
def get_camera() -> Camera:
    with VmbSystem.get_instance() as vmb:
        cams = vmb.get_all_cameras()
        if not cams:
            abort('No Cameras accessible. Abort.')
        print(f'Cameras found: {format(len(cams))}')
        print_camera(cams[0])
        return cams[0]

#Printing camera identification   
def print_camera(cam: Camera):
    print('   Camera Name   : {}'.format(cam.get_name()))
    print('   Model Name    : {}'.format(cam.get_model()))
    print('   Camera ID     : {}'.format(cam.get_id()))
    print('   Serial Number : {}'.format(cam.get_serial()))
    print('   Interface ID  : {}\n'.format(cam.get_interface_id()))

#Set variables for for camera
def setup_camera(cam: Camera, exposureTime: float = 30000, gain: float = 11):
    with cam:
        print("Setting up the camera...")
        try:
            cam.ExposureAuto.set('Off')
            cam.ExposureTime.set(exposureTime)
            cam.Gain.set(gain)
            print("Succesfull!\n")
            
        except Exception as e:
            abort(f"Error setting up camera: {e}")
        
#Capture a frame  
def capture(cam: Camera, width: int = 960, height: int = 640) -> Frame:
    try:
        print("Capturing frame...")
        frame = cam.get_frame()
        frame = frame.convert_pixel_format(opencv_display_format)
        frame = frame.as_opencv_image()
        frame = cv2.resize(frame, (width, height))
        print("Frame captured\n")
        return frame
    except Exception as e:
        abort(f"Error csapturing frame: {e}")        

#Display image
def show_img(frame: Frame, window_name: str = "Image") -> Frame:
    cv2.imshow(window_name, frame)
    return frame

#Save Image
def save_img(frame: Frame, img_name: str="image", dir_path: str=imgs_path):
    if frame is not None:
        img_path = os.path.join(dir_path, img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img_num = img_counter()
        while True:
            img_path = os.path.join(dir_path, img_name+f"_{img_num:04d}.jpg")
            if not os.path.exists(img_path):
                break  # Als de img_name nog niet bestaat, stoppen en img saven volgens dit pad
            img_num = img_counter(img_num) #Anders img_num verhogen

        print(f"saving Image to: {img_path}")
        cv2.imwrite(img_path, frame)
        print("Imaged saved\n")
    else:
        print("Empty frame (geen afbeelding om op te slaan)\n")

#MAIN
def main():        
    with VmbSystem.get_instance():
        with get_camera() as cam:
            window_name = "Image"
            setup_camera(cam,50000,6)     
            frame = None
            while True:
                if keyboard.is_pressed('q'):  # Sluit programma
                    print("Closing Program")
                    cv2.destroyAllWindows()  # Zorg ervoor dat alle vensters netjes worden gesloten
                    sys.exit(0)
                elif keyboard.is_pressed('*'):  # Capture afbeelding en toon
                    frame = capture(cam)
                    print("Displaying image...")
                    frame=show_img(frame)
                    time.sleep(0.3)
                elif keyboard.is_pressed('+'):  # Sla afbeelding op
                    wait_bit = 1
                    img_name = ""
                    print("Select number [0-9] to save image to correct path...")
                    time.sleep(0.1)
                    while wait_bit:
                        if keyboard.is_pressed('q'):  # Sluit programma
                            print("Closing Program")
                            cv2.destroyAllWindows()  # Zorg ervoor dat alle vensters netjes worden gesloten
                            sys.exit(0)
                        elif keyboard.is_pressed('1'):
                            img_name = "RM121263NE-BB"
                            wait_bit = 0
                        elif keyboard.is_pressed('2'):
                            img_name = "RM090955NE-AB"
                            wait_bit = 0
                        elif keyboard.is_pressed('3'):
                            img_name = "RM090955NE-AC"
                            wait_bit = 0
                        elif keyboard.is_pressed('4'):
                            img_name = "RM121279NE-CV"
                            wait_bit = 0
                        elif keyboard.is_pressed('5'):
                            img_name = "RM121279NE-DF"
                            wait_bit = 0
                        elif keyboard.is_pressed('6'):
                            img_name = "RM121279NE-CU"
                            wait_bit = 0
                        elif keyboard.is_pressed('7'):
                            img_name = "LNE443-179"
                            wait_bit = 0
                        elif keyboard.is_pressed('8'):
                            img_name = "SNC-44-60KH04"
                            
                            wait_bit = 0
                        elif keyboard.is_pressed('9'):
                            img_name = "LNE444-161"
                            wait_bit = 0
                        elif keyboard.is_pressed('0'):
                            img_name = "SNC-44-170"
                            wait_bit = 0
                          
                    save_img(frame, img_name, imgs_path+"/"+img_name)
                    print("Saving Image...")

                    frame = None
                    cv2.destroyAllWindows()
                    time.sleep(0.5)

                # Houd venster actief zonder blokkeren
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    cv2.destroyAllWindows()
                    frame = None
                elif frame is not None:
                    frame = show_img(frame)
                    cv2.waitKey(1)  # Houd het venster actief
                

if __name__ == '__main__':
    main()






        

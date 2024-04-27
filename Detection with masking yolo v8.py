import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import os
import serial
# from serial import Serial
import time
from bitstring import Bits
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageWin,ImageDraw,ImageTk,ImageFont
import threading
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from pymodbus.client.sync import ModbusSerialClient as ModbusClient  
import torch
# Pymodbus version should be 2.5.2
from datetime import datetime

model=YOLO('best_helmet_epron_gloves.pt')
# model=YOLO(r'best_v7.pt')
person_model=YOLO(r"yolov8n.pt")
cam_release=False

def Plc_connection():

    try:  
        client = ModbusClient(method='rtu', stopbits=1, bytesize=8, parity="E", port='COM4', baudrate=9600, timeout=3)
        connection = client.connect()
        if connection==True:
            plcstatus_lbl = tk.Label(forFrame, text="PLC Connected", fg="#00aa00", bg="#ffffff")
            plcstatus_lbl.configure(font=("Times New Roman", 11,'bold'))
            plcstatus_lbl.grid(row=3, column=1, padx=2, pady=1)
        else:
            plcstatus_lbl = tk.Label(forFrame, text="PLC Not Connected", fg="#aa0000", bg="#ffffff")
            plcstatus_lbl.configure(font=("Times New Roman", 11,'bold'))
            plcstatus_lbl.grid(row=3, column=1, padx=2, pady=1)

    except Exception as Ex:
        plcstatus_lbl = tk.Label(forFrame, text=Ex, fg="#aa0000", bg="#ffffff")
        plcstatus_lbl.configure(font=("Times New Roman", 11,'bold'))
        plcstatus_lbl.grid(row=3, column=1, padx=2, pady=1)

def Camera_connection():
    try:
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if (cap.isOpened):
            Camerastatus_lbl = tk.Label(forFrame, text=" Camera_LH_Connected", fg="#00aa00", bg="#ffffff")
            Camerastatus_lbl.configure(font=("Times New Roman", 11,'bold'))
            Camerastatus_lbl.grid(row=4, column=1, padx=2, pady=1)
            cap.release()
        else:
            Camerastatus_lbl = tk.Label(forFrame, text="Camera_LH_Not Connected", fg="#aa0000", bg="#ffffff")
            Camerastatus_lbl.configure(font=("Times New Roman", 11,'bold'))
            Camerastatus_lbl.grid(row=4, column=1, padx=2, pady=1)

        cap1 = cv2.VideoCapture(1,cv2.CAP_DSHOW)   
        if (cap1.isOpened):
           Camerastatus_lbl = tk.Label(forFrame, text=" Camera_RH_Connected", fg="#00aa00", bg="#ffffff")
           Camerastatus_lbl.configure(font=("Times New Roman", 11,'bold'))
           Camerastatus_lbl.grid(row=5, column=1, padx=2, pady=1)
           cap1.release()
        else:
           Camerastatus_lbl = tk.Label(forFrame, text=" Camera_RH_Not_Connected", fg="#aa0000", bg="#ffffff")
           Camerastatus_lbl.configure(font=("Times New Roman", 11,'bold'))
           Camerastatus_lbl.grid(row=5, column=1, padx=2, pady=1) 
    except Exception as Ex:
        print ("Cameraconnection" +str(Ex))
# modelss = torch.load(r'best_v7.pt')  # Use 'cuda' instead of 'cpu' if you want to use GPU

# Set the model to evaluation mode
# /modelss.eval()

# serialport=serial.Serial(port="COM3",baudrate=9600,bytesize=7,parity=serial.PARITY_EVEN,timeout=2,stopbits=serial.STOPBITS_ONE)
# veri=""
# def sericheck():
#   #
#   serialport.close()
#   if serialport.is_open!=True:
#     serialport.open()
# def y0xon():
#     serialport.write(b':01050500FF00F6\r\n') #y0 on
#     verii=serialport.readline().decode('utf-8')
#     print("gelenver>"+str(verii))
    # time.sleep(0.1)
def RGB(event, x, y,
        flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        # print(point)
  
cv2.namedWindow("RGB",cv2.WINDOW_FREERATIO)
cv2.setMouseCallback('RGB', RGB)

def camera_connect():
    global cap,cap1,size,size1
    
    # cap=cv2.VideoCapture(1)
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)#1920x1080:rtsp://thenmozhi:ktm%4012345@192.168.1.100:554?stream1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)
    frame_width = int(cap.get(3)) 
    frame_height = int(cap.get(4)) 
    size = (frame_width, frame_height) 
    
    # cap1=cv2.VideoCapture(2)
    cap1 = cv2.VideoCapture(2,cv2.CAP_DSHOW)#1920x1080:rtsp://thenmozhi:ktm%4012345@192.168.1.100:554?stream1
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
    frame_width1 = int(cap1.get(3)) 
    frame_height1 = int(cap1.get(4)) 
    size1 = (frame_width1, frame_height1) 
    
    while True:
        ret,frame=cap.read()
        if  ret:
            cv2.namedWindow("frame",cv2.WINDOW_FREERATIO)
            cv2.imshow("frame",frame)
            
            cv2.imwrite("frame.bmp",frame)
            forDispImg = Image.fromarray(frame)
            forDispImg.thumbnail((1700, 700))  # 550,550 
            forDispImg = ImageTk.PhotoImage(forDispImg)
            lblImage.configure(image=forDispImg)
            lblImage.image = forDispImg
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

        ret1,frame1=cap1.read()
        if ret1:
            cv2.imwrite("frame1.bmp",frame1)
            cv2.namedWindow("frame1",cv2.WINDOW_FREERATIO)
            cv2.imshow("frame1",frame1)
            
            forDispImg2 = Image.open(r'frame1.bmp')
            forDispImg2.thumbnail((1700, 700))  # 500,700
            forDispImg2 = ImageTk.PhotoImage(forDispImg2)
            lblImage2.configure(image=forDispImg2)
            lblImage2.image = forDispImg2
            if cv2.waitKey(1)&0xFF == ord('q'):
                break

# camera_connect()

# path=r"D:\THENMOZHI\Tata Helmet Detection\yolov8helmetdetection-main\Software video"
# lst = os.listdir(path) # your directory path
# number_files = len(lst)
# number_files+=2
# size=1920,1080
# result = cv2.VideoWriter(path+"//"+str(number_files)+".avi",  
#                             cv2.VideoWriter_fourcc(*'MJPG'), 
#                             3, size)
# my_file = open("coco1.txt", "r")
# data = my_file.read()
# class_list = data.split("\n") 
#print(class_list)

def person_saftykit_detection(frame):
    # while True:    
        # ret,frame = cap.read()
        # ret1,frame1 = cap1.read()
        CONFIDENCE_THRESHOLD = 0.8
        GREEN = (0, 255, 0)
        detections = model(frame)[0]
        count=0
            
            # not_detected()
                # loop over the detections
        for data in detections.boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the detection
            confidence = data[4]

            # filter out weak detections by ensuring the 
            # confidence is greater than the minimum confidence
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # draw the bounding box on the frame
            # frame.plot()
            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            cv2.rectangle(frame, (xmin, ymin) , (xmax, ymax), GREEN, 2)
            cvzone.putTextRect(frame,'person',(xmin, ymin),1,1)
            # person_detected()
            cv2.imwrite(str(count)+"frame.bmp",frame)
    
            # if count % 3 != 0:
            #     continue
            #camera_1
            inrange=frame.copy()
            frame=cv2.resize(frame,(1800,1000))
            
            results=model.predict(frame)
            print("result=",results[0].names)
            a=results[0].boxes.data
            px=pd.DataFrame(a).astype("float")
            print("a=",a)
            
            list=[]
            epron=""
            gloves=""
            helmet=""
            
            for index,row in px.iterrows():
        #        print(row)
                count += 1
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                print(d)
                # c=class_list[d]
        
                pixelcount=0
                if d==2:
                    cv2.imwrite("inrange.bmp",inrange)
                    inrange_copy=cv2.cvtColor(inrange,cv2.COLOR_BGR2HSV)
                    inrange_copy=cv2.inRange(inrange_copy,(122,102,80),(156,255,255))
                    pixelcount=cv2.countNonZero(inrange_copy)

                if pixelcount>3000:
                    print("pixelcount=",pixelcount)
                    cv2.namedWindow("inrange",cv2.WINDOW_FREERATIO)
                    cv2.imshow("inrange", inrange_copy)
                    
                    cv2.imwrite("inrange1.bmp",inrange_copy)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                    # cvzone.putTextRect(frame,'epron',(x1,y1),1,1)
                    epron="epron"
                if d==1:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
                    # cvzone.putTextRect(frame,'gloves',(x1,y1),1,1)
                    gloves="gloves"
                if d==0:
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                    #    frame = cv2.putText(frame, 'helmet', (x1,y1), cv2.FONT_HERSHEY_PLAIN ,  2, (0, 255, 0) , 1, cv2.LINE_AA)
                    # cvzone.putTextRect(frame,'helmet',(x1,y1),1,1)
                    helmet="helmet"
            # result.write(inrange)
        return frame
        
    #     # camera_2
    #     inrange1=frame1.copy()
    #     frame1=cv2.resize(frame1,(1800,1000))
        

    #     result1=model.predict(frame1)
    #     print("result=",result1[0].names)
    #     a1=result1[0].boxes.data
    #     px1=pd.DataFrame(a1).astype("float")
    #     print("a1=",a1)
        
    #     list1=[]
    #     epron1=""
    #     gloves1=""
    #     helmet1=""
        
    #     for index,row in px1.iterrows():
    # #        print(row)
    #         count += 1
    #         x1=int(row[0])
    #         y1=int(row[1])
    #         x2=int(row[2])
    #         y2=int(row[3])
    #         d=int(row[5])
    #         print(d)
    #         # c=class_list[d]
           
    #         if d==2:
    #             cv2.imwrite("inrange1.bmp",inrange1)
    #             inrange_copy=cv2.cvtColor(inrange1,cv2.COLOR_BGR2HSV)
    #             inrange_copy=cv2.inRange(inrange_copy,(122,102,80),(156,255,255))
    #             pixelcount=cv2.countNonZero(inrange_copy)
    #         if pixelcount>3000:
    #             print("pixelcount1=",pixelcount)
    #             cv2.namedWindow("inrange1",cv2.WINDOW_FREERATIO)
    #             cv2.imshow("inrange1", inrange_copy)
                
    #             cv2.imwrite("inrange1.bmp",inrange_copy)
    #             cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
    #             cvzone.putTextRect(frame,'epron',(x1,y1),1,1)
    #             epron="epron"
    #         if d==1:
    #             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),2)
    #             cvzone.putTextRect(frame,'gloves1',(x1,y1),1,1)
    #             gloves="gloves1"
    #         if d==0:
    #             cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #             #    frame = cv2.putText(frame, 'helmet', (x1,y1), cv2.FONT_HERSHEY_PLAIN ,  2, (0, 255, 0) , 1, cv2.LINE_AA)
    #             cvzone.putTextRect(frame,'helmet1',(x1,y1),1,1)
    #             helmet="helmet1"
    #     result1.write(inrange)
    
    #     if ((helmet!="helmet" and gloves!="gloves") or (helmet!="helmet" and epron!="epron") or (epron!="epron" and gloves!="gloves")) and (count!=0): 
    #         frame = cv2.putText(frame, 'Person missed safty kit', (10,50), cv2.FONT_HERSHEY_SIMPLEX ,  2, (0, 0, 255) , 5, cv2.LINE_AA)
    #     cv2.imshow("RGB", frame)
    #     if cv2.waitKey(1)&0xFF == ord('q'):
    #         break
    # result.release()
    # result1.release()
    # cap.release()
    # cap1.release()
    # cv2.destroyAllWindows()

checking=False
Helmet_epron_count =0
frame_check=0

Helmet__count = 0                  
helmet_checking =False
helmet_frame_check =0

Epron__count = 0                  
epron_checking =False
epron_frame_check =0

def kit_detection(image,draw_img):
    global framecount1
    
    # Both epron and helmet
    global checking
    global Helmet_epron_count
    global frame_check
   
    # Helmet
    global Helmet__count                
    global helmet_checking 
    global helmet_frame_check 
     
    # Epron 
    global Epron__count                 
    global epron_checking 
    global epron_frame_check 

    epron_count =0
    helmet_count =0
    
    results=model.predict(image,conf=0.7)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    #  print(px)          
    list=[]

    kit_helmet = False
    kit_epron = False
    detection = False
    kit_string =""
    
    
    if px.size==0:
            
            lblHeading_cam_Lh_kit = tk.Label(forLabel, text="Helmet and Epron not detected", fg="#aa0000", bg="#ffffff")
            lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 12,'bold'))
            lblHeading_cam_Lh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)

            if checking==False:
                Helmet_epron_count = Helmet_epron_count+1                    
                checking =True
                frame_check =framecount1

            elif (frame_check+5>framecount1):
                Helmet_epron_count = Helmet_epron_count+1     

            else:              
                checking=False
                if Helmet_epron_count>4:
                    Helmet_epron_count=0
                # plc signal 

    else:        
        checking = False  
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])                      
            #  print(d)       
            detection = True

            if d==0:
                helmet_count+=1              
                kit_helmet= True
                cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(draw_img,'Helmet',(x1,y1),2,2)      
            if d==2:
                epron_count+=1               
                kit_epron = True
                cv2.rectangle(draw_img,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(draw_img,'Epron',(x1,y1),2,2)
        
        if detection:
            detection = False
            if person_count_LH<=helmet_count:  #Because In same frame without detecting person helmet counts happening
                
                # for continuous check
                helmet_checking=False
                Helmet__count=0

                kit_string = str(helmet_count)+" Helmet detected"
                lblHeading_cam_Lh_kit = tk.Label(forLabel, text=kit_string, fg="#aa0000", bg="#ffffff")
                lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 12,'bold'))
                lblHeading_cam_Lh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)
                
            else:
                kit_string = str(helmet_count)+" Helmet detected"
                lblHeading_cam_Lh_kit = tk.Label(forLabel, text=kit_string, fg="#aa0000", bg="#ffffff")
                lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 12,'bold'))
                lblHeading_cam_Lh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2) 

                # Continuous check for helmet 
                if helmet_checking==False:
                    Helmet__count = Helmet__count+1                    
                    helmet_checking =True
                    helmet_frame_check =framecount1

                elif (helmet_frame_check+5>framecount1):
                    Helmet__count = Helmet__count+1     

                else:    
                    helmet_checking=False
                    if Helmet__count>4:
                        Helmet__count=0
                        # plc signal           
                
            if person_count_LH <= epron_count:
                # for continuous check
                epron_checking=False
                Epron__count=0
                st = str(epron_count)+" Epron detected"
                kit_string = kit_string+" and "+st
                lblHeading_cam_Lh_kit = tk.Label(forLabel, text=kit_string, fg="#aa0000", bg="#ffffff")
                lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 12,'bold'))
                lblHeading_cam_Lh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)
                
            else:
                st = str(epron_count)+" Epron detected"
                kit_string = kit_string+" and "+st
                lblHeading_cam_Lh_kit = tk.Label(forLabel, text=kit_string, fg="#aa0000", bg="#ffffff")
                lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 12,'bold'))
                lblHeading_cam_Lh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)

                if epron_checking==False:
                    Epron__count = Epron__count+1                    
                    epron_checking =True
                    epron_frame_check =framecount1

                elif (epron_frame_check+5>framecount1):
                    Epron__count = Epron__count+1     

                else:    
                    epron_checking=False
                    if Epron__count>4:
                        Epron__count=0

    return draw_img

def kit_detection2(image):

    kit_helmet = False
    kit_epron = False
    detection = False

    helmet_count =0
    epron_count =0

    results=model.predict(image,conf=0.7)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    #  print(px)          
    list=[]
    
    if px.size==0:
            lblHeading = tk.Label(forLabel1, text="Helmet and Epron not detected", fg="#aa0000", bg="#ffffff")
            lblHeading.configure(font=("Times New Roman",12,'bold'))
            lblHeading.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)
    else :    
        for index,row in px.iterrows():
            x1=int(row[0])
            y1=int(row[1])
            x2=int(row[2])
            y2=int(row[3])
            d=int(row[5])                      
            #  print(d)       
                            
            detection = True

            if d==0:
                helmet_count +=1
                kit_helmet= True
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(image,'Helmet',(x1,y1),2,2)      
            if d==2:
                epron_count+=1
                kit_epron = True
                cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
                cvzone.putTextRect(image,'Epron',(x1,y1),2,2)
        if detection:
            if person_count_RH==helmet_count:
                    kit_string = str(helmet_count)+" Helmet detected"
                    lblHeading_cam_Rh_kit = tk.Label(forLabel1, text=kit_string, fg="#aa0000", bg="#ffffff")
                    lblHeading_cam_Rh_kit.configure(font=("Times New Roman", 12,'bold'))
                    lblHeading_cam_Rh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)
                    
            else:
                    kit_string =str(helmet_count)+ " Helmet detected"  
                    lblHeading_cam_Rh_kit = tk.Label(forLabel1, text=kit_string, fg="#aa0000", bg="#ffffff")
                    lblHeading_cam_Rh_kit.configure(font=("Times New Roman", 12,'bold'))
                    lblHeading_cam_Rh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2) 
                    
            if person_count_RH == epron_count:
                    st = str(epron_count)+" Epron detected"
                    kit_string = kit_string+" and "+st
                    lblHeading_cam_Rh_kit = tk.Label(forLabel1, text=kit_string, fg="#aa0000", bg="#ffffff")
                    lblHeading_cam_Rh_kit.configure(font=("Times New Roman", 12,'bold'))
                    lblHeading_cam_Rh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)               
            else:
                    st =str(epron_count) +" Epron detected"
                    kit_string = kit_string+"and "+st
                    lblHeading_cam_Rh_kit = tk.Label(forLabel1, text=kit_string, fg="#aa0000", bg="#ffffff")
                    lblHeading_cam_Rh_kit.configure(font=("Times New Roman", 12,'bold'))
                    lblHeading_cam_Rh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)
                
    return image

def person_detection_new(frame,draw_img):
     global person_count_LH
     results=person_model.predict(frame,conf=0.75)
     frame_kit = draw_img
     a=results[0].boxes.data
     px=pd.DataFrame(a).astype("float")
     
    #  person_results = [box for box in results[0].boxes.data if box[0] == 0]  # Assuming person class index is 0
    # #  a=person_results[0].boxes.data
    #  px=pd.DataFrame(person_results).astype("float")

    #  lblHeading_cam_Lh_Person = tk.Label(forFrame, text="Person_Status= Person Not detected", fg="#1e6ec8", bg="#ffffff")
    #  lblHeading_cam_Lh_Person.configure(font=("Times New Roman", 15,'bold'))
    #  lblHeading_cam_Lh_Person.grid(row=1, column=4, sticky="nsew", padx=12, pady=7)

    #  print(px)
     list=[]
     person_count_LH =0
     for index,row in px.iterrows():
         #        print(row)
                    
         x1=int(row[0])
         y1=int(row[1])
         x2=int(row[2])
         y2=int(row[3])
         d=int(row[5])                      
         
        #  print(d)       
                        
         if d==0:
            
            cv2.rectangle(frame_kit,(x1,y1),(x2,y2),(255,0,255),2)          
            person_count_LH+=1
            cvzone.putTextRect(frame_kit,'person',(x1,y1),3,3)

            #y3 1284 = plc connection,camera connection
            #y4 1285 = helmet,epron
            #y5 1286 = person
     
     if person_count_LH!=0:  
            lblHeading_cam_Lh_Person = tk.Label(forLabel, text=str(person_count_LH)+" Person Detected!", fg="#ff0000", bg="#ffffff")
            lblHeading_cam_Lh_Person.configure(font=("Times New Roman", 15,'bold'))
            lblHeading_cam_Lh_Person.grid(row=1, column=1, sticky="nsew", padx=2, pady=1)

            frame_kit =kit_detection(frame,draw_img)
     else:
            lblHeading_cam_Lh_Person = tk.Label(forLabel, text="Person Not Detected!", fg="#00aa00", bg="#ffffff")
            lblHeading_cam_Lh_Person.configure(font=("Times New Roman", 15,'bold'))
            lblHeading_cam_Lh_Person.grid(row=1, column=1, sticky="nsew", padx=2, pady=1)

            lblHeading_cam_Lh_kit = tk.Label(forLabel, text="Kit status", fg="#aa0000", bg="#ffffff")
            lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 12,'bold'))
            lblHeading_cam_Lh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)


     return frame_kit 

def person_detection_new2(frame):
     global person_count_RH
     results=person_model.predict(frame,conf=0.75)
     frame_kit = frame
     a=results[0].boxes.data
     px=pd.DataFrame(a).astype("float")
     
    #  person_results = [box for box in results[0].boxes.data if box[0] == 0]  # Assuming person class index is 0
    # #  a=person_results[0].boxes.data
    #  px=pd.DataFrame(person_results).astype("float")

    #  lblHeading_cam_Rh_Person = tk.Label(forLabel, text="Person Not detected", fg="#1e6ec8", bg="#ffffff")
    #  lblHeading_cam_Rh_Person.configure(font=("Times New Roman", 15,'bold'))
    #  lblHeading_cam_Rh_Person.grid(row=1, column=6, sticky="nsew", padx=12, pady=7)
     
    #  print(px)
     list=[]
     person_count_RH =0

     for index,row in px.iterrows():
         #        print(row)
                    
         x1=int(row[0])
         y1=int(row[1])
         x2=int(row[2])
         y2=int(row[3])
         d=int(row[5])                      
         
        #  print(d)       
                        
         if d==0:
            
            cv2.rectangle(frame_kit,(x1,y1),(x2,y2),(255,0,255),2)          
            person_count_RH+=1
            cvzone.putTextRect(frame_kit,'person',(x1,y1),3,3)

            #y3 1284 = plc connection,camera connection
            #y4 1285 = helmet,epron
            #y5 1286 = person
     
     if person_count_RH!=0:
        lblHeading_cam_Rh_Person = tk.Label(forLabel1, text=str(person_count_RH)+" Person Detected!", fg="#ff0000", bg="#ffffff")
        lblHeading_cam_Rh_Person.configure(font=("Times New Roman", 15,'bold'))
        lblHeading_cam_Rh_Person.grid(row=1, column=1,sticky="nsew",  padx=2, pady=1)        
        frame_kit =kit_detection2(frame_kit)

     else :
        lblHeading_cam_Rh_Person = tk.Label(forLabel1, text="Person Not Detected!", fg="#00aa00", bg="#ffffff")
        lblHeading_cam_Rh_Person.configure(font=("Times New Roman", 15,'bold'))
        lblHeading_cam_Rh_Person.grid(row=1, column=1, sticky="nsew",  padx=2, pady=1)

        lblHeading_cam_Rh_kit = tk.Label(forLabel1, text="Kit status", fg="#aa0000", bg="#ffffff")
        lblHeading_cam_Rh_kit.configure(font=("Times New Roman", 12,'bold'))
        lblHeading_cam_Rh_kit.grid(row=2, column=1, sticky="nsew", padx=1, pady=2)    

     return frame_kit 

framecount1 =0

def cam1_fun():   
    global framecount1
    while  True:      
             
            ret, frame = cap1.read()
            cv2.imwrite("a.bmp",frame)
            framecount1 = framecount1+1

            frame_for_save = frame
            frame_draw = frame

            height, width = frame.shape[0],frame.shape[1]
            black_image = np.zeros((height, width), dtype=np.uint8)

            # Step 2: Define polygon vertices (example polygon)
            polygon_vertices = np.array([[776, 4], [1900, 4], [1900, 1072], [620, 1072]], dtype=np.int32)
                                
            # Step 3: Fill the polygon with white color on the black image
            cv2.fillPoly(black_image, [polygon_vertices], 255)

            # Step 4: Use the filled polygon as a mask on your original image
            # original_image = cv2.imread('your_image.jpg')  # Replace 'your_image.jpg' with your image file path
            masked_image = cv2.bitwise_and(frame, frame, mask=black_image)
 
            if video_save.get() == 1:
                video1.write(frame_for_save)
            # lblHeading_cam_Lh_Person = tk.Label(forLabel1, text="Person Not Detected!", fg="#00aa00", bg="#ffffff")
            # lblHeading_cam_Lh_Person.configure(font=("Times New Roman", 15,'bold'))
            # lblHeading_cam_Lh_Person.grid(row=1, column=1,  padx=2, pady=1)

            if not ret:
                break
            
            if video_save.get() != 1: 
                frame =person_detection_new(masked_image,frame_draw)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((700, 400))  # Resize image if necessary
            imgtk = ImageTk.PhotoImage(image=img)
            lblImage.config(image=imgtk)
            lblImage.image = imgtk  # Keep a reference to prevent garbage collection

            if (cam_release):           
              break
            window.update_idletasks()  # Update the GUI
    
    if video_save.get() == 1:   
        video1.release()
    cap1.release()
    messagebox.showinfo("Cam1 Video Finished", "Cam1 Video playback completed.")        

def cam2_fun():    
    while True:
            ret2, frame2 = cap2.read()

            if not ret2:
                break

            if video_save.get() == 1:
                video2.write(frame2) 
            # lblHeading_cam_Rh_Person = tk.Label(forLabel, text="Person Not Detected!", fg="#00aa00", bg="#ffffff")
            # lblHeading_cam_Rh_Person.configure(font=("Times New Roman", 15,'bold'))
            # lblHeading_cam_Rh_Person.grid(row=1, column=1,  padx=2, pady=1)

            # forDispImg = Image.fromarray(frame)
            # forDispImg.thumbnail(((800, 500)))  # 1000, 500
            # forDispImg = ImageTk.PhotoImage(forDispImg)
            # pb1.configure(image=forDispImg)
            # pb1.image = forDispImg    
            if video_save.get() != 1:        
                frame2 =person_detection_new2(frame2)

            frame_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((700, 400))  # Resize image if necessary
            imgtk = ImageTk.PhotoImage(image=img)
            lblImage2.config(image=imgtk)
            lblImage2.image = imgtk  # Keep a reference to prevent garbage collection
            if (cam_release):           
              break
            window.update_idletasks()  # Update the GUIs
    if video_save.get() == 1:   
        video2.release()  
    cap2.release()
    messagebox.showinfo("Cam2 Video Finished", "Cam2 Video playback completed.")

def camera_connect_new():

    global cap2
    global cap1

    global video1
    global video2

    global cam_release
    cam_release =False

    global framecount1
    framecount1 =0
    # Both epron and helmet
    global checking
    global Helmet_epron_count
    global frame_check
    
    global Helmet__count                
    global helmet_checking 
    global helmet_frame_check 
        
    global Epron__count                 
    global epron_checking 
    global epron_frame_check 
    
    checking = False
    helmet_checking  = False
    epron_checking  = False
        
    Epron__count  =0
    Helmet__count=0 
    Helmet_epron_count =0

    epron_frame_check=0
    helmet_frame_check=0
    frame_check =0



    cap1 = cv2.VideoCapture(1,cv2.CAP_DSHOW)#1920x1080:rtsp://thenmozhi:ktm%4012345@192.168.1.100:554?stream1
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    frame_width = int(cap1.get(3)) 
    frame_height = int(cap1.get(4)) 
    size = (frame_width, frame_height) 

    current_time = datetime.now()
    formatted_time = current_time.strftime("%H-%M-%S")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video1 = cv2.VideoWriter("CamLH"+str(formatted_time)+".mp4", fourcc, 5, (1280, 720))

    # cap1=cv2.VideoCapture(2)
    cap2 = cv2.VideoCapture(0,cv2.CAP_DSHOW)#1920x1080:rtsp://thenmozhi:ktm%4012345@192.168.1.100:554?stream1
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    frame_width1 = int(cap2.get(3)) 
    frame_height1 = int(cap2.get(4)) 
    size1 = (frame_width1, frame_height1) 

    fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
    video2 = cv2.VideoWriter("CamRH"+str(formatted_time)+".mp4", fourcc, 5, (1280, 720))


    thread_one = threading.Thread(target=cam1_fun)
    thread_two = threading.Thread(target=cam2_fun)

    thread_one.start()
    thread_two.start()

def camera_release():
    global cam_release
    cam_release = True

def Browse_video_new():
    try:
        global filepath
        global imgForTest
        global cap2
        global cap1

        global cam_release
        cam_release =False
        
        global framecount1
        framecount1 =0
    # Both epron and helmet
        global checking
        global Helmet_epron_count
        global frame_check
    
        # Helmet
        global Helmet__count                
        global helmet_checking 
        global helmet_frame_check 
        
        # Epron 
        global Epron__count                 
        global epron_checking 
        global epron_frame_check 

        checking = False
        helmet_checking  = False
        epron_checking  = False
        
        Epron__count  =0
        Helmet__count=0 
        Helmet_epron_count =0

        epron_frame_check=0
        helmet_frame_check=0
        frame_check =0

        


        filepath = askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
        cap1=cv2.VideoCapture(filepath)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter("5_op.mp4", fourcc, 2, (3840, 2160))
        
        if not cap1.isOpened():
            print("Error: Couldn't open the video file.")
            return
        
        filepath2 = askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
        cap2=cv2.VideoCapture(filepath2)

        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        video2 = cv2.VideoWriter("5_op.mp4", fourcc, 2, (3840, 2160))

        if not cap2.isOpened():
            print("Error: Couldn't open the video file.")
            return
        
        thread_one = threading.Thread(target=cam1_fun)
        thread_two = threading.Thread(target=cam2_fun)

        thread_one.start()
        thread_two.start()

       
    except Exception as Ex:
        print("Browse: " + str(Ex))

def browse_video():
    global lblImage  # Access the global lblImage variable
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mkv")])
    if video_path:
        cap = cv2.VideoCapture(video_path)
        show_frame(cap)

def show_frame(cap):
    global lblImage  # Access the global lblImage variable
    ret, frame = cap.read()
    if ret:
        # frame=person_detection_new(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (800, 700))
        
        image = Image.fromarray(frame)
        image.thumbnail((700, 400))
        photo = ImageTk.PhotoImage(image=image)
        lblImage2.configure(image=photo)
        lblImage2.image = photo
        # lblImage2.after(10, lambda: show_frame(cap))
        
        lblImage.configure(image=photo)
        lblImage.image = photo
        # lblImage.after(10, lambda: show_frame(cap))
        window.update_idletasks()
    else:
        cap.release()
       
##UI##
window = tk.Tk()
window.state('zoomed')
window.title("PERSON INSPECTION")
# window.iconbitmap(r'Icon.ico')
# window.geometry("1920x1080")
w = 1360
h = 768
ws = window.winfo_screenwidth()
hs = window.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)-30
window.geometry('%dx%d+%d+%d' % (w, h, x, y))
window['background'] = '#ffffff'         #'#2e9595'             #'#123456'
##UI##

##Redbot Logo##
forRLogo = tk.Frame(window,bg='#ffffff')
forRLogo.place(x=20, y=15)
lblRLogo = tk.Label(forRLogo)
lblRLogo.grid(row=0, column=0, sticky = "w")
RLogoImage = Image.open(r'Redbot.png')
# cv2.resize(RLogoImage,RLogoImage,(500,500))
RLogoImage.thumbnail((200, 400))
RLogoImage = ImageTk.PhotoImage(RLogoImage)
lblRLogo.configure(image = RLogoImage)
lblRLogo.image = RLogoImage
##Redbot Logo##

##CUSTOMER Logo##
forRLogo1 = tk.Frame(window,bg='#ffffff')
forRLogo1.place(x=1050, y=15)
lblRLogo1 = tk.Label(forRLogo1)
lblRLogo1.grid(row=0, column=0, sticky = "w")
RLogoImage1 = Image.open(r'tata_motors.jpg')
# cv2.resize(RLogoImage,RLogoImage,(500,500))
RLogoImage1.thumbnail((200,150))
RLogoImage1 = ImageTk.PhotoImage(RLogoImage1)
lblRLogo1.configure(image = RLogoImage1)
lblRLogo1.image = RLogoImage1
##CUSTOMER Logo##

##Main Label##
forlabelheading = tk.Frame(window)
forlabelheading.place(x=310, y=15)
lblHeading = tk.Label(forlabelheading, text="HUMAN SAFETY KIT DETECTION", fg="#1e6ec8", bg="#ffffff")
lblHeading.configure(font=("Times New Roman", 32,'bold','underline'))
# lblHeading.configure(underline = True)
lblHeading.grid(row=3, column=0)

##Image Display##
forImage = tk.Frame(window)
forImage.place(x=40, y=140)
lblImage = tk.Label(forImage)
lblImage.grid(row=0, column=0)
forDispImg = Image.open(r'cam11.jpg')
forDispImg.thumbnail((700, 400))  # 550,550 
forDispImg = ImageTk.PhotoImage(forDispImg)
lblImage.configure(image=forDispImg)
lblImage.image = forDispImg
lblImage2 = tk.Label(forImage)
lblImage2.grid(row=0, column=1)
forDispImg2 = Image.open(r'cam22.jpg')
forDispImg2.thumbnail((700, 400))  # 500,700
forDispImg2 = ImageTk.PhotoImage(forDispImg2)
lblImage2.configure(image=forDispImg2)
lblImage2.image = forDispImg2
##Image Display##

 ##Check Box##
# forCheckBox = tk.Frame(window,bg='#ffffff')
# forCheckBox.place(x=700, y=600)
# cam1_var1 = tk.IntVar()
# cam2_var2 = tk.IntVar()
# cam1_cb = tk.Checkbutton(forCheckBox, text='Camera 1', variable=cam1_var1, onvalue=1, offvalue=0)
# cam1_cb.configure(font=("Times New Roman", 15,'bold'))
# cam1_cb.grid(row=0, column=0, sticky = "w", padx=12, pady=7)
# cam2_cb = tk.Checkbutton(forCheckBox, text='Camera 2', variable=cam2_var2, onvalue=1, offvalue=0)
# cam2_cb.configure(font=("Times New Roman", 15,'bold'))
# cam2_cb.grid(row=1, column=0, sticky = "w", padx=12, pady=7)

# ##Check Box##
forCheckBox = tk.Frame(window,bg='#ffffff')
forCheckBox.place(x=20, y=540)
online_var1 = tk.IntVar()
video_save = tk.IntVar()
online_cb = tk.Checkbutton(forCheckBox, text='Online', variable=online_var1, onvalue=1, offvalue=0)
online_cb.configure(font=("Times New Roman", 15,'bold'))
online_cb.grid(row=0, column=0, sticky = "w", padx=12, pady=7)
offline_cb = tk.Checkbutton(forCheckBox, text='Video_save', variable=video_save, onvalue=1, offvalue=0)
offline_cb.configure(font=("Times New Roman", 15,'bold'))
offline_cb.grid(row=1, column=0, sticky = "w", padx=9, pady=7)

# forplcandcamera = tk.Frame(window,bg='#ffffff')
# forplcandcamera.place(x=10, y=600)

# plcstatus_lbl = tk.Label(forplcandcamera, text="PLC_STATUS: ", fg="#5500ff", bg="#000000")
# plcstatus_lbl.configure(font=("Times New Roman", 17,'bold'))
# plcstatus_lbl.grid(row=0, column=0, padx=2, pady=1)

# plcstatus_lbl = tk.Label(forplcandcamera, text=" ", fg="#5500ff", bg="#000000")
# plcstatus_lbl.configure(font=("Times New Roman", 17,'bold'))
# plcstatus_lbl.grid(row=0, column=1, padx=2, pady=1)

##Buttons##
forFrame = tk.Frame(window,bg='#ffffff')
forFrame.place(x=150, y=540)  # 350

btn_browse = tk.Button(forFrame, text="BROWSE VIDEO", command=Browse_video_new, bg="#1e6ec8" , fg="white")
btn_browse.configure(font=("Times New Roman", 14))
btn_browse.grid(row=0, column=0, sticky="nsew", padx=12, pady=7)

# btn_browse = tk.Button(forFrame, text="BROWSE VIDEO 2", command=Browse_video_new, bg="#1e6ec8" , fg="white")
# btn_browse.configure(font=("Times New Roman", 14))
# btn_browse.grid(row=0, column=1, sticky="nsew", padx=12, pady=7)

btn_live = tk.Button(forFrame, text="LIVE VIDEO ►", command=camera_connect_new, bg="#1e6ec8" , fg="white")
btn_live.configure(font=("Times New Roman", 14))
btn_live.grid(row=0, column=1, sticky="nsew", padx=12, pady=7)

btn_live = tk.Button(forFrame, text="Close camera", command=camera_release, bg="#1e6ec8" , fg="white")
btn_live.configure(font=("Times New Roman", 14))
btn_live.grid(row=1, column=1, sticky="nsew", padx=12, pady=7)

plcstatus_lbl = tk.Label(forFrame, text="PLC_STATUS: ", fg="#000000", bg="#ffffff")
plcstatus_lbl.configure(font=("Times New Roman", 12,'bold'))
plcstatus_lbl.grid(row=3, column=0, padx=2, pady=1)

plcstatus_lbl = tk.Label(forFrame, text="PLC_STATUS: ", fg="#000000", bg="#ffffff")
plcstatus_lbl.configure(font=("Times New Roman", 12,'bold'))
plcstatus_lbl.grid(row=3, column=1, padx=2, pady=1)

Camerastatus_lbl = tk.Label(forFrame, text="Camera_RH_Status: ", fg="#000000", bg="#ffffff")
Camerastatus_lbl.configure(font=("Times New Roman", 12,'bold'))
Camerastatus_lbl.grid(row=4, column=0, padx=2, pady=1)

Camerastatus_lbl = tk.Label(forFrame, text=" ", fg="#000000", bg="#ffffff")
Camerastatus_lbl.configure(font=("Times New Roman", 12,'bold'))
Camerastatus_lbl.grid(row=4, column=1, padx=2, pady=1)

Camerastatus_lbl = tk.Label(forFrame, text="Camera_LH_Status: ", fg="#000000", bg="#ffffff")
Camerastatus_lbl.configure(font=("Times New Roman", 12,'bold'))
Camerastatus_lbl.grid(row=5, column=0, padx=2, pady=1)

Camerastatus_lbl = tk.Label(forFrame, text=" ", fg="#000000", bg="#ffffff")
Camerastatus_lbl.configure(font=("Times New Roman", 12,'bold'))
Camerastatus_lbl.grid(row=5, column=1, padx=2, pady=1)

forLabel = tk.Frame(window,bg='#ffffff')
forLabel.place(x=500, y=560)  # 350

lblHeading_cam_Lh = tk.Label(forLabel, text="Camera LH", fg="#5500ff", bg="#ffffff")
lblHeading_cam_Lh.configure(font=("Times New Roman", 17,'bold'))
lblHeading_cam_Lh.grid(row=0, column=0, padx=2, pady=1)

lblHeading_cam_Lh_Person_label = tk.Label(forLabel, text="Person_status:", fg="#000000", bg="#ffffff")
lblHeading_cam_Lh_Person_label.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Lh_Person_label.grid(row=1, column=0,  padx=2, pady=1)

lblHeading_cam_Lh_Person = tk.Label(forLabel, text="Person Result", fg="#00aa00", bg="#ffffff")
lblHeading_cam_Lh_Person.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Lh_Person.grid(row=1, column=1,  padx=2, pady=1)

lblHeading_cam_Lh_kit_label = tk.Label(forLabel, text="     kit_status: ", fg="#000000", bg="#ffffff")
lblHeading_cam_Lh_kit_label.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Lh_kit_label.grid(row=2, column=0, padx=2, pady=1)

lblHeading_cam_Lh_kit = tk.Label(forLabel, text="Kit Result ", fg="#ff0000", bg="#ffffff")
lblHeading_cam_Lh_kit.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Lh_kit.grid(row=2, column=1, padx=2, pady=1)



# lblHeading = tk.Label(forFrame, text="Epron_status", fg="#1e6ec8", bg="#ffffff")
# lblHeading.configure(font=("Times New Roman", 15,'bold'))
# lblHeading.grid(row=3, column=4, sticky="nsew", padx=12, pady=7)

forLabel1 = tk.Frame(window,bg='#ffffff')
forLabel1.place(x=920, y=560)  # 350

lblHeading_cam_Rh = tk.Label(forLabel1, text="Camera RH" , fg="#5500ff", bg="#ffffff")
lblHeading_cam_Rh.configure(font=("Times New Roman", 17,'bold'))
lblHeading_cam_Rh.grid(row=0, column=0, padx=2, pady=1)

lblHeading_cam_Rh_Person_label = tk.Label(forLabel1, text="Person_status: ", fg="#000000", bg="#ffffff")
lblHeading_cam_Rh_Person_label.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Rh_Person_label.grid(row=1, column=0, padx=2, pady=1)

lblHeading_cam_Rh_Person = tk.Label(forLabel1, text="Person Result", fg="#00aa00", bg="#ffffff")
lblHeading_cam_Rh_Person.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Rh_Person.grid(row=1, column=1,  padx=2, pady=7)

lblHeading_cam_Rh_kit_label = tk.Label(forLabel1, text="    kit_status: ", fg="#000000", bg="#ffffff")
lblHeading_cam_Rh_kit_label.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Rh_kit_label.grid(row=2, column=0,  padx=2, pady=1)

lblHeading_cam_Rh_kit = tk.Label(forLabel1, text="Kit Result", fg="#ff0000", bg="#ffffff")
lblHeading_cam_Rh_kit.configure(font=("Times New Roman", 15,'bold'))
lblHeading_cam_Rh_kit.grid(row=2, column=1, padx=2, pady=1)

# lblHeading = tk.Label(forFrame, text="Epron_status", fg="#1e6ec8", bg="#ffffff")
# lblHeading.configure(font=("Times New Roman", 15,'bold'))
# lblHeading.grid(row=3, column=4, sticky="nsew", padx=12, pady=7)

Plc_connection()
Camera_connection()

window.mainloop()

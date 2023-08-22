import cv2
import os
from ultralytics import YOLO
import bbox_visualizer as bbv
import numpy as np
from typing import Union
from collections import Counter
import time
import torch
from lumina import Lumina

#https://www.youtube.com/live/zzJjopSjIMc?feature=share
# url = 'https://www.youtube.com/watch?v=zzJjopSjIMc'  # Replace with your desired YouTube video URL

# video = pafy.new(url)
# best = video.getbest(preftype='mp4')  # Get the best quality video


# random_colors = {i:tuple(map(int,col)) for i , col in enumerate(np.random.randint(10,255,(100,3)))}

lumina = Lumina()


class ObjectTracker():
    def __init__(self,video_path:os.path,model:os.path) -> None:
        self.cap = cv2.VideoCapture(video_path)
        # self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'height : {self.frame_height} , width : {self.frame_width}')
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.output_video = cv2.VideoWriter('output_video.mp4', self.fourcc, self.fps, (self.frame_width, self.frame_height))
        self.model = YOLO(model)
     
        self.name = self.model.names
        
        # {0: 'face', 1: 'not_face', 2: 'person', 3: 'side'}

    

    def track(self,draw_tail:bool=False,confidence=0.50,iou=.50,skip_frames:Union[None,int]=None,slow_down_video=1,add_label=False):
        if skip_frames:
            count = 1
            
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            frame = cv2.flip(frame,1)

            
            if not ret:
                break
            
            if skip_frames:
                count += 1
                if count % skip_frames != 0:
                    continue
            
            results = self.model.predict(frame,stream=True,conf=0.50,iou=.55)
   
            for result in results:
                box = result.boxes.xyxy.cpu().numpy().astype(int)  
                clss = result.boxes.cls.cpu().numpy().astype(int) 
                conf = result.boxes.conf.cpu().numpy().astype(float)
                for bbox , cls , conf in zip(box , clss , conf):
                   
                    if cls == 0 and conf > 0.75:
                        lumina.rectangle(frame,bbox,(0,255,0),2)
                        lumina.putText(frame,str(self.name[cls]),bbox,rect_color=(0,255,0))
                    elif cls == 1 and conf > 0.50:
                        lumina.rectangle(frame,bbox,(255,0,255),2)
                        lumina.putText(frame,str(self.name[cls]),bbox,rect_color=(255,0,255))
                    elif cls == 2 and conf > 0.80:
                   
                        lumina.rectangle(frame,bbox,(255,0,0),2)
                        lumina.putText(frame,str(self.name[cls]),bbox,rect_color=(255,0,0))
                    elif cls == 3 and conf > 0.50:
                        lumina.rectangle(frame,bbox,(0,0,255),2)
                        lumina.putText(frame,str(self.name[cls]),bbox,rect_color=(0,0,255))
                    
                
            
                    
                    
                   
                        
            #         # color = (255,0,0)
            #         # result = cv2.pointPolygonTest(np.array(self.lane_crop_region), (middle_x,middle_y), measureDist=False)
          
                 
            #         if add_label:
            #             bbv.add_label(frame,f'id:{str(id)}', bbox=bboxs, top=False,text_color=(0,255,0),draw_bg=True,)
            
                    # if draw_tail:
                    #     if id not in self.position:
                    #         self.position[id] = [center]
                    #     if id in self.position:
                    #         self.position[id].append(center)
                            
                    #     for ids , pos in self.position.items():
                 
                    #         for i, p in enumerate(pos): 
                                  
                    #             cv2.circle(frame,p,2,(0,255,0),2)
                     
                        
                
            
            
            FPS = 1.0 / (time.time() - start_time)
            cv2.putText(frame,f'FPS: {round(FPS)}',(20,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
         
            cv2.imshow("frame", frame)
        
            if cv2.waitKey(slow_down_video) & 0xFF == ord("q"):
                break
        # self.cap.release()
        # self.output_video.release()

if __name__ == '__main__':
    tracker = ObjectTracker(video_path=0,model='model\\best_yolov8_home_surveillance.pt') # models name ['yolov8l.pt','yolov8m.pt','yolov8s.pt']
    tracker.track(draw_tail=False,confidence=0.45,iou=0.45,skip_frames=5,)







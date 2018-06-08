
import os
import cv2
#import sys
import pygubu
import numpy as np
import tkinter as tk
from tkinter import filedialog
import base64
from icon import img

import time
from skimage.draw import polygon
from skimage.filters import gaussian
from skimage.measure import find_contours
from skimage.segmentation import active_contour

class Application:
    def __init__(self, master):
        self.master = master

        #create builder
        self.builder = builder = pygubu.Builder()
        #load ui file
        builder.add_from_file('OCT_UI.ui')
        #create a widget
        self.mainwindow = builder.get_object('window', master)
        #connect callback
        builder.connect_callbacks(self)

        #initial item
        self.Spinbox_Interval = builder.get_object('Spinbox_Interval', master)
        self.Checkbutton_Show_Contour = builder.get_object('Checkbutton_Show_Contour', master)
        self.Checkbutton_Show_Contour.state(['!alternate']) # (alternate,) -> ()
        self.Checkbutton_Show_Contour.state(['selected']) # () -> (selected,)
        self.Checkbutton_Fine_Tune = builder.get_object('Checkbutton_Fine_Tune', master)
        self.Checkbutton_Fine_Tune.state(['!alternate']) # (alternate,) -> ()
        return
####################################################################################################
    def Button_Load_File_Click(self):
#        print('Button_Load_File_Click')
        path_name = filedialog.askdirectory()
        if path_name == '':
            return
        self.input_path_name = path_name
        print ('Input_files:', path_name)
        
        self.frames_all = []
        self.contour_all = []
        self.contour_x = []
        self.contour_y = []
        self.now_frame = 0
        self.now_Interval = 25
        self.Spinbox_Interval.set(25)
        self.ix = 0
        self.iy = 0
        self.L_button_down = False
        self.R_button_down = False
        self.fine_tune = False
        self.Checkbutton_Fine_Tune.state(['!selected'])
        
        files = os.listdir(path_name)
        while(len(files)>0):
            frame = cv2.imread(path_name+'/'+files.pop(0))
            self.frames_all.append(np.array(frame, dtype=np.uint8))
            self.contour_all.append(np.zeros_like(frame[:,:,0], dtype=np.uint8))
        
        cv2.destroyAllWindows()
        cv2.namedWindow(path_name)
        self.update_image()
        cv2.createTrackbar('frame', path_name, 0, len(self.frames_all), self.Trackbar_Change)
        cv2.setMouseCallback(path_name, self.draw_contour)
        return
####################################################################################################
    def Button_Load_Init_Click(self):
        print('Button_Load_Init_Click')
        filename = filedialog.askopenfilename()
        if filename == '':
            return
        print ('Input_file:', filename)
        inputim = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        inputim[inputim>=128]=255
        inputim[inputim<128]=0
        self.contour_all[self.now_frame][0:360,0:800] = inputim
        self.update_image()
        return
####################################################################################################
    def Button_Save_Result_Click(self):
        print('Button_Save_Result_Click')
        filename = self.input_path_name.split('/')[-1].split('.')[0]
        print(filename)
        path_name = filedialog.askdirectory()
        if path_name == '':
            return
        path_name = path_name + '/' + filename + '/'
        print ('Output_path:', path_name)
        L = len(self.contour_all)
        os.mkdir(path_name)
        file_number = 0
        for i in range(L):
            print(i)
            if np.sum(self.contour_all[i])!=0 :
                cv2.imwrite(path_name+str(file_number)+'.bmp', self.contour_all[i][0:360,0:800])
                file_number = file_number + 1
        return
####################################################################################################
    def predict(self, contour_number, frame_number, snake=None):
        ''' initial '''
        if snake==None:
            img_init = self.contour_all[contour_number]
            init = find_contours(img_init, 0.5)
            init = init[0][::self.now_Interval]
            init = init[:,::-1]
            init = init[init[:,0].argsort()]
        else:
            init = snake.copy()
        print(frame_number, len(init))
        ''' target '''
        img_org = self.frames_all[frame_number]
        img_org = gaussian(img_org, 3)
        ''' snake '''
        snake = active_contour(img_org, init, alpha=0.001, beta=0.001, w_line=0, w_edge=5, max_iterations=500)
        snake_int = np.array(snake, dtype=np.int32)
        rr, cc = polygon(snake_int[:,1], snake_int[:,0])
        self.contour_all[frame_number] = np.zeros_like(self.contour_all[frame_number])
        self.contour_all[frame_number][rr, cc] = 255
        return
####################################################################################################
    def Button_Tracking_All_Click(self):
        print('Button_Tracking_All_Click')
        timeStart = time.time()
        self.predict(0, 0)
        for frame in range(1, len(self.frames_all)):
            self.predict(frame-1, frame)
        self.update_image()
        timeEnd = time.time()
        print('Done, time= %.5f'%(timeEnd-timeStart))
        return
####################################################################################################
    def Button_Tracking_Forward_Click(self):
        print('Button_Tracking_Forward_Click')
        for frame in range(self.now_frame+1, len(self.frames_all)):
            self.predict(frame-1, frame)
        self.update_image()
        print('Done')
        return
####################################################################################################
    def Button_Tracking_Backward_Click(self):
        print('Button_Tracking_Backward_Click')
        for frame in range(self.now_frame-1, -1, -1):
            self.predict(frame+1, frame)
        self.update_image()
        print('Done')
        return
####################################################################################################
    def Button_Clear_Current_Click(self):
        print('Button_Clear_Current_Click')
        self.contour_all[self.now_frame] = np.zeros_like(self.contour_all[self.now_frame])
        self.update_image()
        return
####################################################################################################
    def Button_Clear_All_Click(self):
        print('Button_Clear_All_Click')
        L = len(self.contour_all)
        for i in range(L):
            self.contour_all[i] = np.zeros_like(self.contour_all[i])
        self.update_image()
        return
####################################################################################################
    def Spinbox_Interval_Change(self):
#        print('Spinbox_Interval_Change')
        self.now_frame = 0
        self.now_Interval = int(self.Spinbox_Interval.get())
        cv2.createTrackbar('frame', self.input_path_name, 0, len(self.frames_all)//self.now_Interval, self.Trackbar_Change)
        return
####################################################################################################
    def Checkbutton_Fine_Tune_Change(self):
#        print('Checkbutton_Fine_Tune_Change')
        self.fine_tune = True if 'selected' in self.Checkbutton_Fine_Tune.state() else False
        self.update_image()
        return
####################################################################################################
    def Checkbutton_Show_Contour_Change(self):
#        print('Checkbutton_Show_Contour_Change')
        self.update_image()
        return
####################################################################################################
    def Trackbar_Change(self, x):
        self.now_frame = x
        self.update_image()
        return
####################################################################################################
    def update_image(self):
#        print('update_image')
        self.image_show = np.array(self.frames_all[self.now_frame], dtype=np.int16)
        if 'selected' in self.Checkbutton_Show_Contour.state():
            contour = self.contour_all[self.now_frame]
            self.image_show[contour>0] = self.image_show[contour>0]+[-32,+32,+32]
            self.image_show[self.image_show>255] = 255
            self.image_show[self.image_show<0] = 0
        self.image_show = np.array(self.image_show, dtype=np.uint8)
        cv2.imshow(self.input_path_name, self.image_show)
        return
####################################################################################################
    def draw_contour(self,event,x,y,flags,param):
#        print('draw_contour')
        if self.fine_tune:
            color = (0,0,255)
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.line(self.contour_all[self.now_frame], (x,y), (x,y), 255, 5)
                self.ix = x
                self.iy = y
                self.update_image()
                self.L_button_down = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.L_button_down = False
            elif self.L_button_down:
                cv2.line(self.contour_all[self.now_frame], (self.ix,self.iy), (x,y), 255, 5)
                self.ix = x
                self.iy = y
                self.update_image()
            elif event == cv2.EVENT_RBUTTONDOWN:
                color = (255,0,0)
                cv2.line(self.contour_all[self.now_frame], (x,y), (x,y), 0, 5)
                self.ix = x
                self.iy = y
                self.update_image()
                self.R_button_down = True
            elif event == cv2.EVENT_RBUTTONUP:
                self.R_button_down = False
            elif self.R_button_down:
                color = (255,0,0)
                cv2.line(self.contour_all[self.now_frame], (self.ix,self.iy), (x,y), 0, 5)
                self.ix = x
                self.iy = y
                self.update_image()
            image_show_temp = np.copy(self.image_show)
            cv2.circle(image_show_temp, (x,y), 5, color, -1)
            cv2.imshow(self.input_path_name, image_show_temp)
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
#                print('cv2.EVENT_LBUTTONDOWN')
                self.contour_x.append(x)
                self.contour_y.append(y)
                cv2.circle(self.image_show,(x,y),4,(255,0,0),-1)
                cv2.imshow(self.input_path_name, self.image_show)
                print('%d [x,y] = [%d,%d]'%(len(self.contour_x),x,y))
                if abs(self.contour_x[0]-x)+abs(self.contour_y[0]-y)<8 and len(self.contour_x)>2 :
                    rr, cc = polygon(self.contour_y[0:-1],self.contour_x[0:-1])
                    self.contour_all[self.now_frame][rr, cc] = 255
                    self.contour_x = []
                    self.contour_y = []
                    self.update_image()
            elif event == cv2.EVENT_RBUTTONDOWN:
#                print('cv2.EVENT_RBUTTONDOWN')
                self.ix = x
                self.iy = y
                self.R_button_down = True
            elif event == cv2.EVENT_RBUTTONUP:
#                print('cv2.EVENT_RBUTTONUP')
                cv2.rectangle(self.contour_all[self.now_frame],(self.ix,self.iy),(x,y),0,-1)
                self.update_image()
                self.R_button_down = False
            elif self.R_button_down:
#                print('self.R_button_down')
                image_show_temp = np.copy(self.image_show)
                cv2.rectangle(image_show_temp,(self.ix,self.iy),(x,y),(255,0,0),1)
                cv2.imshow(self.input_path_name, image_show_temp)
        return
####################################################################################################

####################################################################################################

####################################################################################################

####################################################################################################

root = tk.Tk()

tmp = open('tmp.ico','wb+')
tmp.write(base64.b64decode(img))
tmp.close()
root.iconbitmap('tmp.ico')
os.remove('tmp.ico')

app = Application(root)
root.mainloop()

cv2.destroyAllWindows()
app.frames_all = []
app.contour_all = []
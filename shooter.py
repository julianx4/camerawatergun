import time
import pigpio
import math
import RPi.GPIO as GPIO
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
import cv2
import numpy as np
import sys
import re

pi = pigpio.pi()
GPIO.setmode(GPIO.BCM)
trimhorizontal=7
trimvertical=10

HorizontalServo=17
VerticalServo=27
ShootServo=24

GPIO.setup(ShootServo, GPIO.OUT)
RESX = 640
RESY = 10
XFOV = 54
 
def shoot():

	GPIO.output(ShootServo, GPIO.LOW)
	
def stopshoot():
	GPIO.output(ShootServo, GPIO.HIGH)
 

def ReadLabelFile(file_path):
	with open(file_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	ret = {}
	for line in lines:
		pair = re.split(r'[:\s]+', line.strip(), maxsplit=1)
		ret[int(pair[0])] = pair[1].strip()
	return ret
	

def angleFromPixel(x, y, w, h):
    maxAngle = XFOV / 2 / 180 * math.pi
    xDiff = (w / 2 - x) / w * 2 * math.tan(maxAngle)
    yDiff = (h / 2 - y) / w * 2 * math.tan(maxAngle)
    return -math.atan(xDiff) / math.pi * 180, math.atan(yDiff)/ math.pi * 180
	

def turntoangle(servopin,angle):
	signal=-8.83*angle+1445
	pi.set_servo_pulsewidth(servopin, signal)


def turntotarget(targetx,targety, width ,height):
	targetanglehorizontal,targetanglevertical=angleFromPixel(targetx,targety, width ,height)

	turntoangle(HorizontalServo,targetanglehorizontal+trimhorizontal)
	turntoangle(VerticalServo,targetanglevertical+trimvertical)
	#print(targetanglehorizontal)
	#print(targetanglevertical)


def main():
	detectionengine = DetectionEngine("./mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite")
	#detectionengine = DetectionEngine("./juliandetector.tflite")
	detectionlabels = ReadLabelFile("./coco_labels.txt")
	vs = cv2.VideoCapture(0)
	
	vs.set(cv2.CAP_PROP_FRAME_WIDTH, RESX)
	vs.set(cv2.CAP_PROP_FRAME_HEIGHT, RESY)
	time.sleep(1)
	lastupload=time.time()
	lasttimeshoot=0
	lastclassifylabel=""
	lastdetectionlabel=""
	counter=0
	lastturn=0
	font = cv2.FONT_HERSHEY_SIMPLEX
	bottomLeftCornerOfText = (10,500)
	fontScale = 1
	fontColor = (255,0,0) 
	lineType = 2
	
	LABEL_PERSON = 0
	LABEL_DOG = 17
	
	while 1:
		try:
			anfangzeit = time.time()
			_, frame = vs.read()

			frame = cv2.resize(frame,(int(800),int(480)))
			frame = cv2.flip(frame,-1)
			image = Image.fromarray(frame)
			
			width,height=image.size
			print(width,height)
			
			objs = detectionengine.DetectWithImage(image, threshold=0.1, keep_aspect_ratio=True, relative_coord=True, top_k=1)

			
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break			
			for obj in objs:
				print(obj, obj.label_id)		
				x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
				x0=int(x0*width)
				y0=int(y0*height)
				x1=int(x1*width)
				y1=int(y1*height)
		
			if objs:
				if obj.label_id == LABEL_DOG and obj.score > 0.8:
					#cropimage=image.crop((x0,y0,x1,y1))
					#image.save(str(counter)+"_complete.jpg")
					#cropimage.save(str(counter)+".jpg")
					targetx,targety = int((x0+x1)/2), int(y0 +(y1-y0)/8) - int(200 / (y1-y0) * 160) - 20 
					cv2.rectangle(frame,(x0,y0),(x1,y1),(0,255,0),3)
					cv2.circle(frame, (targetx,targety), 5, (255,255,0), thickness=3, lineType=8, shift=0)
					cv2.circle(frame, (targetx,targety), 2, (0,0,0), thickness=3, lineType=8, shift=0)
					cv2.putText(frame, str(int(obj.score*100)), (targetx+10,targety-20), font, fontScale, fontColor, lineType)
					
					
					if time.time()-lastturn>0.1:
						lastturn=time.time()
						turntotarget(targetx,targety, width,height)
						
					if time.time() - lasttimeshoot>2:
						shoot()
						lasttimeshoot=time.time()
			if not objs or lasttimeshoot + 2 < time.time():
				stopshoot()
				
			endzeit = time.time()
			#print(endzeit - anfangzeit)
			
			cv2.imshow('Shooter',frame)
		except (KeyboardInterrupt):
			cv2.destroyAllWindows()
			sys.exit(0)




if __name__ == '__main__':
	main()


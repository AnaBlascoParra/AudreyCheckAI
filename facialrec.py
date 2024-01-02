import cv2
from deepface import DeepFace
import torch

#paths to ref and base images 
refimg_pth = 'refimg.jpg'
img1_pth = 'img1.jpg'
img2_pth = 'img2.jpg'
img3_pth = 'img3.png'
img4_pth = 'img4.jpg'
img5_pth = 'img5.jpg'
img6_pth = 'img6.png'
img_paths = [img1_pth, img2_pth, img3_pth,img4_pth, img5_pth, img6_pth]

#loads yolov5s model for object detection
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

refimg = cv2.imread(refimg_pth)

#iterates over images to crop detected bounding boxes
for img_path in img_paths:

    img = cv2.imread(img_path)
    height, width = img.shape[:2] #gets h & w

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #preprocesses image
    result = model(img) #inference

    for i, det in enumerate(result.xyxy[0]): #iterates over detections 
        x, y, w, h = det[:4].cpu().numpy().astype(int)
        det_img = img[y:y + h, x:x + w]
        output_path = f"{img_path.split('.')[0]}_det_{i+1}.jpg"  
        cv2.imwrite(output_path, cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))

#checks folder for detection images (cropped bounding boxes)
import os
folder = '.'
det_paths = []
for img in os.listdir(folder):
    if "det" in img: #checks if filename has 'det'and puts it in a list
        det_paths.append(img)

#applies face recognition ai to detections path list
for det_path in det_paths:
    facialcheck = DeepFace.verify(img1_path=refimg_pth,img2_path=det_path,model_name="Facenet",enforce_detection=False) 
    if facialcheck["verified"]:
        print(f"Audrey Hepburn detected in image:"+det_path)


    


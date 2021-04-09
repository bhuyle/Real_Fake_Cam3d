from imutils import face_utils
import pyrealsense2 as rs
import numpy as np
import cv2
import dlib
import time


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# cascPath = 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(cascPath)
padding = 0
# detect_face = dlib.get_frontal_face_detector()
# detect_mouth = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

prev_frame_time = 0


print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    'deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

#Stream loop
try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        #get alignes frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
    
        #render images
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha = 0.05,),cv2.COLORMAP_RAINBOW)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        # rects = detect_face(gray,0)
        frame = color_image.copy()
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
    
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # x,y,x1,y1 = rect.left(), rect.top(), rect.right(), rect.bottom()    
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            # frame = api_face(frame,startX,startY,endX,endY)
            
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            depth_image_c = depth_image[startY:endY,startX:endX]
            try:
                scale = (depth_image_c.max() - depth_image_c.min())
            except ValueError:  #raised if `y` is empty.
                pass
            # print (scale)
            if scale<800:
                frame = cv2.rectangle(frame,(startX,startY),(endX,endY) , (0,0,255),2)
                cv2.putText(frame,"Fake",(startX,startY-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else: 
                frame = cv2.rectangle(frame,(startX,startY),(endX,endY) , (255,0,0),2)
                cv2.putText(frame,"Real",(startX,startY-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = str(int(fps))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(frame, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)


        images = np.hstack((frame,depth_colormap))
        cv2.namedWindow("Align",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Align",images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
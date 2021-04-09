from imutils import face_utils
import pyrealsense2 as rs
import numpy as np
import cv2
import dlib
import time
# pipeline = rs.pipeline()
# config = rs.config()

# config.enable_stream(rs.stream.depth, 648,480,rs.format.z16,30)
# config.enable_stream(rs.stream.color, 648,480,rs.format.bgr8,30)

# #Start stream
# # profile = pipeline.start(config)
# pipeline.start(config)
# #Create an align obj


# import pyrealsense2 as rs
# import numpy as np
# import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

cascPath = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascPath)
padding = 0
detect_face = dlib.get_frontal_face_detector()
detect_mouth = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

prev_frame_time = 0

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
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image,alpha = 0.05,),cv2.COLORMAP_RAINBOW)

        gray = cv2.cvtColor(color_image,cv2.COLOR_BGR2GRAY)
        rects = detect_face(gray,0)


        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps = str(int(fps))
        font = cv2.FONT_HERSHEY_SIMPLEX 
        cv2.putText(color_image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

        for rect in rects:
            x,y,x1,y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
            
            # shape = detect_mouth(gray,rect)
            # print (1,shape)
            # shape = face_utils.shape_to_np(shape)
            # print (2,shape)
            depth_image_c = depth_image[y:y1,x:x1]
            try:
                scale = (depth_image_c.max() - depth_image_c.min())
            except ValueError:  #raised if `y` is empty.
                pass
            print (scale)
            if scale<800:
                color_image = cv2.rectangle(color_image,(x,y),(x1,y1) , (0,0,255),2)
                cv2.putText(color_image,"Fake",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            else: 
                color_image = cv2.rectangle(color_image,(x,y),(x1,y1) , (255,0,0),2)
                cv2.putText(color_image,"Real",(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        images = np.hstack((color_image,depth_colormap))
        cv2.namedWindow("Align",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Face Anti-spoofing",images)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    pipeline.stop()
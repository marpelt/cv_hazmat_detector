#!/usr/bin/env python3

import rospy
import rospkg
import os
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import torch

from cv_msg.msg import CV_msg, Hazmat_Detection, BBox

class ImageProcessor:
    def __init__(self):
        self.bridge = CvBridge()

        model_file = rospy.get_param('~model_file', 'hazmat.pt')
        self.min_confidence = rospy.get_param('~min_confidence', 75)
        device_type = rospy.get_param('~device', 'cpu')
        cam_input_topic = rospy.get_param('~camera_topic', '/screen/camera/image_raw/compressed')
        self.camera_id = rospy.get_param('~camera_id', 0)
        self.cv_msg_topic = rospy.get_param('~cv_msg_topic', '/cv_bundle')

        self.cv_pub = rospy.Publisher(self.cv_msg_topic, CV_msg, queue_size=1)

        rospack = rospkg.RosPack()
        try:
            package_path = rospack.get_path('cv_hazmat_detector')
            model_path = os.path.join(package_path, 'models', model_file)
            
            self.model = YOLO(model_path)
            device = 'cuda' if device_type == 'cuda' and torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            rospy.loginfo(f"Modell auf {device}: {model_path}")
        except Exception as e:
            rospy.logerr(f"Fehler des Modells: {e}")
            return

        self.image_sub = rospy.Subscriber(cam_input_topic, CompressedImage, self.image_callback)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            img_h, img_w = cv_image.shape[:2]
            results = self.model(cv_image, verbose=False)

            bundle_msg = CV_msg()
            bundle_msg.header = msg.header
            bundle_msg.camera_id = self.camera_id

            hazmat_detections = []

            for result in results:
                for box in result.boxes:
                    confidence = box.conf[0].item() * 100 
                    
                    if confidence > self.min_confidence:
                        box_data = box.xywh[0].tolist()
                        
                        det = Hazmat_Detection()
                        det.content = result.names[int(box.cls[0].item())]
                        
                        det.bbox.cx = box_data[0] / img_w
                        det.bbox.cy = box_data[1] / img_h
                        det.bbox.width = box_data[2] / img_w
                        det.bbox.height = box_data[3] / img_h
                        
                        hazmat_detections.append(det)

            bundle_msg.hazmat_detections = hazmat_detections

            if hazmat_detections:
                self.cv_pub.publish(bundle_msg)

        except CvBridgeError as e:
            rospy.logerr(f'CvBridge Error: {e}')

def main():
    rospy.init_node('hazmat_detector_node', anonymous=True)
    process = ImageProcessor()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down hazmat detector node.")

if __name__ == '__main__':
    main()
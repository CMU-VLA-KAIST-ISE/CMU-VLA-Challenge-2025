#!/usr/bin/env python3
import rospy, numpy as np, io
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from PIL import Image as PILImage

PATH = "/home/aailab/cwh316/CMU-VLA-Challenge/ai_module/src/gemini_api/0000.png"
TOPIC = "/partial_sg_generator/current_image"

def pil_to_rosimg(pil_img, frame_id="camera_color_optical_frame"):
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    arr = np.array(pil_img)  # (H,W,3) RGB uint8
    msg = Image()
    msg.header = Header(stamp=rospy.Time.now(), frame_id=frame_id)
    msg.height = h
    msg.width = w
    msg.encoding = "rgb8"
    msg.is_bigendian = 0
    msg.step = w * 3
    msg.data = arr.tobytes()
    return msg

if __name__ == "__main__":
    rospy.init_node("temp_image_publisher_no_cvbridge")
    pub = rospy.Publisher(TOPIC, Image, queue_size=1, latch=True)

    with open(PATH, "rb") as f:
        pil = PILImage.open(io.BytesIO(f.read()))
        pil.load()

    msg = pil_to_rosimg(pil)
    rate = rospy.Rate(1)  # 1 Hz
    rospy.loginfo(f"Publishing {PATH} to {TOPIC} (rgb8)")
    while not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()  # 타임스탬프 갱신
        pub.publish(msg)
        rate.sleep()
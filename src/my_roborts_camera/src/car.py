#! /usr/bin/python3
import torch.backends.cudnn as cudnn
import cv2
import numpy as np
import platform
import mvsdk
import rospy
from models.experimental import *
from utils.datasets import *
from utils.utils import *
from my_roborts_camera.msg import car_position

# mainly parameter
########################################################################
# car position
car_x = 0
car_y = 0

# the size field, see in the readme
field_x = 224
field_y0 = 390
field_y1 = 360

# show img
show_image = True


################################################################################
# The field area
left_top = (964, 94)
right_top = (1237, 136)
left_top_list = [left_top[0], left_top[1]]
Right_top_list = [right_top[0], right_top[1]]

left_bottom = (178, 386)
right_bottom = (811, 719)
left_bottom_list = [left_bottom[0], left_bottom[1]]
right_bottom_list = [right_bottom[0], right_bottom[1]]
List_pst = [left_top_list, Right_top_list, left_bottom_list, right_bottom_list]
################################################################################
# Get the image shape
Image_W = 1280
Image_H = 720

# parameter for transform
pst1 = np.float32(List_pst)
pst2 = np.float32([[0, 0], [1280, 0], [0, 720], [1280, 720]])

# Calculate the transform matrix
M = cv2.getPerspectiveTransform(pst1, pst2)

# Set the size to inference
model_size = (256, 256)

# set the model path and select cpu/gpu
# weights = 'runs/Jetson_exp12/weights/Innocent_Bird.pt'
# weights = 'runs/exp22/weights/best.pt'
weights = '/home/xutao/.local/lib/python3.6/site-packages/runs/exp23/weights/best.pt'  # 路径要改
device = torch_utils.select_device()

# Load model  and  check image size
model = attempt_load(weights, map_location=device)
image_size = check_img_size(model_size[0], s=model.stride.max())
model.half()

# speed up constant image size inference
cudnn.benchmark = True

# Get names and rectangle color
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# ROS节点初始化
rospy.init_node('Bird_node', anonymous=True)
# 创建一个Publisher，发布名为/car_position_info的topic，消息类型为car_position，队列长度10
position_info_pub = rospy.Publisher('/car_position_info', car_position, queue_size=10)


# resize input image for preparing inference
def letterbox(img, new_shape=(256, 256), color=(114, 114, 114), auto=True, scale_up=True):
    # Resize image to a 32*x rectangle
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new_W / old_w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down to get better mAP
    if not scale_up:
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # minimum rectangle
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding

    # divide padding into 2 sides
    dw /= 2
    dh /= 2
    # to resize image
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


# object detection
def detect(Bird_img):
    global car_x
    global car_y
    global field_x
    global field_y0
    global field_y1
    global position_info_pub
    global colors

    # create a car_armor_position msg class
    Car_Position_Msgs = car_position()
    # MSG_Init，clc per period
    Car_Position_Msgs.name = 'Tom'
    Car_Position_Msgs.color = 0
    Car_Position_Msgs.x = 0
    Car_Position_Msgs.y = 0

    boxed_image = letterbox(Bird_img, model_size)
    # Stack
    image_data = np.stack(boxed_image, 0)

    # Convert, BGR to RGB, bsx3x416x416
    image_data = image_data[:, :, ::-1].transpose(2, 0, 1)
    image_data = np.ascontiguousarray(image_data)

    image_data = torch.from_numpy(image_data).to(device)
    # u8 to fp16/32
    image_data = image_data.half()
    # from 0~255 to 0.0~1.0
    image_data /= 255.0
    if image_data.ndimension() == 3:
        image_data = image_data.unsqueeze(0)

    # Inference
    t1 = torch_utils.time_synchronized()

    predict = model(image_data, augment=False)[0]
    # Apply NMS
    predict = non_max_suppression(predict, 0.6, 0.5, classes=0, agnostic=False)

    t2 = torch_utils.time_synchronized()

    print("Inference Time:", t2 - t1)

    # Process detections
    for i, det in enumerate(predict):
        s = '%g:' % i
        s += '%gx%g ' % image_data.shape[2:]

        labels_list = []
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(image_data.shape[2:], det[:, :4], Bird_img.shape).round()

            # Print results
            for c in det[:, -1].detach().unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, names[int(c)])  # add to string
            # Write results
            for *xy, conf, cls in det:
                if show_image:  # Add bbox to image
                    label = '%s %.2f' % (names[int(cls)], conf)
                    labels_list.append(label)
                    color = colors[int(cls)]
                    # draw rect and put text
                    if label.startswith('car_red') or label.startswith('car_blue'):
                        pst1, pst2 = (int(xy[0]), int(xy[1])), (int(xy[2]), int(xy[3]))
                        Car_Center = ((pst1[0]+pst2[0])/2, (pst1[1]+pst2[1])/2)

                        # *****************************mainly***************************
                        if ref_flag == 0:
                            adjust_r = field_y1 / field_y0
                            if Car_Center[1] < ref_point[1]:
                                # y0那边的区域
                                car_y = ((ref_point[1] - Car_Center[1]) / ref_point[1]) * field_y0 * adjust_r
                                car_x = ((Car_Center[0] - ref_point[0]) / Bird_img.shape[1]) * field_x * adjust_r
                            elif Car_Center[1] > ref_point[1]:
                                # y1这边的区域
                                car_x = ((Car_Center[0] - ref_point[0]) / Bird_img.shape[1]) * field_x * adjust_r
                                car_y = ((Car_Center[1] - ref_point[1]) / (Bird_img.shape[0] - ref_point[1])) * (-field_y1) * adjust_r

                            print("car position : ", "x=%d" % car_x, "y=%d" % car_y)
                            cv2.putText(Bird_img, "(%d, %d)" % (car_x, car_y), (int(xy[2]), int(xy[3])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

                            if label.startswith('car_red'):
                                color = (0, 0, 255)
                                Car_Position_Msgs.color = car_position.red
                            else:
                                color = (255, 0, 0)
                                Car_Position_Msgs.color = car_position.blue
                            Car_Position_Msgs.x = car_x
                            Car_Position_Msgs.y = car_y
                       

                        if conf > 0.55:
                            plot_one_box(xy, Bird_img, label=label, color=color, line_thickness=2)
                        else:
                            pass
                    else:
                        plot_one_box(xy, Bird_img, label=label, color=color, line_thickness=2)
        # 发布消息
        position_info_pub.publish(Car_Position_Msgs)


# about reference point
ref_flag = 1
ref_point = (0, 0)
ref_point_t = (0, 0)


# for calibration, print the coordinate
def mouse_event(event, x, y, flags, param):
    global ref_flag
    global ref_point
    global ref_point_t
    if event == cv2.EVENT_LBUTTONDOWN:
        if ref_flag == 1:
            ref_flag = 2
            ref_point_t = (x, y)
            print("Point:", ref_point_t)
        else:
            temp_point = (x, y)
            # print the point clicked
            print("Point", temp_point)
            # Double click mouse to set the reference point
            if abs((ref_point_t[0]-temp_point[0]) < 3) and (abs(ref_point_t[1]-temp_point[1]) < 3) and (ref_flag == 2):
                ref_point = (round((ref_point_t[0]+temp_point[0])/2), round((ref_point_t[1]+temp_point[1])/2))
                print("Reference P:", ref_point)
                ref_flag = 0


class Camera(object):
    def __init__(self, DevInfo):
        super(Camera, self).__init__()
        self.DevInfo = DevInfo
        self.hCamera = 0
        self.cap = None
        self.pFrameBuffer = 0

    def open(self):
        if self.hCamera > 0:
            return True

        # 打开相机
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return False

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        mvsdk.CameraReadParameterFromFile(hCamera, '/home/xutao/.local/lib/python3.6/site-packages/Camera/Configs/MV-SUA133GC-Group0.config')
        print('yes')

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(hCamera)

        self.hCamera = hCamera
        self.pFrameBuffer = pFrameBuffer
        self.cap = cap
        return True

    def close(self):
        if self.hCamera > 0:
            mvsdk.CameraUnInit(self.hCamera)
            self.hCamera = 0

        mvsdk.CameraAlignFree(self.pFrameBuffer)
        self.pFrameBuffer = 0

    def grab(self):
        # 从相机取一帧图片
        hCamera = self.hCamera
        pFrameBuffer = self.pFrameBuffer
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(hCamera, 200)
            mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            # if platform.system() == "Windows":
            #    mvsdk.CameraFlipFrameBuffer(pFrameBuffer, FrameHead, 1)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            return frame
        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))
            return None


def main_loop():
    # 枚举相机
    count = 0
    number = 0
    DevList = mvsdk.CameraEnumerateDevice()
    nDev = len(DevList)
    if nDev < 1:
        print("No camera was found!")
        return

    for i, DevInfo in enumerate(DevList):
        print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))

    cams = []
    for i in map(lambda x: int(x), input("Select cameras: ").split()):
        cam = Camera(DevList[i])
        if cam.open():
            cams.append(cam)

    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        for cam in cams:
            frame = cam.grab()
            if frame is not None:
                image = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

                cv2.circle(image, left_top, 2, (0, 255, 0), thickness=2)
                cv2.circle(image, right_top, 2, (0, 255, 0), thickness=2)
                cv2.circle(image, left_bottom, 2, (0, 255, 0), thickness=2)
                cv2.circle(image, right_bottom, 2, (0, 255, 0), thickness=2)
                
                cv2.rectangle(image, (left_top[0] + 4, left_top[1] + 4), (left_top[0] - 4, left_top[1] - 4),
                              (0, 0, 255), thickness=2)
                cv2.rectangle(image, (right_top[0] + 4, right_top[1] + 4), (right_top[0] - 4, right_top[1] - 4),
                              (0, 0, 255), thickness=2)
                cv2.rectangle(image, (left_bottom[0] + 4, left_bottom[1] + 4), (left_bottom[0] - 4, left_bottom[1] - 4),
                              (0, 0, 255), thickness=2)
                cv2.rectangle(image, (right_bottom[0] + 4, right_bottom[1] + 4),
                              (right_bottom[0] - 4, right_bottom[1] - 4), (0, 0, 255), thickness=2)

                cv2.imshow("{} Press q to end".format(cam.DevInfo.GetFriendlyName()), image)

                Bird_img = cv2.warpPerspective(image, M, (Image_W, Image_H))
                Bird_img = cv2.resize(Bird_img, (720, 900))

                cv2.circle(Bird_img, ref_point, 2, (0, 99, 99), thickness=2)
                cv2.rectangle(Bird_img, (ref_point[0] + 4, ref_point[1] + 4), (ref_point[0] - 4, ref_point[1] - 4),
                              (0, 255, 0), thickness=2)

                # # detection
                detect(Bird_img)
                cv2.imshow("Result", Bird_img)
                cv2.setMouseCallback("Result", mouse_event)

    for cam in cams:
        cam.close()

def main():
    try:
        main_loop()
    finally:
        cv2.destroyAllWindows()


main()


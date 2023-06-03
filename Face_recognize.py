import cv2
import os
import numpy as np

import sys


def progress_bar(finish_tasks_number, tasks_number):
    percentage = round(finish_tasks_number / tasks_number * 100)
    if percentage <= 100:
        print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), end="")
    else:
        print("正在处理音频，请稍后······")
    sys.stdout.flush()


def getFileSize(video_path):
    fsize = os.path.getsize(video_path)
    return fsize


# 皮肤光滑处理
def smooth_skin(image):
    # 使用高斯滤波器平滑皮肤
    smoothed = cv2.GaussianBlur(image, (0, 0), 5)
    # 使用高斯双边滤波器保留边缘细节
    skin_smoothed = cv2.addWeighted(image, 1.5, smoothed, -0.5, 0)
    return skin_smoothed

# 亮度调整
def adjust_brightness(image, brightness):
    # 调整亮度
    adjusted = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    return adjusted

# 磨皮处理
def skin_smoothing(image, kernel_size=15, strength=3):
    # 双边滤波器参数
    d = kernel_size * 2 + 1
    sigma_color = 10
    sigma_space = 10

    # 高斯双边滤波器
    smoothed = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    # 磨皮效果
    mask = cv2.subtract(smoothed, image)
    mask = cv2.add(mask, (strength, strength, strength, 0))
    skin_smoothed = cv2.add(image, mask)

    return skin_smoothed


def beautify_face(img):
    image = smooth_skin(img)
    image = adjust_brightness(image,1.2)
    image = skin_smoothing(image)
    return image

def change_face(faces,image):
    # for face in faces:
        # 获取人脸区域的边界框
    (x, y, w, h) = faces

    # 提取人脸区域
    face_image = image[y:y+h, x:x+w]


    # 对人脸进行美颜处理
    beautified_face = beautify_face(face_image)

    # 将美颜处理后的人脸替换回原图像
    image[y:y+h, x:x+w] = beautified_face
        # return image[y:y+h, x:x+w]


def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #加载OpenCV人脸检测分类器Haar
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if (len(faces) == 0):
        return None, None
    (x, y, w, h) = faces[0]
    return gray[y:y + w, x:x + h], faces[0] 
def prepare_training_data(data_folder_path):
    dirs =  os.listdir(data_folder_path)
    # 两个列表分别保存所有的脸部和标签
    faces = []
    labels = []
    # 浏览每个目录并访问其中的图像
    for dir_name in dirs:
    # dir_name(str类型)即标签
        # print(dir_name)
        label = int(dir_name)
    # 建立包含当前主题主题图像的目录路径
        subject_dir_path = data_folder_path + "/" + dir_name
    # 获取给定主题目录内的图像名称
        subject_images_names = os.listdir(subject_dir_path)
    # 浏览每张图片并检测脸部，然后将脸部信息添加到脸部列表faces[]
        for image_name in subject_images_names:
    # 建立图像路径
            image_path = subject_dir_path + "/" + image_name
    # 读取图像
            image = cv2.imread(image_path)
    # 显示图像0.1s
        # cv2.imshow("Training on image...", image)
        # cv2.waitKey(100)
    # 检测脸部
            face, rect = detect_face(image)
    # 忽略未检测到的脸部
        if face is not None:
    #将脸添加到脸部列表并添加相应的标签
            faces.append(face)
            labels.append(label)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
    return faces, labels
faces,labels = prepare_training_data("test_data")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (128, 128, 0), 2)
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 128, 0), 2)

subjects = ["gangtiexia", "xiaolajiao"]
def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    if face is None:
        return img
    label = face_recognizer.predict(face)
    label_text = subjects[label[0]]

    draw_rectangle(img, rect)
    change_face(rect,img)
    draw_text(img, label_text, rect[0], rect[1] - 5)
    return img

def get_audio(video_path,audio_path):
    os.system('ffmpeg -i '+ video_path +' -vn -codec copy ' + audio_path)

def merge_video(video_path,audio_path,final_path):
    os.system('ffmpeg -i '+ video_path + ' -i '+ audio_path+ ' -c:v copy -c:a copy -bsf:a aac_adtstoasc' + final_path)  #converting the image files to mp4 format using FFMPEG. Note that the "-y" option means "overwrite output file if it already exists." -codec copy means that the output file will have the same codec as the input files.

def change_video(ori_path,output_path):
    os.system('ffmpeg -i '+ ori_path+' -af "asetrate=44100, rubberband=pitch=2:tempo=1:formant=1, aresample=44100" '+ output_path)

video_path = "./video/video.mp4"
output_path = "./video/VideoWithFace.mp4"
audio_path = "./audio/audio.m4a"

# 打开视频文件
video_capture = cv2.VideoCapture(video_path)

# 创建输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 加载人像检测模型
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
    # 检测人像ß
    predicted_img = predict(frame)

    # 将帧写入输出视频
    output_video.write(predicted_img)
    ori_time = getFileSize(video_path)
    new_time = getFileSize(output_path)
    # print(ori_time,new_time)
    if new_time == -1:
        new_time = 0
    progress_bar(new_time,ori_time)


# 释放资源
video_capture.release()
output_video.release()

get_audio(video_path, audio_path)

beautyFace_path = "./video/VideoWithbeautifulFace.mp4"
merge_video(output_path,audio_path,beautyFace_path)

final_path = "./video/finalVideo.mp4"
change_video(beautyFace_path,final_path)


cv2.destroyAllWindows()

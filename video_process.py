import os
import cv2


def extract_frame(video_path):
    # 定义输出文件名
    filename, file_extension = os.path.splitext(os.path.basename(video_path))
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否打开成功
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    # 读取第一帧
    count = 0
    save_dir = video_path.replace(file_extension,"")
    os.makedirs(save_dir,exist_ok=True)
    while True:
        ret, frame = cap.read()
        if ret:
            if count%30==0:
                # 保存第一帧为 .jpg 文件
                output_image_path = os.path.join(save_dir,f"_{count}.jpg")
                cv2.imwrite(output_image_path, frame)
                print(f"第{count}帧已保存为 {output_image_path}")
        else:
            break
        count += 1
    # 释放视频捕获对象
    cap.release()

if __name__ == '__main__':
    video_dir = r"D:\Datasets\Park"
    for video_name in os.listdir(video_dir):
        video_path = os.path.join(video_dir,video_name)
        extract_frame(video_path)
    print("done")
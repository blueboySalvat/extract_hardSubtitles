import os
import cv2
from paddleocr import PaddleOCR
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time

# 配置日志级别，减少不必要的输出
logging.basicConfig(level=logging.INFO)
for logger_name in ['ppocr', 'paddle', 'matplotlib']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

def save_frame(frame, frame_path, crop_height):
    """
    保存帧并裁剪字幕区域。
    
    Args:
        frame: 视频帧数据
        frame_path: 保存路径
        crop_height: 裁剪区域的高度范围 (tuple: (start, end))
    """
    # 获取帧的高度和宽度
    h = frame.shape[0]
    # 定义字幕区域：裁剪底部 15% 到 20% 的高度
    subtitle_region = frame[int(h * crop_height[0]):int(h * crop_height[1]), :]
    # 保存裁剪后的帧到输出目录，使用零填充的编号命名
    cv2.imwrite(frame_path, subtitle_region)

def is_similar_frame(frame1, frame2, threshold=8):
    """判断两帧是否相似"""
    # 转为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    # 计算两帧的绝对差
    diff = cv2.absdiff(gray1, gray2)
    # 计算差异的平均值
    mean_diff = np.mean(diff)
    return mean_diff < threshold

def extract_frames(video_path, output_dir, frame_rate=2):
    """提取视频帧
    
    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        frame_rate: 每秒提取的帧数
    """
    # 如果输出目录不存在，则创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 使用 OpenCV 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    # 获取视频的帧率（FPS）
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # 计算帧间隔：从视频中每隔 frame_interval 帧提取一帧
    frame_interval = fps // frame_rate
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化帧计数器
    frame_count = 0  # 当前处理的总帧数
    saved_count = 0  # 成功保存的帧数
    
    # 定义裁剪区域为视频底部 15% 到 20%
    crop_height = (0.8, 0.95)
    tasks = []

    # 使用线程池处理帧保存
    with ThreadPoolExecutor(max_workers=4) as executor:
        # 创建一个进度条，显示帧提取的进度
        with tqdm(total=total_frames, desc="提取视频帧") as pbar:
            prev_frame = None
            while cap.isOpened():
                # 逐帧读取视频
                ret, frame = cap.read()
                # 如果无法读取帧（例如视频结束），退出循环
                if not ret:
                    break

                # 仅处理符合提取间隔的帧
                if frame_count % frame_interval == 0:
                    # 构建帧图片的路径
                    frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                    # 前一帧为空，或者当前帧与上一帧重复度 < 时，则继续
                    if prev_frame is None or not is_similar_frame(prev_frame, frame):
                        # 提交保存任务给线程池
                        tasks.append(executor.submit(save_frame, frame, frame_path, crop_height))
                        saved_count += 1
                        prev_frame = frame

                # 增加当前帧计数器
                frame_count += 1
                # 更新进度条
                pbar.update(1)

        # 确保所有任务完成
        for task in tasks:
            task.result()

    cap.release()
    return saved_count

def extract_subtitles_from_frames(frame_dir, ocr):
    """从帧中提取字幕文本"""

    # 初始化字幕列表和前一帧的字幕文本
    subtitles = []
    prev_text = None
    
    # 获取帧文件列表，并按编号排序
    frame_files = sorted(os.listdir(frame_dir), 
                        key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # 添加进度条，显示字幕识别的进度
    for frame_file in tqdm(frame_files, desc="识别字幕"):
        frame_path = os.path.join(frame_dir, frame_file)# 构造完整帧路径
        
        try:
            # 使用 OCR 引擎识别帧中的文本
            results = ocr.ocr(frame_path)
            # 如果识别结果为空或格式不符合预期，跳过该帧
            if not results or not results[0]:
                continue
                
            # 提取当前帧的所有识别文本
            current_texts = []
            for line in results[0]:
                if line and len(line) >= 2: # 确保每行数据格式正确
                    text, confidence = line[1] # 提取文本和置信度
                    if confidence > 0.6:  ## 仅保留置信度高于 0.6 的文本
                        current_texts.append(text.strip())
            
            # 如果没有符合条件的文本，跳过该帧
            if not current_texts:
                continue
                
            # 将当前帧的所有文本合并为单个字符串
            current_text = ' '.join(current_texts)
            
            # 清理文本，去掉多余的换行符和空格
            current_text = current_text.replace('\n', ' ').strip()
            
            # 检查与上一帧文本的相似度，避免重复字幕
            if prev_text and current_text:
                # 计算文本相似度
                similarity = len(set(current_text) & set(prev_text)) / max(len(set(current_text)), len(set(prev_text)))
                if similarity > 0.8:  # 如果相似度超过 0.8，认为是重复字幕，跳过
                    continue

            # 如果当前文本有效，添加到字幕列表，并更新 prev_text
            if current_text:
                subtitles.append(current_text)
                prev_text = current_text
                
        # 记录处理帧时的错误信息，便于排查问题
        except Exception as e:
            logging.error(f"处理帧 {frame_file} 时出错: {str(e)}")
            continue
         
    # 返回提取的字幕列表   
    return subtitles

def display_summary_statistics(video_count, total_duration, total_subtitles, total_time):
    """在命令行显示总体统计信息"""
    print("\n" + "=" * 50)
    print("统计信息总结".center(48))
    print("=" * 50)
    print(f"作者: 忧郁男孩的救赎")
    print(f"网站: blueboySalvat.top")
    print("-" * 50)
    print(f"处理视频总数: {video_count} 个")
    print(f"视频总时长: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)")
    print(f"识别字幕总数: {total_subtitles} 条")
    print(f"处理总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"平均处理速度: {total_duration/total_time:.2f}x")
    print("=" * 50 + "\n")

def process_videos(input_folder):
    """处理指定文件夹中的所有视频"""
    output_txt = os.path.join(input_folder, "output.txt")
    start_time = time.time()
    total_duration = 0
    total_subtitles = 0
    
    # 初始化 PaddleOCR，优化参数配置
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='ch',
        use_gpu=False,
        show_log=False,
        rec_thresh=0.6,  # 识别置信度阈值
    )
    
    with open(output_txt, 'w', encoding='utf-8') as output_file:
        video_files = [f for f in sorted(os.listdir(input_folder)) 
                      if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        
        for video_file in video_files:
            video_path = os.path.join(input_folder, video_file)
            frame_dir = os.path.join(input_folder, "frames")
            
            print(f"\n处理视频: {video_file}")
            
            try:
                # 获取视频时长
                with VideoFileClip(video_path) as video:
                    duration = video.duration
                    total_duration += duration

                # 提取帧
                extract_frames(video_path, frame_dir)
                
                # 识别字幕
                subtitles = extract_subtitles_from_frames(frame_dir, ocr)
                total_subtitles += len(subtitles)
                
                # 写入结果
                output_file.write(f"{video_file}\n")
                for subtitle in subtitles:
                    output_file.write(f"{subtitle}\n")
                output_file.write("\n")
                
                # 清理临时文件
                for frame_file in os.listdir(frame_dir):
                    os.remove(os.path.join(frame_dir, frame_file))
                os.rmdir(frame_dir)
                
            except Exception as e:
                logging.error(f"处理视频 {video_file} 时出错: {str(e)}")
                continue

    # 显示总体统计信息
        total_time = time.time() - start_time
        display_summary_statistics(
            len(video_files),
            total_duration,
            total_subtitles,
            total_time
        )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从视频中提取字幕")
    parser.add_argument("input_folder", help="包含视频文件的文件夹路径")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        print(f"错误：文件夹 '{args.input_folder}' 不存在")
        exit(1)
        
    process_videos(args.input_folder)
import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from paddleocr import PaddleOCR
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import json
import threading

# 配置日志级别，减少不必要的输出
logging.basicConfig(level=logging.INFO)
for logger_name in ['ppocr', 'paddle', 'matplotlib']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频字幕提取器")
        self.root.geometry("800x600")

        # 配置根窗口的网格权重，使其可以随窗口调整大小
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 初始化配置
        self.config = {
            'filter_words': [],
            'crop_top': 0.8,
            'crop_bottom': 0.95
        }
        self.load_config()

        self.create_gui()

    def create_gui(self):
        """创建图形界面"""
        # 创建主框架并配置其网格权重
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(1, weight=1)  # 使中间列可以扩展

        # 文件夹选择区域
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        folder_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(folder_frame, text="输入文件夹:").grid(row=0, column=0, padx=(0, 5))
        self.folder_path = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_path).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(folder_frame, text="浏览", command=self.browse_folder).grid(row=0, column=2, padx=(5, 0))

        # 过滤词设置区域
        filter_frame = ttk.Frame(main_frame)
        filter_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        filter_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(filter_frame, text="过滤词(用逗号分隔):").grid(row=0, column=0, padx=(0, 5))
        self.filter_words = tk.StringVar(value=','.join(self.config['filter_words']))
        ttk.Entry(filter_frame, textvariable=self.filter_words).grid(row=0, column=1, sticky=(tk.W, tk.E))

        # 裁剪区域设置框
        crop_frame = ttk.LabelFrame(main_frame, text="裁剪区域设置", padding="10")
        crop_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        crop_frame.grid_columnconfigure(1, weight=1)
        crop_frame.grid_columnconfigure(3, weight=1)

        ttk.Label(crop_frame, text="上边界 (0-1):").grid(row=0, column=0, padx=(0, 5))
        self.crop_top = tk.DoubleVar(value=self.config['crop_top'])
        ttk.Entry(crop_frame, textvariable=self.crop_top, width=10).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(crop_frame, text="下边界 (0-1):").grid(row=0, column=2, padx=(20, 5))
        self.crop_bottom = tk.DoubleVar(value=self.config['crop_bottom'])
        ttk.Entry(crop_frame, textvariable=self.crop_bottom, width=10).grid(row=0, column=3, sticky=tk.W)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))

        # 日志显示区域
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 开始处理按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(button_frame, text="开始处理", command=self.start_processing, width=20).pack()

        # 配置主框架的行权重
        main_frame.grid_rowconfigure(4, weight=1)  # 让日志区域可以垂直扩展
        
    def browse_folder(self):
        """选择输入文件夹"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)
            
    def load_config(self):
        """加载配置文件"""
        config_path = 'config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config.update(json.load(f))
                
    def save_config(self):
        """保存配置到文件"""
        self.config['filter_words'] = [word.strip() for word in self.filter_words.get().split(',') if word.strip()]
        self.config['crop_top'] = self.crop_top.get()
        self.config['crop_bottom'] = self.crop_bottom.get()
        
        with open('config.json', 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
            
    def log(self, message):
        """显示日志信息"""
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END)
        
    def save_frame(self, frame, frame_path, crop_height):
        """
        保存帧并裁剪字幕区域。
        
        Args:
            frame: 视频帧数据
            frame_path: 保存路径
            crop_height: 裁剪区域的高度范围 (tuple: (start, end))
        """
        h = frame.shape[0]
        subtitle_region = frame[int(h * crop_height[0]):int(h * crop_height[1]), :]
        cv2.imwrite(frame_path, subtitle_region)
        
    def is_similar_frame(self, frame1, frame2, threshold=8):
        """判断两帧是否相似"""
        # 转为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # 计算两帧的绝对差
        diff = cv2.absdiff(gray1, gray2)
        # 计算差异的平均值
        mean_diff = np.mean(diff)
        return mean_diff < threshold

    def extract_frames(self, video_path, output_dir):
        """提取视频帧
        
        Args:
            video_path: 视频文件路径
            output_dir: 输出目录
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
        frame_interval = fps // 2  # 每秒2帧
        # 获取视频的总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化帧计数器
        frame_count = 0
        saved_count = 0
        crop_height = (self.crop_top.get(), self.crop_bottom.get())
        tasks = []

        self.update_log("开始提取视频帧...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 创建帧提取进度条
            frame_progress = 0
            prev_frame = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                    if prev_frame is None or not self.is_similar_frame(prev_frame, frame):
                        tasks.append(executor.submit(self.save_frame, frame, frame_path, crop_height))
                        saved_count += 1
                        prev_frame = frame
                    
                    # 更新提取进度
                    #new_progress = int((frame_count / total_frames) * 100)
                    #if new_progress != frame_progress:
                        #frame_progress = new_progress
                        #self.update_log(f"\r提取帧进度: {frame_progress}%")

                frame_count += 1

            cap.release()
            
        self.update_log(f"视频帧提取完成，共保存 {saved_count} 帧")
        return saved_count

    def extract_subtitles_from_frames(self, frame_dir, ocr):
        """从帧中提取字幕文本"""
        subtitles = []
        prev_text = None
        
        frame_files = sorted(os.listdir(frame_dir), 
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        total_frames = len(frame_files)
        self.update_log(f"\n开始识别字幕，共 {total_frames} 帧...")
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frame_dir, frame_file)
            
            try:
                # 更新识别进度
                if i % 60 == 0:  # 每5帧更新一次进度，避免刷新太频繁
                    progress = (i + 1) / total_frames * 100
                    self.update_log(f"\r字幕识别进度: {progress:.1f}%")
                
                results = ocr.ocr(frame_path)
                if not results or not results[0]:
                    continue
                    
                current_texts = []
                for line in results[0]:
                    if line and len(line) >= 2:
                        text, confidence = line[1]
                        if confidence > 0.6:
                            current_texts.append(text.strip())
                
                if not current_texts:
                    continue
                    
                current_text = ' '.join(current_texts)
                current_text = current_text.replace('\n', ' ').strip()
                
                if prev_text and current_text:
                    similarity = len(set(current_text) & set(prev_text)) / max(len(set(current_text)), len(set(prev_text)))
                    if similarity > 0.8:
                        continue

                if current_text:
                    subtitles.append(current_text)
                    prev_text = current_text
                    
            except Exception as e:
                self.update_log(f"\n处理帧 {frame_file} 时出错: {str(e)}")
                continue
        
        self.update_log(f"\n字幕识别完成，共识别出 {len(subtitles)} 条字幕")
        return subtitles

    def filter_subtitles(self, subtitles):
        """过滤字幕中包含指定关键词的内容"""
        filter_words = [word.strip() for word in self.filter_words.get().split(',') if word.strip()]
        if not filter_words:
            return subtitles
            
        filtered = []
        for subtitle in subtitles:
            should_filter = False
            for word in filter_words:
                if word in subtitle:
                    should_filter = True
                    break
            if not should_filter:
                filtered.append(subtitle)
        return filtered
        
    def start_processing(self):
        """开始处理视频"""
        self.save_config()
        folder_path = self.folder_path.get()
        if not folder_path:
            self.log("请选择输入文件夹")
            return
            
        # 禁用开始按钮，避免重复点击
        for child in self.root.winfo_children():
            if isinstance(child, ttk.Frame):
                for widget in child.winfo_children():
                    if isinstance(widget, ttk.Button):
                        widget.configure(state='disabled')
        
        # 创建并启动处理线程
        processing_thread = threading.Thread(
            target=self.process_videos,
            args=(folder_path,),
            daemon=True
        )
        processing_thread.start()
        
        # 启动周期性检查线程状态
        self.check_thread_status(processing_thread)
    
    def check_thread_status(self, thread):
        """检查处理线程的状态并更新GUI"""
        if thread.is_alive():
            # 线程仍在运行，100ms后再次检查
            self.root.after(100, lambda: self.check_thread_status(thread))
        else:
            # 线程结束，重新启用按钮
            for child in self.root.winfo_children():
                if isinstance(child, ttk.Frame):
                    for widget in child.winfo_children():
                        if isinstance(widget, ttk.Button):
                            widget.configure(state='normal')
            self.log("\n处理完成！")

    def update_progress(self, value):
        """从线程安全地更新进度条"""
        self.root.after(0, lambda: self.progress.configure(value=value))

    def update_log(self, message):
        """从线程安全地更新日志"""
        self.root.after(0, lambda: self.log(message))

    def process_videos(self, input_folder):
        """处理指定文件夹中的所有视频"""
        try:
            output_txt = os.path.join(input_folder, "output.txt")
            start_time = time.time()
            total_duration = 0
            total_subtitles = 0
            
            self.update_log("初始化 OCR 引擎...")
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=False,
                show_log=False,
                rec_thresh=0.6,
            )
            
            with open(output_txt, 'w', encoding='utf-8') as output_file:
                video_files = [f for f in sorted(os.listdir(input_folder)) 
                              if f.endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv'))]
                
                self.root.after(0, lambda: self.progress.configure(maximum=len(video_files)))
                
                for i, video_file in enumerate(video_files):
                    video_path = os.path.join(input_folder, video_file)
                    frame_dir = os.path.join(input_folder, "frames")
                    
                    self.update_log(f"\n处理视频 ({i+1}/{len(video_files)}): {video_file}")
                    
                    try:
                        self.update_log("获取视频信息...")
                        with VideoFileClip(video_path) as video:
                            duration = video.duration
                            total_duration += duration
                            self.update_log(f"视频时长: {duration:.1f} 秒")

                        self.extract_frames(video_path, frame_dir)
                        subtitles = self.extract_subtitles_from_frames(frame_dir, ocr)
                        
                        if self.filter_words.get().strip():
                            self.update_log("\n开始过滤字幕...")
                            filtered_subtitles = self.filter_subtitles(subtitles)
                            self.update_log(f"过滤前: {len(subtitles)} 条，过滤后: {len(filtered_subtitles)} 条")
                        else:
                            filtered_subtitles = subtitles
                            
                        total_subtitles += len(filtered_subtitles)
                        
                        self.update_log("\n保存字幕结果...")
                        output_file.write(f"{video_file}\n")
                        for subtitle in filtered_subtitles:
                            output_file.write(f"{subtitle}\n")
                        output_file.write("\n")
                        
                        self.update_log("清理临时文件...")
                        for frame_file in os.listdir(frame_dir):
                            os.remove(os.path.join(frame_dir, frame_file))
                        os.rmdir(frame_dir)
                        
                    except Exception as e:
                        self.update_log(f"处理视频时出错: {str(e)}")
                        continue
                        
                    self.update_progress(i + 1)

            total_time = time.time() - start_time
            self.display_summary_statistics(len(video_files), total_duration, total_subtitles, total_time)
            
        except Exception as e:
            self.update_log(f"处理过程中出现错误: {str(e)}")        
    def display_summary_statistics(self, video_count, total_duration, total_subtitles, total_time):
        """在界面显示总体统计信息"""
        summary = f"""
{'='*50}
                统计信息总结
{'='*50}
作者: 忧郁男孩的救赎
网站: blueboySalvat.top
{'-'*50}
处理视频总数: {video_count} 个
视频总时长: {total_duration:.2f} 秒 ({total_duration/60:.2f} 分钟)
识别字幕总数: {total_subtitles} 条
处理总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)
平均处理速度: {total_duration/total_time:.2f}x
{'='*50}
"""
        self.log(summary)

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleExtractorGUI(root)
    root.mainloop()
import json
import logging
import multiprocessing
import os
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from tkinter import ttk, filedialog, scrolledtext

import cv2
import torch  # 用于GPU检测
from moviepy.video.io.VideoFileClip import VideoFileClip
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor, as_completed

# 配置日志级别，减少不必要的输出
logging.basicConfig(level=logging.INFO)
for logger_name in ['ppocr', 'paddle', 'matplotlib']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频字幕提取器")
        self.root.geometry("800x600")

        # 获取CPU核心数
        self.cpu_count = multiprocessing.cpu_count()
        # 检测是否有可用的GPU
        self.has_gpu = torch.cuda.is_available()

        # 初始化处理线程
        self.processing_thread = None
        self.stop_processing = False

        # 配置根窗口的网格权重
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 初始化配置
        self.config = {
            'filter_words': [],
            'crop_top': 0.8,
            'crop_bottom': 0.95,
            'use_gpu': False,
            'thread_count': min(4, self.cpu_count)
        }
        self.load_config()

        self.create_gui()

    def create_gui(self):
        """创建图形界面"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(1, weight=1)

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

        # 性能设置区域
        settings_frame = ttk.LabelFrame(main_frame, text="性能设置", padding="10")
        settings_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        settings_frame.grid_columnconfigure(1, weight=1)

        # GPU选项
        self.use_gpu = tk.BooleanVar(value=self.config['use_gpu'])
        gpu_check = ttk.Checkbutton(settings_frame, text="使用GPU加速", variable=self.use_gpu)
        gpu_check.grid(row=0, column=0, padx=(0, 20))
        if not self.has_gpu:
            gpu_check.configure(state='disabled')
            ttk.Label(settings_frame, text="(未检测到可用GPU)").grid(row=0, column=1, sticky=tk.W)

        # 线程数设置
        ttk.Label(settings_frame, text="CPU线程数:").grid(row=0, column=2, padx=(20, 5))
        self.thread_count = tk.IntVar(value=self.config['thread_count'])
        thread_spinbox = ttk.Spinbox(settings_frame, from_=1, to=self.cpu_count,
                                     textvariable=self.thread_count, width=5)
        thread_spinbox.grid(row=0, column=3)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))

        # 日志显示区域
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=(10, 0))

        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing, width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="停止处理", command=self.stop_processing_handler,
                                      width=20, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 配置主框架的行权重
        main_frame.grid_rowconfigure(5, weight=1)

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

    def save_frame(self, frame, frame_path):
        """
        保存帧。

        Args:
            frame: 已经裁剪好的字幕区域帧数据
            frame_path: 保存路径
        """
        cv2.imwrite(frame_path, frame)

    def is_similar_frame(self, frame1, frame2, threshold=0.95):
        """判断两帧是否相似（基于 SSIM）"""
        # 转为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # 计算 SSIM
        score, _ = ssim(gray1, gray2, full=True)
        return score > threshold

    def extract_frames(self, video_path, output_dir):
        """提取视频帧"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 性能优化：使用opencv的硬件加速
        if self.use_gpu.get():
            cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_DSHOW)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // 2)  # 确保最小间隔为1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        saved_count = 0
        crop_height = (self.crop_top.get(), self.crop_bottom.get())

        # 使用动态线程池大小
        thread_count = self.thread_count.get() if not self.use_gpu.get() else 2

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            prev_subtitle_frame = None

            while cap.isOpened():
                if self.stop_processing:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # 裁剪字幕区域用于相似性判断
                    h = frame.shape[0]
                    subtitle_frame = frame[int(h * crop_height[0]):int(h * crop_height[1]), :]

                    # 判断是否相似
                    if prev_subtitle_frame is None or not self.is_similar_frame(prev_subtitle_frame, subtitle_frame):
                        frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                        # 提交保存任务（使用裁剪后的字幕帧）
                        futures.append(executor.submit(self.save_frame, subtitle_frame.copy(), frame_path))
                        saved_count += 1
                        prev_subtitle_frame = subtitle_frame.copy()

                frame_count += 1

            # 等待所有任务完成
            for future in futures:
                future.result()

        cap.release()
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
                if i % 120 == 0:  # 每120帧更新一次进度，避免刷新太频繁
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

        # 重置停止标志
        self.stop_processing = False

        # 更新按钮状态
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')

        # 创建并启动处理线程
        self.processing_thread = threading.Thread(
            target=self.process_videos,
            args=(folder_path,),
            daemon=True
        )
        self.processing_thread.start()

        # 启动周期性检查线程状态
        self.check_thread_status()

    def stop_processing_handler(self):
        """处理停止按钮点击事件"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_processing = True
            self.update_log("\n正在停止处理...")
            self.stop_button.configure(state='disabled')

    def check_thread_status(self):
        """检查处理线程的状态并更新GUI"""
        if self.processing_thread and self.processing_thread.is_alive():
            self.root.after(100, self.check_thread_status)
        else:
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            if self.stop_processing:
                self.log("\n处理已停止！")
            else:
                self.log("\n处理完成！")

    def update_progress(self, value):
        """从线程安全地更新进度条"""
        self.root.after(0, lambda: self.progress.configure(value=value))

    def update_log(self, message):
        """从线程安全地更新日志"""
        self.root.after(0, lambda: self.log(message))

    def process_videos(self, input_folder):
        """并行处理指定文件夹中的所有视频"""
        try:
            start_time = time.time()
            total_duration = 0
            total_subtitles = 0

            self.update_log("初始化 OCR 引擎...")
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=self.use_gpu.get(),
                show_log=False,
                rec_thresh=0.6,
            )

            # 获取所有视频文件
            video_files = [f for f in sorted(os.listdir(input_folder))
                           if f.endswith(('.mp4', '.avi', '.mkv', '.mov', '.flv'))]

            self.root.after(0, lambda: self.progress.configure(maximum=len(video_files)))

            # 使用线程池并行处理视频
            with ThreadPoolExecutor(max_workers=self.thread_count.get()) as executor:
                futures = []
                for video_file in video_files:
                    if self.stop_processing:
                        break
                    future = executor.submit(
                        self.process_single_video,
                        input_folder, video_file, ocr
                    )
                    futures.append(future)

                # 等待所有任务完成并收集结果
                for future in as_completed(futures):
                    try:
                        duration, subtitles_count = future.result()
                        total_duration += duration
                        total_subtitles += subtitles_count
                    except Exception as e:
                        self.update_log(f"处理视频时出错: {str(e)}")

            total_time = time.time() - start_time
            self.display_summary_statistics(len(video_files), total_duration, total_subtitles, total_time)

        except Exception as e:
            self.update_log(f"处理过程中出现错误: {str(e)}")

    def process_single_video(self, input_folder, video_file, ocr):
        """处理单个视频"""
        video_path = os.path.join(input_folder, video_file)
        frame_dir = os.path.join(input_folder, f"frames_{os.path.splitext(video_file)[0]}")
        output_txt = os.path.join(input_folder, f"{os.path.splitext(video_file)[0]}.txt")

        self.update_log(f"\n处理视频: {video_file}")

        try:
            # 获取视频信息
            with VideoFileClip(video_path) as video:
                duration = video.duration
                self.update_log(f"视频时长: {duration:.1f} 秒")

            # 提取帧
            self.extract_frames(video_path, frame_dir)

            # 识别字幕
            subtitles = self.extract_subtitles_from_frames(frame_dir, ocr)

            # 过滤字幕
            if self.filter_words.get().strip():
                self.update_log("\n开始过滤字幕...")
                filtered_subtitles = self.filter_subtitles(subtitles)
                self.update_log(f"过滤前: {len(subtitles)} 条，过滤后: {len(filtered_subtitles)} 条")
            else:
                filtered_subtitles = subtitles

            # 保存字幕结果
            self.update_log("\n保存字幕结果...")
            with open(output_txt, 'w', encoding='utf-8') as output_file:
                output_file.write(f"{video_file}\n")
                for subtitle in filtered_subtitles:
                    output_file.write(f"{subtitle}\n")

            # 清理临时文件
            self.update_log("清理临时文件...")
            for frame_file in os.listdir(frame_dir):
                os.remove(os.path.join(frame_dir, frame_file))
            os.rmdir(frame_dir)

            return duration, len(filtered_subtitles)

        except Exception as e:
            self.update_log(f"处理视频 {video_file} 时出错: {str(e)}")
            raise
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
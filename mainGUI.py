import json
import logging
import multiprocessing
import os
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import ttk, filedialog, scrolledtext

import cv2
import torch  # 用于GPU检测
from PIL import Image, ImageTk  # 新增导入
from moviepy.video.io.VideoFileClip import VideoFileClip
from paddleocr import PaddleOCR
from skimage.metrics import structural_similarity as ssim

# 配置日志级别，减少不必要的输出
logging.basicConfig(level=logging.INFO)
for logger_name in ['ppocr', 'paddle', 'matplotlib']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)


class SubtitleExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("视频字幕提取器")
        self.root.geometry("1000x800")

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

        # 初始化当前帧
        self.frames = []  # 存储从视频中抽取的三帧
        self.frame_labels = []  # 存储用于展示帧的 Label 控件

        # 添加视频相关的成员变量
        self.video_path = tk.StringVar()
        self.video_capture = None
        
        # 添加预览控制相关变量
        self.preview_timer = None
        self.preview_label = None  # 将在create_gui中初始化
        
        self.create_gui()

    def create_gui(self):
        """创建图形界面"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(1, weight=1)

        # 修改为视频文件选择区域
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        file_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(file_frame, text="视频文件:").grid(row=0, column=0, padx=(0, 5))
        ttk.Entry(file_frame, textvariable=self.video_path).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(file_frame, text="浏览", command=self.browse_video).grid(row=0, column=2, padx=(5, 0))
        
        # 修改预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="视频预览", padding="10")
        preview_frame.grid(row=1, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # 预览显示区域 - 调整大小为500像素宽
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.grid(row=0, column=0, pady=(0, 10))
        
        # 添加进度滑块
        self.video_progress = ttk.Scale(
            preview_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            command=self.on_progress_change
        )
        self.video_progress.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 过滤词设置区域
        filter_frame = ttk.Frame(main_frame)
        filter_frame.grid(row=2, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        filter_frame.grid_columnconfigure(1, weight=1)

        ttk.Label(filter_frame, text="过滤词(用逗号分隔):").grid(row=0, column=0, padx=(0, 5))
        self.filter_words = tk.StringVar(value=','.join(self.config['filter_words']))
        ttk.Entry(filter_frame, textvariable=self.filter_words).grid(row=0, column=1, sticky=(tk.W, tk.E))

        # 裁剪区域设置框
        crop_frame = ttk.LabelFrame(main_frame, text="裁剪区域设置", padding="10")
        crop_frame.grid(row=3, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        crop_frame.grid_columnconfigure(1, weight=1)
        crop_frame.grid_columnconfigure(3, weight=1)

        # 上下边界设置
        ttk.Label(crop_frame, text="上边界 (0-1):").grid(row=0, column=0, padx=(0, 5))
        self.crop_top = tk.DoubleVar(value=self.config['crop_top'])
        self.crop_top_scale = ttk.Scale(crop_frame, from_=0, to=1, variable=self.crop_top, orient=tk.HORIZONTAL, length=200)
        self.crop_top_scale.grid(row=0, column=1, sticky=tk.W)
        self.crop_top_label = ttk.Label(crop_frame, text=f"{self.crop_top.get():.2f}")
        self.crop_top_label.grid(row=0, column=2, padx=(5, 0))

        ttk.Label(crop_frame, text="下边界 (0-1):").grid(row=0, column=3, padx=(20, 5))
        self.crop_bottom = tk.DoubleVar(value=self.config['crop_bottom'])
        self.crop_bottom_scale = ttk.Scale(crop_frame, from_=0, to=1, variable=self.crop_bottom, orient=tk.HORIZONTAL, length=200)
        self.crop_bottom_scale.grid(row=0, column=4, sticky=tk.W)
        self.crop_bottom_label = ttk.Label(crop_frame, text=f"{self.crop_bottom.get():.2f}")
        self.crop_bottom_label.grid(row=0, column=5, padx=(5, 0))

        # 左右边界设置
        ttk.Label(crop_frame, text="左边界 (0-1):").grid(row=1, column=0, padx=(0, 5), pady=(10, 0))
        self.crop_left = tk.DoubleVar(value=self.config.get('crop_left', 0))
        self.crop_left_scale = ttk.Scale(crop_frame, from_=0, to=1, variable=self.crop_left, orient=tk.HORIZONTAL, length=200)
        self.crop_left_scale.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        self.crop_left_label = ttk.Label(crop_frame, text=f"{self.crop_left.get():.2f}")
        self.crop_left_label.grid(row=1, column=2, padx=(5, 0), pady=(10, 0))

        ttk.Label(crop_frame, text="右边界 (0-1):").grid(row=1, column=3, padx=(20, 5), pady=(10, 0))
        self.crop_right = tk.DoubleVar(value=self.config.get('crop_right', 1))
        self.crop_right_scale = ttk.Scale(crop_frame, from_=0, to=1, variable=self.crop_right, orient=tk.HORIZONTAL, length=200)
        self.crop_right_scale.grid(row=1, column=4, sticky=tk.W, pady=(10, 0))
        self.crop_right_label = ttk.Label(crop_frame, text=f"{self.crop_right.get():.2f}")
        self.crop_right_label.grid(row=1, column=5, padx=(5, 0), pady=(10, 0))

        # 绑定滑块值变化事件
        self.crop_top_scale.bind("<Motion>", lambda e: self.update_crop_labels_and_preview())
        self.crop_bottom_scale.bind("<Motion>", lambda e: self.update_crop_labels_and_preview())
        self.crop_left_scale.bind("<Motion>", lambda e: self.update_crop_labels_and_preview())
        self.crop_right_scale.bind("<Motion>", lambda e: self.update_crop_labels_and_preview())

        # 性能设置区域
        settings_frame = ttk.LabelFrame(main_frame, text="性能设置", padding="10")
        settings_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        settings_frame.grid_columnconfigure(1, weight=1)

        # GPU 选项
        self.use_gpu = tk.BooleanVar(value=self.config['use_gpu'])
        gpu_check = ttk.Checkbutton(settings_frame, text="使用GPU加速", variable=self.use_gpu)
        gpu_check.grid(row=0, column=0, padx=(0, 20))
        if not self.has_gpu:
            gpu_check.configure(state='disabled')
            ttk.Label(settings_frame, text="(未检测到可用GPU)").grid(row=0, column=1, sticky=tk.W)

        # 内存加速选项
        self.use_memory = tk.BooleanVar(value=self.config.get('use_memory', False))
        memory_check = ttk.Checkbutton(settings_frame, text="使用内存加速", variable=self.use_memory)
        memory_check.grid(row=1, column=0, padx=(0, 20))

        # 线程数设置
        ttk.Label(settings_frame, text="CPU线程数:").grid(row=1, column=2, padx=(20, 5))
        self.thread_count = tk.IntVar(value=self.config['thread_count'])
        thread_spinbox = ttk.Spinbox(settings_frame, from_=1, to=self.cpu_count,
                                     textvariable=self.thread_count, width=5)
        thread_spinbox.grid(row=1, column=3)

        # 进度条
        self.progress = ttk.Progressbar(main_frame, mode='determinate')
        self.progress.grid(row=5, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))

        # 日志显示区域
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=15)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=(10, 0))

        self.start_button = ttk.Button(button_frame, text="开始处理", command=self.start_processing, width=20)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(button_frame, text="停止处理", command=self.stop_processing_handler,
                                      width=20, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # 配置主框架的行权重
        main_frame.grid_rowconfigure(6, weight=1)

    def browse_video(self):
        """选择视频文件"""
        video_file = filedialog.askopenfilename(
            filetypes=[
                ("视频文件", "*.mp4 *.avi *.mkv *.mov *.flv"),
                ("所有文件", "*.*")
            ]
        )
        if video_file:
            self.video_path.set(video_file)
            # 初始化视频捕获
            self.init_video_capture()

    def init_video_capture(self):
        """初始化视频捕获"""
        try:
            if self.video_capture is not None:
                self.video_capture.release()
            
            self.video_capture = cv2.VideoCapture(self.video_path.get())
            if not self.video_capture.isOpened():
                self.log("无法打开视频文件")
                return False
            
            # 获取视频总帧数
            total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                self.log("无法获取视频帧数")
                return False
            
            # 读取第一帧来初始化预览
            ret, frame = self.video_capture.read()
            if ret:
                frame_with_box = self.draw_crop_box(frame)
                self.show_preview_frame(frame_with_box)
                # 重置到开始位置
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return True
            return False
        except Exception as e:
            self.log(f"初始化视频失败: {str(e)}")
            return False

    # def toggle_preview(self):
    #     """切换预览状态"""
    #     if not self.preview_playing:
    #         if self.video_capture is None:
    #             if not self.init_video_capture():
    #                 return
    #         self.preview_playing = True
    #         self.preview_button.configure(text="停止预览")
    #         self.update_preview()
    #     else:
    #         self.stop_preview()

    def update_preview(self):
        """更新预览画面"""
        try:
            if self.preview_playing and self.video_capture is not None and self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    # 更新时间显示
                    current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
                    if total_frames > 0 and fps > 0:
                        current_time = current_frame / fps
                        total_time = total_frames / fps
                        self.time_label.config(
                            text=f"{int(current_time//60):02d}:{int(current_time%60):02d} / "
                                 f"{int(total_time//60):02d}:{int(total_time%60):02d}"
                        )
                    # 更新进度滑块
                    current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                    total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total_frames > 0:
                        progress = (current_frame / total_frames) * 100
                        self.video_progress.set(progress)
                    
                    # 显示带裁剪框的帧
                    frame_with_box = self.draw_crop_box(frame)
                    self.show_preview_frame(frame_with_box)
                    
                    # 继续更新
                    self.preview_timer = self.root.after(30, self.update_preview)
                else:
                    # 视频结束，重新开始
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.preview_timer = self.root.after(30, self.update_preview)
            else:
                self.stop_preview()
        except Exception as e:
            self.log(f"更新预览时出错: {str(e)}")
            self.stop_preview()

    # def stop_preview(self):
    #     """停止预览"""
    #     self.preview_playing = False
    #     self.preview_button.configure(text="播放预览")
    #     if self.preview_timer:
    #         self.root.after_cancel(self.preview_timer)
    #         self.preview_timer = None

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
        self.config['crop_left'] = self.crop_left.get()
        self.config['crop_right'] = self.crop_right.get()
        self.config['use_gpu'] = self.use_gpu.get()
        self.config['use_memory'] = self.use_memory.get()

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
        """提取视频帧，根据用户选择使用内存加速"""
        use_memory = self.use_memory.get()
        if not use_memory:  # 仅在磁盘存储模式下创建文件夹
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")

        # 性能优化：使用 OpenCV 的硬件加速
        if self.use_gpu.get():
            cap.set(cv2.CAP_PROP_BACKEND, cv2.CAP_DSHOW)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(1, fps // 2)  # 确保最小间隔为 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 根据用户选择使用内存加速
        if use_memory:
            self.update_log("使用内存缓存帧数据")
            frame_cache = []
        else:
            self.update_log("使用磁盘存储帧数据")

        frame_count = 0
        saved_count = 0
        crop_height = (self.crop_top.get(), self.crop_bottom.get())
        crop_width = (self.crop_left.get(), self.crop_right.get())

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
                    h, w = frame.shape[:2]
                    subtitle_frame = frame[int(h * crop_height[0]):int(h * crop_height[1]), 
                                         int(w * crop_width[0]):int(w * crop_width[1])]

                    # 判断是否相似
                    if prev_subtitle_frame is None or not self.is_similar_frame(prev_subtitle_frame, subtitle_frame):
                        if use_memory:
                            # 使用内存缓存帧
                            frame_cache.append(subtitle_frame.copy())
                        else:
                            # 使用磁盘存储帧
                            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                            futures.append(executor.submit(self.save_frame, subtitle_frame.copy(), frame_path))
                        saved_count += 1
                        prev_subtitle_frame = subtitle_frame.copy()

                frame_count += 1

            # 等待所有任务完成
            for future in futures:
                future.result()

        cap.release()

        # 如果使用内存加速，返回帧数据；否则返回 None
        return frame_cache if use_memory else None

    def extract_subtitles_from_frames(self, frame_dir, ocr, frame_cache=None):
        """从帧中提取字幕文本"""
        subtitles = []
        prev_text = None

        if frame_cache is not None:
            # 从内存中读取帧数据
            total_frames = len(frame_cache)
            self.update_log(f"\n开始识别字幕，共 {total_frames} 帧...")

            for i, frame in enumerate(frame_cache):
                try:
                    # 更新识别进度
                    if i % 60 == 0:  # 每120帧更新一次进度，避免刷新太频繁
                        progress = (i + 1) / total_frames * 100
                        self.update_log(f"\r字幕识别进度: {progress:.1f}%")

                    # 直接使用内存中的帧数据进行 OCR
                    results = ocr.ocr(frame)
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
                        similarity = len(set(current_text) & set(prev_text)) / max(len(set(current_text)),
                                                                                   len(set(prev_text)))
                        if similarity > 0.8:
                            continue

                    if current_text:
                        subtitles.append(current_text)
                        prev_text = current_text

                except Exception as e:
                    self.update_log(f"\n处理帧 {i} 时出错: {str(e)}")
                    continue
        else:
            # 从磁盘中读取帧数据
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

                    # 从磁盘读取帧数据进行 OCR
                    frame = cv2.imread(frame_path)
                    results = ocr.ocr(frame)
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
                        similarity = len(set(current_text) & set(prev_text)) / max(len(set(current_text)),
                                                                                   len(set(prev_text)))
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
        video_path = self.video_path.get()
        if not video_path:
            self.log("请选择视频文件")
            return

        # 重置停止标志
        self.stop_processing = False

        # 更新按钮状态
        self.start_button.configure(state='disabled')
        self.stop_button.configure(state='normal')

        # 创建并启动处理线程
        self.processing_thread = threading.Thread(
            target=self.process_videos,
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

    def process_videos(self):
        """处理单个视频"""
        try:
            video_path = self.video_path.get()
            if not video_path:
                self.update_log("请选择视频文件")
                return

            start_time = time.time()
            
            self.update_log("初始化 OCR 引擎...")
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='ch',
                use_gpu=self.use_gpu.get(),
                show_log=False,
                ocr_version='PP-OCRv4',
                det_limit_side_len=2880,
                det_db_thresh=0.1,
                det_db_box_thresh=0.3,
                det_db_unclip_ratio=2.0,
                rec_thresh=0.6,
                cls_thresh=0.9,
                use_space_char=True,
                use_mp=True
            )

            # 处理视频
            output_txt = os.path.splitext(video_path)[0] + ".txt"
            frame_dir = os.path.splitext(video_path)[0] + "_frames"

            if not self.use_memory.get():
                os.makedirs(frame_dir, exist_ok=True)

            # 提取帧
            frame_cache = self.extract_frames(video_path, frame_dir)

            # 识别字幕
            subtitles = self.extract_subtitles_from_frames(frame_dir, ocr, frame_cache)

            # 过滤字幕
            if self.filter_words.get().strip():
                self.update_log("\n开始过滤字幕...")
                filtered_subtitles = self.filter_subtitles(subtitles)
                self.update_log(f"过滤前: {len(subtitles)} 条，过滤后: {len(filtered_subtitles)} 条")
            else:
                filtered_subtitles = subtitles

            # 保存字幕结果
            self.update_log("\n保存字幕结果...")
            with open(output_txt, 'w', encoding='utf-8') as f:
                f.write(f"{os.path.basename(video_path)}\n")
                for subtitle in filtered_subtitles:
                    f.write(f"{subtitle}\n")

            # 清理临时文件
            if not self.use_memory.get() and os.path.exists(frame_dir):
                self.update_log("清理临时文件...")
                for frame_file in os.listdir(frame_dir):
                    os.remove(os.path.join(frame_dir, frame_file))
                os.rmdir(frame_dir)

            total_time = time.time() - start_time
            self.update_log(f"\n处理完成！总用时: {total_time:.2f} 秒")

        except Exception as e:
            self.update_log(f"处理过程中出现错误: {str(e)}")

    def update_crop_labels_and_preview(self):
        """更新裁剪区域标签和预览帧"""
        # 更新标签
        self.crop_top_label.config(text=f"{self.crop_top.get():.2f}")
        self.crop_bottom_label.config(text=f"{self.crop_bottom.get():.2f}")
        self.crop_left_label.config(text=f"{self.crop_left.get():.2f}")
        self.crop_right_label.config(text=f"{self.crop_right.get():.2f}")

        # 如果视频已加载，更新预览帧
        if self.video_capture is not None:
            # 保存当前帧位置
            current_pos = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.video_capture.read()
            if ret:
                # 绘制裁剪框并显示
                frame_with_box = self.draw_crop_box(frame)
                self.show_preview_frame(frame_with_box)
                # 恢复帧位置
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

    def on_progress_change(self, value):
        """处理进度滑块变化"""
        try:
            if self.video_capture is not None and self.video_capture.isOpened():
                # 设置视频位置
                total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    target_frame = int((float(value) / 100) * total_frames)
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    
                    # 显示当前帧
                    ret, frame = self.video_capture.read()
                    if ret:
                        frame_with_box = self.draw_crop_box(frame)
                        self.show_preview_frame(frame_with_box)
        except Exception as e:
            self.log(f"更新进度时出错: {str(e)}")

    def show_preview_frame(self, frame):
        """显示预览帧"""
        # 转换帧为RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # 计算调整后的尺寸（固定宽度为500像素，最大高度为300像素）
        new_width = 500
        new_height = int(h * (new_width / w))
        
        # 如果高度超过300像素，则按最大高度等比例缩放
        max_height = 500
        if new_height > max_height:
            new_width = int(w * (max_height / h))
            new_height = max_height
        
        frame = cv2.resize(frame, (new_width, new_height))
        
        # 转换为PhotoImage
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # 保持引用

    def draw_crop_box(self, frame):
        """在帧上绘制裁剪区域的预选框"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        frame_with_box = frame.copy()
        
        # 获取裁剪区域坐标
        top = int(h * self.crop_top.get())
        bottom = int(h * self.crop_bottom.get())
        left = int(w * self.crop_left.get())
        right = int(w * self.crop_right.get())
        
        # 绘制矩形框
        cv2.rectangle(frame_with_box, (left, top), (right, bottom), (0, 255, 0), 2)
        
        return frame_with_box

if __name__ == "__main__":
    root = tk.Tk()
    app = SubtitleExtractorGUI(root)
    root.mainloop()
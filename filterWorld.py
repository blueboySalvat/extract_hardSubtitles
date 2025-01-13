import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import re
import threading
from queue import Queue
from typing import Optional

class TextFilterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("文本过滤器")
        self.root.geometry("800x600")
        
        # 配置文件路径和数据
        self.config_file = "filter_config.json"
        self.filter_words = self.load_config()
        self.current_file: Optional[str] = None
        self.file_content: Optional[str] = None
        
        # 线程安全的队列，用于处理过滤结果
        self.result_queue = Queue()
        
        # 语言过滤设置
        self.filter_chinese = tk.BooleanVar(value=False)
        self.filter_english = tk.BooleanVar(value=False)
        
        # 创建界面
        self.create_widgets()
        
        # 开始检查过滤结果队列
        self.check_filter_queue()
        
    def create_widgets(self):
        # 创建主分割容器
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # 左侧面板
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # 过滤词列表
        ttk.Label(left_frame, text="过滤词列表").pack(pady=5)
        
        list_frame = ttk.Frame(left_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5)
        
        self.filter_listbox = tk.Listbox(list_frame)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.filter_listbox.yview)
        self.filter_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.filter_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 添加现有过滤词
        for word in self.filter_words:
            self.filter_listbox.insert(tk.END, word)
        
        # 按钮区域
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(btn_frame, text="添加", command=self.add_filter_word).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除", command=self.delete_filter_word).pack(side=tk.LEFT, padx=2)
        
        # 语言过滤选项
        lang_frame = ttk.LabelFrame(left_frame, text="语言过滤")
        lang_frame.pack(fill=tk.X, pady=5, padx=5)
        
        ttk.Checkbutton(lang_frame, text="过滤中文", variable=self.filter_chinese,
                       command=self.trigger_filter).pack(anchor=tk.W)
        ttk.Checkbutton(lang_frame, text="过滤英文", variable=self.filter_english,
                       command=self.trigger_filter).pack(anchor=tk.W)
        
        # 右侧面板
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)
        
        # 文件操作按钮
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5, padx=5)
        ttk.Button(btn_frame, text="选择文件", command=self.choose_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="复制结果", command=self.copy_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="导出结果", command=self.export_result).pack(side=tk.LEFT, padx=2)
        
        # 进度条和状态显示
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(right_frame, textvariable=self.status_var).pack(fill=tk.X, padx=5)
        
        # 文本显示区域
        self.create_text_area(right_frame, "原文预览", "original_text")
        self.create_text_area(right_frame, "过滤结果", "filtered_text")

    def create_text_area(self, parent, label, name):
        """创建文本显示区域"""
        ttk.Label(parent, text=label).pack(padx=5)
        
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
        text_widget = tk.Text(frame, height=10, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        setattr(self, name, text_widget)

    def choose_file(self):
        """选择文件并加载内容"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not file_path:
            return
            
        self.current_file = file_path
        self.status_var.set(f"正在加载文件: {os.path.basename(file_path)}")
        self.root.update()
        
        try:
            # 在单独的线程中加载文件
            threading.Thread(target=self.load_file_content, args=(file_path,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("错误", f"无法加载文件: {str(e)}")
            self.status_var.set("加载失败")

    def load_file_content(self, file_path):
        """在单独的线程中加载文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.file_content = content
            
            # 在主线程中更新UI
            self.root.after(0, self.update_original_text, content)
            self.root.after(0, self.trigger_filter)
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "错误", f"读取文件失败: {str(e)}")
            self.root.after(0, lambda: self.status_var.set("读取失败"))

    def update_original_text(self, content):
        """更新原文显示"""
        self.original_text.delete('1.0', tk.END)
        self.original_text.insert('1.0', content)
        self.status_var.set("文件已加载")

    def filter_text(self):
        """在单独的线程中执行文本过滤"""
        if not self.file_content:
            return
            
        content = self.file_content
        filtered = content
        
        # 关键词过滤
        for word in self.filter_words:
            filtered = filtered.replace(word, '')
        
        # 语言过滤
        if self.filter_chinese.get():
            filtered = re.sub(r'[\u4e00-\u9fff]+', '', filtered)
        
        if self.filter_english.get():
            filtered = re.sub(r'[a-zA-Z]+', '', filtered)
        
        # 清理多余空格
        filtered = re.sub(r'\s+', ' ', filtered).strip()
        
        # 将结果放入队列
        self.result_queue.put(filtered)

    def trigger_filter(self):
        """触发过滤操作"""
        if not self.file_content:
            return
            
        self.status_var.set("正在过滤...")
        threading.Thread(target=self.filter_text, daemon=True).start()

    def check_filter_queue(self):
        """定期检查过滤结果队列"""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.filtered_text.delete('1.0', tk.END)
                self.filtered_text.insert('1.0', result)
                self.status_var.set("过滤完成")
        finally:
            # 每100ms检查一次队列
            self.root.after(100, self.check_filter_queue)

    def copy_result(self):
        """复制过滤结果到剪贴板"""
        result = self.filtered_text.get('1.0', tk.END).strip()
        if result:
            self.root.clipboard_clear()
            self.root.clipboard_append(result)
            messagebox.showinfo("成功", "已复制到剪贴板")
        else:
            messagebox.showwarning("警告", "没有可复制的内容")

    def export_result(self):
        """导出过滤结果"""
        if not self.current_file or not self.filtered_text.get('1.0', tk.END).strip():
            messagebox.showwarning("警告", "没有可导出的内容")
            return
            
        file_name, ext = os.path.splitext(self.current_file)
        export_path = f"{file_name}_filtered{ext}"
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(self.filtered_text.get('1.0', tk.END))
            messagebox.showinfo("成功", f"已导出到：\n{export_path}")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

    def add_filter_word(self):
        """添加过滤词"""
        dialog = tk.Toplevel(self.root)
        dialog.title("添加过滤词")
        dialog.geometry("300x100")
        
        ttk.Label(dialog, text="请输入过滤词:").pack(pady=5)
        entry = ttk.Entry(dialog)
        entry.pack(pady=5)
        
        def confirm():
            word = entry.get().strip()
            if word:
                self.filter_words.append(word)
                self.filter_listbox.insert(tk.END, word)
                self.save_config()
                dialog.destroy()
                self.trigger_filter()
            else:
                messagebox.showwarning("警告", "请输入有效的过滤词")
                
        ttk.Button(dialog, text="确定", command=confirm).pack(pady=5)

    def delete_filter_word(self):
        """删除过滤词"""
        selection = self.filter_listbox.curselection()
        if selection:
            index = selection[0]
            word = self.filter_listbox.get(index)
            self.filter_words.remove(word)
            self.filter_listbox.delete(index)
            self.save_config()
            self.trigger_filter()
        else:
            messagebox.showwarning("警告", "请先选择要删除的过滤词")

    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_config(self):
        """保存配置文件"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.filter_words, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = TextFilterApp(root)
    root.mainloop()
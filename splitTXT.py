import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import os


class FileSplitterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("文件分割工具")
        self.root.geometry("400x200")

        # 创建主框架
        self.main_frame = tk.Frame(root, padx=20, pady=20)
        self.main_frame.pack(expand=True, fill='both')

        # 选择文件按钮
        self.select_button = tk.Button(
            self.main_frame,
            text="选择文件",
            command=self.select_file,
            height=2
        )
        self.select_button.pack(fill='x', pady=(0, 10))

        # 显示选中的文件路径
        self.file_label = tk.Label(
            self.main_frame,
            text="未选择文件",
            wraplength=350
        )
        self.file_label.pack(fill='x', pady=(0, 10))

        # 开始分割按钮
        self.split_button = tk.Button(
            self.main_frame,
            text="开始分割",
            command=self.split_file,
            height=2,
            state='disabled'
        )
        self.split_button.pack(fill='x')

        # 状态标签
        self.status_label = tk.Label(
            self.main_frame,
            text="",
            wraplength=350
        )
        self.status_label.pack(fill='x', pady=(10, 0))

        self.selected_file = None

    def select_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            self.selected_file = filename
            self.file_label.config(text=f"已选择: {filename}")
            self.split_button.config(state='normal')

    def split_file(self):
        if not self.selected_file:
            return

        try:
            # 获取文件名和路径
            base_path = Path(self.selected_file)
            base_name = base_path.stem
            output_dir = base_path.parent

            # 读取文件并分割
            with open(self.selected_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            total_parts = (len(lines) + 499) // 500  # 向上取整

            for i in range(total_parts):
                start_idx = i * 500
                end_idx = min((i + 1) * 500, len(lines))

                output_file = output_dir / f"{base_name}_P{i + 1}.txt"

                with open(output_file, 'w', encoding='utf-8') as f:
                    f.writelines(lines[start_idx:end_idx])

            messagebox.showinfo(
                "完成",
                f"文件分割完成!\n共生成 {total_parts} 个文件\n保存在: {output_dir}"
            )
            self.status_label.config(text=f"处理完成，共生成 {total_parts} 个文件")

        except Exception as e:
            messagebox.showerror("错误", f"处理过程中出现错误:\n{str(e)}")
            self.status_label.config(text="处理失败")


if __name__ == "__main__":
    root = tk.Tk()
    app = FileSplitterGUI(root)
    root.mainloop()
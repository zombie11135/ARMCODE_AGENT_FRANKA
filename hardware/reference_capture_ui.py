"""Small Tk desktop form for deliberate, cropped reference-photo registration."""
from collections import Counter
from pathlib import Path
import re
import time
from hardware.reference_ui_style import configure_fonts, prepare_tk_runtime

prepare_tk_runtime()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import uuid

from PIL import Image, ImageTk

from hardware.reference_capture import ImageViewport, PreviewFrame, VIEW_LABELS, save_snapshot


class ReferenceCaptureApp:
    def __init__(self, root, gallery, cameras, *, worker=None, offline_frame=None):
        self.root, self.gallery, self.cameras = root, gallery, cameras
        self.worker, self.offline_frame = worker, offline_frame
        self.live_frame = offline_frame
        self.display_frame = None
        self.frozen = None
        self.bbox = None
        self.drag_start = None
        self.viewport = None
        self._loaded_id = None
        self._seen_error = None
        self._closing = False
        self._session = "registration_" + uuid.uuid4().hex
        self._photo = self._crop_photo = None
        self.fonts = configure_fonts(root)
        self.root.title("物品图片录入 · 腕部相机")
        width = min(1280, max(960, root.winfo_screenwidth() - 60))
        height = min(960, max(720, root.winfo_screenheight() - 80))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(min(1040, width), min(820, height))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.form = {key: tk.StringVar(root) for key in
                     ("name", "category", "prompt", "aliases", "brand", "specification", "view")}
        self.form["category"].set("medicine_box")
        self.form["prompt"].set("medicine package")
        self.form["view"].set("正面")
        self.status = tk.StringVar(root, "先选择或输入物品名称，然后连接腕部相机。")
        self.object_info = tk.StringVar(root, "新物品：第一次保存时创建，后续照片追加到同一身份。")
        self._build()
        self.refresh_names()
        if offline_frame:
            self.status.set("本地图片模式，不连接任何设备。点击“拍摄 / 冻结”，再框选目标。")
        root.bind("<F8>", lambda event: self.capture())
        root.bind("<Control-s>", lambda event: self.save())
        root.bind("<Escape>", lambda event: self.resume())
        root.after(60, self.poll)

    def _build(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="物品图片录入", font=self.fonts["title"]).pack(anchor="w")
        source = ("本地图片模式" if self.offline_frame else
                  f"腕部 D435i · {self.cameras.wrist_serial} · {self.cameras.width} × {self.cameras.height} · {self.cameras.fps} fps")
        ttk.Label(outer, text=source + "   |   预览 → 拍摄 → 框选 → 保存", font=self.fonts["hint"]).pack(anchor="w", pady=(4, 10))
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 10))
        self.buttons = {}
        for key, label, command in [
            ("connect", "连接相机", self.connect), ("disconnect", "断开相机", self.disconnect),
            ("capture", "拍摄 / 冻结  F8", self.capture), ("resume", "继续预览  Esc", self.resume),
            ("save", "保存并追加  Ctrl+S", self.save), ("import", "打开本地图片", self.import_image),
        ]:
            button = ttk.Button(toolbar, text=label, command=command)
            button.pack(side="left", padx=(0, 6))
            self.buttons[key] = button
        body = ttk.Frame(outer)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        self.canvas = tk.Canvas(left, width=700, height=480, bg="#17212b", highlightthickness=0)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<ButtonPress-1>", self.begin_box)
        self.canvas.bind("<B1-Motion>", self.drag_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_box)
        left_controls = ttk.Frame(left)
        left_controls.pack(side="bottom", fill="x")
        ttk.Label(left_controls, text="冻结后，拖动鼠标框住一个完整物品，尽量少带背景。", wraplength=620).pack(anchor="w", pady=6)
        row = ttk.Frame(left_controls)
        row.pack(fill="x")
        ttk.Button(row, text="重新框选", command=self.clear_box).pack(side="left")
        ttk.Button(row, text="整张图片作为物品", command=self.full_image).pack(side="left", padx=8)
        self.selection_text = tk.StringVar(self.root, "尚未框选")
        ttk.Label(row, textvariable=self.selection_text).pack(side="left")
        self.canvas.pack(fill="both", expand=True)
        panel_area = ttk.Frame(body)
        panel_area.grid(row=0, column=1, sticky="ns")
        # The form can scroll on smaller desktops; readable fonts must not hide the footer.
        panel_view = tk.Canvas(panel_area, width=360, highlightthickness=0,
                               bg=ttk.Style(self.root).lookup("TFrame", "background"))
        panel_scroll = ttk.Scrollbar(panel_area, orient="vertical", command=panel_view.yview)
        panel_view.configure(yscrollcommand=panel_scroll.set)
        panel_scroll.pack(side="right", fill="y")
        panel_view.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(panel_view)
        panel_window = panel_view.create_window(0, 0, window=panel, anchor="nw", width=360)
        panel.bind("<Configure>", lambda event: panel_view.configure(scrollregion=panel_view.bbox("all")))
        panel_view.bind("<Configure>", lambda event: panel_view.itemconfigure(panel_window, width=event.width))
        panel.columnconfigure(0, weight=1)
        ttk.Label(panel, text="物品名称（已有物品可直接选择）").pack(anchor="w")
        name_row = ttk.Frame(panel)
        name_row.pack(fill="x", pady=(3, 7))
        self.name_box = ttk.Combobox(name_row, textvariable=self.form["name"], width=24)
        self.name_box.pack(side="left", fill="x", expand=True)
        self.name_box.bind("<<ComboboxSelected>>", lambda event: self.load_object())
        self.name_box.bind("<FocusOut>", lambda event: self.load_object())
        ttk.Button(name_row, text="新物品", command=self.new_object, width=7).pack(side="left", padx=(5, 0))
        for key, label in [("aliases", "其他名称（可选，用分号分隔）"), ("brand", "品牌（可选，可由多种物品共享）"),
                           ("specification", "规格（可选）"), ("category", "类别（如 medicine_box / cube / bottle）"),
                           ("prompt", "检测提示词（英文，如 medicine package）")]:
            ttk.Label(panel, text=label).pack(anchor="w")
            ttk.Entry(panel, textvariable=self.form[key], width=34).pack(fill="x", pady=(2, 6))
        ttk.Label(panel, text="这张照片的视角").pack(anchor="w")
        ttk.Combobox(panel, textvariable=self.form["view"], values=list(VIEW_LABELS),
                     state="readonly", width=30).pack(fill="x", pady=(2, 6))
        ttk.Label(panel, textvariable=self.object_info, wraplength=350, justify="left").pack(anchor="w", pady=6)
        ttk.Label(panel, text="将录入图库的目标裁图").pack(anchor="w", pady=(5, 2))
        self.crop_canvas = tk.Canvas(panel, width=350, height=110, bg="#e8edf2", highlightthickness=0)
        self.crop_canvas.pack()
        ttk.Label(panel, text="视角标签用于整理照片；正面和侧面仍属于同一个物品。", wraplength=350, font=self.fonts["hint"]).pack(anchor="w", pady=5)
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x")
        ttk.Label(footer, textvariable=self.status, wraplength=1090, justify="left").pack(fill="x", pady=(10, 5))
        self.history = ttk.Treeview(footer, columns=("name", "view", "result"), show="headings", height=2)
        for key, title, width in [("name", "本次录入物品", 280), ("view", "照片视角", 100), ("result", "保存结果", 540)]:
            self.history.heading(key, text=title)
            self.history.column(key, width=width)
        self.history.pack(fill="x")
        ttk.Label(footer, text=f"图库：{self.gallery.directory}", wraplength=1200, font=self.fonts["hint"]).pack(anchor="w", pady=(5, 0))
        body.pack(fill="both", expand=True)

        def bind_scroll(widget):
            if widget.winfo_class() != "TCombobox":
                widget.bind("<Button-4>", lambda event: panel_view.yview_scroll(-2, "units"), add="+")
                widget.bind("<Button-5>", lambda event: panel_view.yview_scroll(2, "units"), add="+")
            for child in widget.winfo_children():
                bind_scroll(child)
        bind_scroll(panel)
        self.sync_buttons()

    def refresh_names(self):
        self.name_box["values"] = [obj["name"] for obj in self.gallery.catalog()]

    def load_object(self):
        obj = self.gallery.resolve(self.form["name"].get())
        if obj:
            if obj["object_id"] != self._loaded_id:
                self._loaded_id = obj["object_id"]
                self.form["name"].set(obj["name"])
                self.form["category"].set(obj["category"])
                self.form["prompt"].set(obj["detector_prompt"])
                self.form["aliases"].set("; ".join(obj["aliases"]))
                for key in ("brand", "specification"):
                    self.form[key].set(obj["attributes"].get(key, ""))
            counts = Counter(ref.get("viewpoint", "unspecified") for ref in obj["references"])
            detail = " / ".join(f"{label} {counts[key]}" for label, key in VIEW_LABELS.items() if counts[key])
            self.object_info.set(f"已有 {len(obj['references'])} 张 · 保存时追加\n{detail}")
        else:
            if self._loaded_id is not None:
                for key in ("aliases", "brand", "specification"):
                    self.form[key].set("")
            self._loaded_id = None
            self.object_info.set("新物品：第一次保存时创建。\n同一物品的不同视角使用相同名称。")

    def new_object(self):
        self._loaded_id = None
        for key in ("name", "aliases", "brand", "specification"):
            self.form[key].set("")
        self.form["category"].set("medicine_box")
        self.form["prompt"].set("medicine package")
        self.form["view"].set("正面")
        self.load_object()
        self.name_box.focus_set()

    def sync_buttons(self):
        running = bool(self.worker and self.worker.running)
        states = {"connect": bool(self.worker) and not running,
                  "disconnect": running, "capture": self.display_frame is not None and self.frozen is None,
                  "resume": self.frozen is not None, "save": self.frozen is not None and self.bbox is not None,
                  "import": True}
        for key, enabled in states.items():
            self.buttons[key].configure(state="normal" if enabled else "disabled")

    def connect(self):
        if not self.worker or self.worker.running:
            return
        self.offline_frame = None
        self.live_frame = self.display_frame = self.frozen = None
        self.bbox = None
        self._seen_error = None
        self.worker.start()
        self.status.set("正在连接腕部相机，请稍候……")
        self.draw()
        self.sync_buttons()

    def disconnect(self):
        if self.worker:
            self.worker.stop()
        if self.frozen is None:
            self.live_frame = self.display_frame = None
        self.status.set("正在释放相机；已冻结的照片仍可保存。")
        self.draw()
        self.sync_buttons()

    def poll(self):
        if self._closing:
            return
        if self.worker and not self.offline_frame:
            frame, error = self.worker.latest()
            if error and error != self._seen_error:
                self._seen_error = error
                self.status.set(f"相机连接/取图失败：{error}。若设备被占用，请先退出 Agent 或其他相机程序，再重新连接。")
            if self.worker.running and not self.worker.stop_event.is_set() and frame is not self.live_frame:
                self.live_frame = frame
                if frame and self.frozen is None:
                    self.draw()
                    self.status.set("实时预览中。调整物品朝向后点击“拍摄 / 冻结”，然后框选。")
        if self.display_frame is None and self.live_frame:
            self.draw()
        self.sync_buttons()
        self.root.after(80, self.poll)

    def capture(self):
        if self.frozen is not None:
            return
        frame = self.display_frame
        if frame is None:
            self.status.set("尚无画面，请先连接相机或打开本地图片。")
            return
        if frame.metadata.get("source") == "realsense_rgb" and (
            not self.worker or not self.worker.running or self.worker.stop_event.is_set()
            or time.monotonic() - frame.received_at > 2
        ):
            self.status.set("当前相机画面已过期，请重新连接并等待新画面后再拍摄。")
            return
        self.load_object()
        self.frozen = PreviewFrame(frame.image.copy(), dict(frame.metadata), frame.received_at)
        self.bbox = None
        self.drag_start = None
        self.status.set("画面已冻结。拖动鼠标框选物品，核对名称和视角后保存。")
        self.draw()
        self.sync_buttons()

    def resume(self):
        self.frozen = None
        self.bbox = None
        self.drag_start = None
        self.status.set("已取消当前未保存照片，可以继续预览或重新拍摄。")
        self.draw()
        self.sync_buttons()

    def import_image(self):
        path = filedialog.askopenfilename(title="选择参考照片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp"), ("所有文件", "*")])
        if not path:
            return
        try:
            with Image.open(path) as image:
                frame = PreviewFrame(image.convert("RGB").copy(),
                    {"source": "local_image", "original_path": str(Path(path).resolve())}, time.monotonic())
            self.frozen = frame
            self.bbox = None
            self.drag_start = None
            self.draw()
            self.sync_buttons()
            self.status.set("本地图片已冻结；请框选目标并填写名称、视角，再保存。")
        except Exception as exc:
            messagebox.showerror("无法打开图片", str(exc), parent=self.root)

    def draw(self):
        frame = self.frozen or self.live_frame
        self.canvas.delete("all")
        self.display_frame = frame
        if frame is None:
            self.canvas.create_text(max(1, self.canvas.winfo_width()) / 2,
                max(1, self.canvas.winfo_height()) / 2, text="连接腕部相机，或打开本地图片", fill="white", font=self.fonts["body"])
            self.draw_crop()
            return
        self.viewport = ImageViewport(*frame.image.size, max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height()))
        self._photo = ImageTk.PhotoImage(frame.image.resize(self.viewport.display_size, Image.Resampling.BILINEAR))
        self.canvas.create_image(*self.viewport.offset, image=self._photo, anchor="nw")
        if self.bbox:
            x1, y1, x2, y2 = self.bbox
            self.canvas.create_rectangle(*self.viewport.to_canvas(x1, y1), *self.viewport.to_canvas(x2, y2),
                                         outline="#20e7ae", width=3)
        self.draw_crop()

    def draw_crop(self):
        self.crop_canvas.delete("all")
        if self.frozen and self.bbox:
            box = [round(v) for v in self.bbox]
            if box[2] > box[0] and box[3] > box[1]:
                crop = self.frozen.image.crop(box)
                width, height = int(self.crop_canvas["width"]), int(self.crop_canvas["height"])
                crop.thumbnail((width, height), Image.Resampling.LANCZOS)
                self._crop_photo = ImageTk.PhotoImage(crop)
                self.crop_canvas.create_image(width / 2, height / 2, image=self._crop_photo)
                self.selection_text.set(f"目标裁图 {box[2]-box[0]} × {box[3]-box[1]} 像素")
                return
        self.selection_text.set("尚未框选")

    def begin_box(self, event):
        if self.frozen is None or self.viewport is None:
            return
        self.drag_start = self.viewport.to_image(event.x, event.y)
        self.bbox = None
        self.draw()
        self.sync_buttons()

    def drag_box(self, event):
        if self.drag_start is None:
            return
        end = self.viewport.to_image(event.x, event.y, clamp=True)
        self.bbox = [min(self.drag_start[0], end[0]), min(self.drag_start[1], end[1]),
                     max(self.drag_start[0], end[0]), max(self.drag_start[1], end[1])]
        self.draw()

    def end_box(self, event):
        self.drag_box(event)
        self.drag_start = None
        if self.bbox:
            box = [round(v) for v in self.bbox]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4:
                self.bbox = None
                self.status.set("选框太小，请重新框选一个完整物品。")
        self.draw()
        self.sync_buttons()

    def clear_box(self):
        self.bbox = None
        self.drag_start = None
        self.draw()
        self.sync_buttons()

    def full_image(self):
        if self.frozen:
            self.bbox = [0, 0, self.frozen.image.width, self.frozen.image.height]
            self.draw()
            self.sync_buttons()

    def save(self):
        if not self.frozen or self.bbox is None or self.drag_start is not None:
            self.status.set("请先拍摄并完成物品框选，然后再保存。")
            return
        try:
            fields = {key: var.get().strip() for key, var in self.form.items()}
            aliases = [s.strip() for s in re.split(r"[;；,，\n]", fields["aliases"]) if s.strip()]
            result = save_snapshot(self.gallery, frame=self.frozen, bbox=self.bbox,
                name=fields["name"], category=fields["category"], detector_prompt=fields["prompt"],
                viewpoint=VIEW_LABELS[fields["view"]], aliases=aliases,
                attributes={key: fields[key] for key in ("brand", "specification") if fields[key]},
                source_group=self._session)
            info = (f"新增成功 · 共 {result['reference_count']} 张" if result["added"] else
                    f"图片已存在，未重复追加 · 共 {result['reference_count']} 张（保留原视角标签）")
            self.history.insert("", 0, values=(result["name"], fields["view"], info))
            self.form["name"].set(result["name"])
            self.refresh_names()
            self.load_object()
            self.resume()
            self.status.set(f"{result['name']}：{info}。可更换视角继续拍摄；新图片会在下次识别时更新特征缓存。")
        except Exception as exc:
            self.status.set(f"保存失败：{exc}。照片仍保留在窗口中，可修改后再保存。")
            messagebox.showerror("保存失败", str(exc), parent=self.root)

    def close(self):
        self._closing = True
        if self.worker:
            self.worker.stop()
        self.root.destroy()

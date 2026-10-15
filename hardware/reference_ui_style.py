"""Font setup for the photo tool; optional private Tk avoids changing Conda's Tk package."""
import ctypes
import os
from pathlib import Path
import sys

_runtime_handles = []


def prepare_tk_runtime():
    """Load the same-version Xft-enabled Tk before Python loads _tkinter (Linux only)."""
    if _runtime_handles:
        return True
    if not sys.platform.startswith("linux") or "_tkinter" in sys.modules:
        return False
    prefix = Path(sys.prefix)
    runtime = prefix / "opt/tk-xft-8.6.15"
    library = runtime / "lib/libtk8.6.so"
    scripts = runtime / "lib/tk8.6"
    if not library.is_file() or not (scripts / "tk.tcl").is_file():
        return False
    tcl = ctypes.CDLL(str(prefix / "lib/libtcl8.6.so"), mode=ctypes.RTLD_GLOBAL)
    tk = ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    _runtime_handles.extend((tcl, tk))
    os.environ["TK_LIBRARY"] = str(scripts)
    return True


def configure_fonts(root):
    from tkinter import font, ttk
    available = set(font.families(root))
    family = next((name for name in ("Noto Sans CJK SC", "Noto Sans SC", "Microsoft YaHei",
                                    "WenQuanYi Micro Hei", "Droid Sans Fallback") if name in available), "sans")
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont",
                 "TkSmallCaptionFont", "TkIconFont", "TkTooltipFont"):
        font.nametofont(name, root=root).configure(family=family, size=11, weight="normal")
    fonts = {"body": font.nametofont("TkDefaultFont", root=root),
             "title": font.Font(root=root, family=family, size=20, weight="bold"),
             "hint": font.Font(root=root, family=family, size=10),
             "heading": font.Font(root=root, family=family, size=11, weight="bold")}
    root.option_add("*Font", str(fonts["body"]))
    root.option_add("*TCombobox*Listbox.font", str(fonts["body"]))
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    style.configure(".", font=fonts["body"])
    style.configure("TButton", font=fonts["body"], padding=(8, 4))
    style.configure("TEntry", font=fonts["body"], padding=(4, 1))
    style.configure("TCombobox", font=fonts["body"], padding=(4, 1))
    style.configure("Treeview", font=fonts["body"], rowheight=fonts["body"].metrics("linespace") + 8)
    style.configure("Treeview.Heading", font=fonts["heading"])
    return fonts

"""
Constellation GUI

Fixes in this version:
- Scrollable main card (so enabling Alignment/Angle toggles never hides the Run button)
- Splash start button no longer "jumps" on startup (single draw path; no duplicate show_splash)
"""

from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Dict, List, Optional
import threading

import tkinter.font as tkfont

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk

from map_analysis import run_analysis, get_protein_pairs_with_distances

# -----------------------------
# Paths / resources
# -----------------------------

def here(*parts: str) -> str:
    # When packaged (PyInstaller), files live in sys._MEIPASS
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)


LOGO_PATH = here("images", "logo.png")
BG_PATH = here("images", "background.png")
LOGO_CIRCLE_PATH = here("images", "logo_circle.png")
ICON_FOLDER_PATH = here("images", "icon_folder.png")
ICON_FILE_PATH   = here("images", "icon_file.png")
START_PATH  = here("images", "start.png")
START_HOVER_PATH   = here("images", "start_hover.png")
HEADER_STRIP_PATH = here("images", "header_strip.png")
DIVIDER_PATH = here("images", "divider.png")
ICON_APP_PATH = here("images", "app_icon.ico")




def load_pil(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")

def resize_image_keep_ratio(img: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    img_w, img_h = img.size
    scale = min(canvas_w / img_w, canvas_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)

def resize_image_cover(img: Image.Image, canvas_w: int, canvas_h: int) -> Image.Image:
    """Resize image to cover canvas while preserving aspect ratio (may crop)."""
    img_w, img_h = img.size
    scale = max(canvas_w / img_w, canvas_h / img_h)
    new_w = max(1, int(img_w * scale))
    new_h = max(1, int(img_h * scale))
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    left = max(0, (new_w - canvas_w) // 2)
    top = max(0, (new_h - canvas_h) // 2)
    right = left + canvas_w
    bottom = top + canvas_h
    return resized.crop((left, top, right, bottom))


# -----------------------------
# App state
# -----------------------------

available_pairs: List[str] = []
all_proteins: List[str] = []
protein_location_vars: Dict[str, tk.StringVar] = {}

LOCATION_OPTIONS = [
    "Presynaptic cytosol",
    "Presynaptic membrane",
    "Postsynaptic membrane",
    "Postsynaptic cytosol",
]


# -----------------------------
# UI helpers
# -----------------------------

def open_folder(path: str) -> None:
    if not path:
        return
    try:
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
    except Exception:
        messagebox.showerror("Error", f"Could not open folder:\n{path}")


# -----------------------------
# Build UI
# -----------------------------

app = tb.Window(themename="flatly")
try:
    if os.name == "nt" and ICON_APP_PATH.lower().endswith(".ico"):
        # Windows: .ico works best
        app.iconbitmap(ICON_APP_PATH)
    else:
        # Cross-platform: use iconphoto (works great with .png)
        icon_img = ImageTk.PhotoImage(load_pil(ICON_APP_PATH).resize((64, 64), Image.LANCZOS))
        app.iconphoto(True, icon_img)
        app._icon_ref = icon_img  # keep a reference (prevents garbage collection)
except Exception as e:
    print(f"Could not set app icon: {e}")

app.title("Constellation 1.01")
app.minsize(980, 650)
style = tb.Style()
style.configure("White.TFrame", background="white")

# Start fullscreen (entire monitor)

app.attributes("-fullscreen", True)

# Optional: let user exit / toggle fullscreen
app.bind("<Escape>", lambda e: app.attributes("-fullscreen", False))
app.bind("<F11>",    lambda e: app.attributes("-fullscreen", not app.attributes("-fullscreen")))


# Canvas background (full window)
bg_canvas = tk.Canvas(app, highlightthickness=0)
bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

_bg_pil = load_pil(BG_PATH) if os.path.exists(BG_PATH) else Image.new("RGBA", (10, 10), (230, 230, 230, 255))
_logo_pil = load_pil(LOGO_PATH) if os.path.exists(LOGO_PATH) else Image.new("RGBA", (10, 10), (40, 40, 40, 255))
_logo_circle_pil = load_pil(LOGO_CIRCLE_PATH) if os.path.exists(LOGO_CIRCLE_PATH) else Image.new("RGBA", (10, 10), (40, 40, 40, 255))
_icon_folder_tk = ImageTk.PhotoImage(load_pil(ICON_FOLDER_PATH).resize((18,18), Image.LANCZOS))
_icon_file_tk   = ImageTk.PhotoImage(load_pil(ICON_FILE_PATH).resize((18,18), Image.LANCZOS))
_start_normal_pil = load_pil(START_PATH) if os.path.exists(START_PATH) else Image.new("RGBA", (10, 10), (0, 0, 0, 0))
_start_hover_pil  = load_pil(START_HOVER_PATH) if os.path.exists(START_HOVER_PATH) else _start_normal_pil
_divider_tk = ImageTk.PhotoImage(load_pil(DIVIDER_PATH))

bg_photo = None

def redraw_background(*_):
    global bg_photo
    bg_canvas.update_idletasks()
    w = bg_canvas.winfo_width()
    h = bg_canvas.winfo_height()
    if w < 2 or h < 2:
        return
    img = resize_image_cover(_bg_pil, w, h)
    bg_photo = ImageTk.PhotoImage(img)
    bg_canvas.delete("bg")
    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw", tags="bg")

_bg_job = None
def schedule_bg_redraw(event=None):
    global _bg_job
    if _bg_job is not None:
        app.after_cancel(_bg_job)
    _bg_job = app.after(60, redraw_background)  # 60ms debounce


bg_canvas.bind("<Configure>", schedule_bg_redraw)
app.bind("<Configure>", schedule_bg_redraw)

app.withdraw()
# -----------------------------
# Card in the center (scrollable content + pinned footer)
# -----------------------------

card = tb.Frame(app, padding=0, bootstyle="light")
card.configure(style="White.TFrame")

def show_main_ui():
    card.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.88, relheight=0.93)

    # Let Tk draw widgets immediately (so Run Analysis shows right away)
    app.update_idletasks()

    # Do heavy work AFTER the first paint
    def _finalize():
        card_canvas.itemconfig(content_window, width=card_canvas.winfo_width())
        _update_scrollregion()
        redraw_header_bg()
        redraw_background()

    app.after_idle(_finalize)


# Grid: [0] scrollable content, [1] pinned footer
card.grid_rowconfigure(0, weight=1)
card.grid_rowconfigure(1, weight=0)
card.grid_columnconfigure(0, weight=1)
card.grid_columnconfigure(1, weight=0)

# Scroll canvas lives in row=0
card_canvas = tk.Canvas(card, highlightthickness=0, bd=0)
card_canvas.grid(row=0, column=0, sticky="nsew")

vscroll = tb.Scrollbar(card, orient="vertical", command=card_canvas.yview)
vscroll.grid(row=0, column=1, sticky="ns")

card_canvas.configure(yscrollcommand=vscroll.set)

# Inner content frame (all main widgets go here)
content_frame = tb.Frame(card_canvas, padding=(16, 10, 16, 14), bootstyle="light")
content_frame.configure(style="White.TFrame")
content_window = card_canvas.create_window((0, 0), window=content_frame, anchor="nw")

def _update_scrollregion(*_):
    card_canvas.update_idletasks()
    bbox = card_canvas.bbox("all")
    if bbox:
        card_canvas.configure(scrollregion=bbox)

def _match_inner_width(event):
    # Keep inner frame width matched to canvas width so we don't get horizontal scrolling.
    card_canvas.itemconfig(content_window, width=event.width)

content_frame.bind("<Configure>", _update_scrollregion)
card_canvas.bind("<Configure>", _match_inner_width)

def _can_scroll() -> bool:
    # True only if content is taller than the visible canvas.
    sr = card_canvas.cget("scrollregion")
    if not sr:
        return False
    x1, y1, x2, y2 = map(float, sr.split())
    return (y2 - y1) > max(1, card_canvas.winfo_height())

# Mouse wheel scrolling (clamped, only when pointer is over the card content)
def _on_mousewheel(event):
    if not _can_scroll():
        return

    w = app.winfo_containing(event.x_root, event.y_root)
    if w is None:
        return

    over = (
        (w == card_canvas)
        or str(w).startswith(str(card_canvas))
        or str(w).startswith(str(content_frame))
    )
    if not over:
        return

    first, last = card_canvas.yview()

    # Windows/macOS: event.delta (+ = scroll up)
    if getattr(event, "delta", 0):
        if event.delta > 0 and first <= 0.0:
            return
        if event.delta < 0 and last >= 1.0:
            return
        card_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return

    # Linux: Button-4 (up) / Button-5 (down)
    if event.num == 4:
        if first <= 0.0:
            return
        card_canvas.yview_scroll(-3, "units")
    elif event.num == 5:
        if last >= 1.0:
            return
        card_canvas.yview_scroll(3, "units")

card_canvas.bind_all("<MouseWheel>", _on_mousewheel)
card_canvas.bind_all("<Button-4>", _on_mousewheel)
card_canvas.bind_all("<Button-5>", _on_mousewheel)

# -----------------------------
# Header (logo + title)
# -----------------------------

header_frame = tb.Frame(content_frame)
header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
header_frame.columnconfigure(0, weight=1)

_header_strip_pil = load_pil(HEADER_STRIP_PATH)
_header_strip_tk = None

header_bg = tb.Label(header_frame)
header_bg.place(x=0, y=0, relwidth=1, relheight=1)
header_bg.lower()

def redraw_header_bg(event=None):
    def _rgb_to_hex(rgb):
        return "#%02x%02x%02x" % rgb[:3]

    global _header_strip_tk
    w = header_frame.winfo_width()
    h = header_frame.winfo_height()
    if w < 2 or h < 2:
        return
    img = resize_image_cover(_header_strip_pil, w, h)
    _header_strip_tk = ImageTk.PhotoImage(img)
    header_bg.configure(image=_header_strip_tk)

    # pick colors from the strip (left for logo, center for title)
    c_logo  = _rgb_to_hex(img.getpixel((20, h // 2)))
    c_title = _rgb_to_hex(img.getpixel((w // 2, h // 2)))

    logo_circle_label.configure(bg=c_logo)
    title.configure(bg=c_title, fg="#1B4B7A")


header_frame.bind("<Configure>", redraw_header_bg)


logo_circle_img = resize_image_keep_ratio(_logo_circle_pil, 120, 60)
logo_circle_tk = ImageTk.PhotoImage(logo_circle_img)
logo_circle_label = tk.Label(header_frame, image=logo_circle_tk)
logo_circle_label.image = logo_circle_tk
logo_circle_label.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(16, 6))
header_frame.columnconfigure(1, weight=1)

title = tk.Label(
    header_frame,
    text="Proteome Analysis",
    font=("Avenir", 16, "bold"),
    fg="#1B4B7A",
    bd=0,
    highlightthickness=0
)
title.grid(row=0, column=1, sticky="w", padx=(12, 0), pady=(16, 6))

header_frame.bind("<Configure>", redraw_header_bg)
app.after_idle(redraw_header_bg)


content_frame.columnconfigure(0, weight=1)

# -----------------------------
# Variables
# -----------------------------

data_folder_var = tk.StringVar(value="")
expansion_file_var = tk.StringVar(value="")
num_proteins_var = tk.StringVar(value="4")

alignment_var = tk.BooleanVar(value=True)
angle_var = tk.BooleanVar(value=False)
enrichment_var = tk.BooleanVar(value=False)
enrichment_cutoff_var = tk.StringVar(value="0.3")  # µm (300 nm)

distance_cutoff_var = tk.StringVar(value="0.3")  # µm (cluster cutoff; 300 nm = 0.3)


reference_pair_var = tk.StringVar(value="")
alignment_tolerance_var = tk.StringVar(value="0.2")  # fraction (0.1 = 10%)

# Angle analysis vars
protein_A_var = tk.StringVar(value="")
protein_B_var = tk.StringVar(value="")
protein_C_var = tk.StringVar(value="")
angle_a_var = tk.StringVar(value="180")
angle_b_var = tk.StringVar(value="180")
angle_c_var = tk.StringVar(value="180")


# -----------------------------
# Folder pickers + layout (Two-column)
# -----------------------------

def select_data_folder():
    folder = filedialog.askdirectory(title="Select Data Folder")
    if not folder:
        return
    data_folder_var.set(folder)
    refresh_pairs_and_locations()

def select_expansion_file():
    f = filedialog.askopenfilename(
        title="Select Expansion Factor File",
        filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
    )
    if not f:
        return
    expansion_file_var.set(f)

# Main body (2 columns) under the header
body = tb.Frame(content_frame)
body.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
content_frame.rowconfigure(1, weight=1)
body.columnconfigure(0, weight=1, uniform="cols")
body.columnconfigure(1, weight=1, uniform="cols")
body.rowconfigure(0, weight=1)

left = tb.Frame(body, padding=(0, 0, 10, 0))
right = tb.Frame(body, padding=(10, 0, 0, 0))
left.grid(row=0, column=0, sticky="nsew")
right.grid(row=0, column=1, sticky="nsew")
left.columnconfigure(0, weight=1, minsize=440)
right.columnconfigure(0, weight=1)

# ---- Left column: setup + parameters ----
tb.Label(left, text="Data inputs", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))

tb.Button(
    left,
    text="Select Data Folder",
    image=_icon_folder_tk,
    compound="left",
    command=select_data_folder,
    bootstyle="secondary",
    padding=(14, 10),
).grid(row=1, column=0, sticky="ew")

tb.Label(left, textvariable=data_folder_var, wraplength=420).grid(row=2, column=0, sticky="w", pady=(4, 10))


tb.Button(
    left,
    text="Select Expansion Factor File",
    image=_icon_file_tk,
    compound="left",
    command=select_expansion_file,
    bootstyle="secondary",
    padding=(14, 10),
).grid(row=3, column=0, sticky="ew")

tb.Label(left, textvariable=expansion_file_var, wraplength=420).grid(row=4, column=0, sticky="w", pady=(4, 10))

tb.Separator(left).grid(row=5, column=0, sticky="ew", pady=(6, 10))

tb.Label(left, text="Analyses", font=("Segoe UI", 13, "bold")).grid(row=6, column=0, sticky="w", pady=(0, 6))
tb.Checkbutton(left, text="Enable Alignment Analysis", variable=alignment_var).grid(row=7, column=0, sticky="w")
tb.Checkbutton(left, text="Enable Angle Analysis", variable=angle_var).grid(row=8, column=0, sticky="w")
tb.Checkbutton(left, text="Enable Enrichment Analysis", variable=enrichment_var).grid(row=9, column=0, sticky="w", pady=(0, 10))

tb.Separator(left).grid(row=10, column=0, sticky="ew", pady=(6, 10))

tb.Label(left, text="Parameters", font=("Segoe UI", 13, "bold")).grid(row=11, column=0, sticky="w", pady=(0, 6))
tb.Label(left, text="Number of Proteins (3 or 4)").grid(row=12, column=0, sticky="w")
tb.Entry(left, textvariable=num_proteins_var, width=8).grid(row=13, column=0, sticky="w", pady=(0, 10))
tb.Label(left, text="Cluster distance cutoff (µm)  (300 nm = 0.3)").grid(row=14, column=0, sticky="w")
dist_entry = tb.Entry(left, textvariable=distance_cutoff_var, width=8)
dist_entry.grid(row=15, column=0, sticky="w", pady=(0, 10))

tb.Label(left, text="Enrichment cutoff (µm)  (300 nm = 0.3)").grid(row=16, column=0, sticky="w")
enrich_entry = tb.Entry(left, textvariable=enrichment_cutoff_var, width=8)
enrich_entry.grid(row=17, column=0, sticky="w", pady=(0, 10))

# ---- Right column: analysis panels ----
right.rowconfigure(0, weight=1)
right.rowconfigure(1, weight=1)

alignment_frame = tb.Labelframe(right, text="Alignment analysis", padding=10)
alignment_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
alignment_frame.columnconfigure(0, weight=1)

tb.Label(alignment_frame, text="Select the Longest Protein Pair").grid(row=0, column=0, sticky="w")
ref_pair_cb = tb.Combobox(alignment_frame, textvariable=reference_pair_var, values=[], state="disabled", width=30, height=8)
ref_pair_cb.grid(row=1, column=0, sticky="w", pady=(2, 10))

protein_location_outer = tb.Labelframe(alignment_frame, text="Protein locations", padding=8)
protein_location_outer.grid(row=2, column=0, sticky="nsew", pady=(0, 8))
protein_location_outer.columnconfigure(0, weight=1)
protein_location_outer.rowconfigure(0, weight=1)

_pl_canvas = tk.Canvas(protein_location_outer, highlightthickness=0, height=220)
_pl_canvas.grid(row=0, column=0, sticky="nsew")
_pl_scroll = tb.Scrollbar(protein_location_outer, orient="vertical", command=_pl_canvas.yview)
_pl_scroll.grid(row=0, column=1, sticky="ns")
_pl_canvas.configure(yscrollcommand=_pl_scroll.set)

_pl_inner = tb.Frame(_pl_canvas)
_pl_window = _pl_canvas.create_window((0, 0), window=_pl_inner, anchor="nw")

def _pl_update_scroll(*_):
    _pl_canvas.update_idletasks()
    _pl_canvas.configure(scrollregion=_pl_canvas.bbox("all"))

def _pl_match_width(event):
    _pl_canvas.itemconfig(_pl_window, width=event.width)

_pl_inner.bind("<Configure>", _pl_update_scroll)
_pl_canvas.bind("<Configure>", _pl_match_width)

protein_location_frame = _pl_inner

tb.Label(alignment_frame, text="Alignment tolerance (fraction, e.g. 0.1 = 10%)").grid(row=3, column=0, sticky="w")
tb.Entry(alignment_frame, textvariable=alignment_tolerance_var, width=10).grid(row=4, column=0, sticky="w")

angle_frame = tb.Labelframe(right, text="Angle analysis", padding=10)
angle_frame.grid(row=1, column=0, sticky="nsew")
angle_frame.columnconfigure(1, weight=1)

tb.Label(angle_frame, text="Pick 3 proteins to form a triangle (A-B, A-C, B-C distances).").grid(
    row=0, column=0, columnspan=2, sticky="w", pady=(0, 6)
)

tb.Label(angle_frame, text="Protein A").grid(row=1, column=0, sticky="w")
angleA_cb = tb.Combobox(angle_frame, textvariable=protein_A_var, values=[], state="disabled", width=30, height=8)
angleA_cb.grid(row=1, column=1, sticky="ew", pady=2)

tb.Label(angle_frame, text="Protein B").grid(row=2, column=0, sticky="w")
angleB_cb = tb.Combobox(angle_frame, textvariable=protein_B_var, values=[], state="disabled", width=30, height=8)
angleB_cb.grid(row=2, column=1, sticky="ew", pady=2)

tb.Label(angle_frame, text="Protein C").grid(row=3, column=0, sticky="w")
angleC_cb = tb.Combobox(angle_frame, textvariable=protein_C_var, values=[], state="disabled", width=30, height=8)
angleC_cb.grid(row=3, column=1, sticky="ew", pady=(2, 8))

tb.Label(angle_frame, text="Max angle at A (deg)").grid(row=4, column=0, sticky="w")
tb.Entry(angle_frame, textvariable=angle_a_var, width=10).grid(row=4, column=1, sticky="w", pady=2)

tb.Label(angle_frame, text="Max angle at B (deg)").grid(row=5, column=0, sticky="w")
tb.Entry(angle_frame, textvariable=angle_b_var, width=10).grid(row=5, column=1, sticky="w", pady=2)

tb.Label(angle_frame, text="Max angle at C (deg)").grid(row=6, column=0, sticky="w")
tb.Entry(angle_frame, textvariable=angle_c_var, width=10).grid(row=6, column=1, sticky="w", pady=2)

# -----------------------------
# Footer (always visible)
# -----------------------------

results_dir_var = tk.StringVar(value="")

def open_results_folder():
    open_folder(results_dir_var.get())

footer = tb.Frame(card, padding=(16, 10, 16, 16), bootstyle="light")
footer.grid(row=1, column=0, columnspan=2, sticky="ew")
footer.columnconfigure(0, weight=1)
footer.configure(style="White.TFrame")

run_btn = tb.Button(footer, text="Run Analysis", bootstyle="success")
run_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

open_results_btn = tb.Button(footer, text="Open Results Folder", command=open_results_folder, bootstyle="info")
open_results_btn.grid(row=0, column=1, sticky="ew", padx=(0, 8))
open_results_btn.grid_remove()

progress = tb.Progressbar(footer, mode="indeterminate")
progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
progress.grid_remove()



created_by_lbl = tk.Label(
    footer,
    text="Created by: Rebecca E. Twilley",
    fg="#9AA0A6",
    bg="#F8F9FA",  # must match the footer background
    font=("Segoe UI", 9)
)
created_by_lbl.grid(row=2, column=0, columnspan=3, sticky="e", pady=(10, 0))


def reset_app_state():
    # If analysis is running, don't reset mid-run.
    try:
        if str(run_btn.cget("state")) == "disabled":
            messagebox.showwarning(
                "Busy",
                "Analysis is currently running. Please wait for it to finish before refreshing."
            )
            return
    except Exception:
        pass

    if not messagebox.askyesno(
            "Refresh",
            "Reset the app to a clean state?\n\n"
            "This will clear selected folders/files, reset parameters, and hide results."
    ):
        return

    try:
        progress.stop()
        progress.grid_remove()
    except Exception:
        pass

    results_dir_var.set("")
    try:
        open_results_btn.grid_remove()
    except Exception:
        pass

    data_folder_var.set("")
    expansion_file_var.set("")
    num_proteins_var.set("4")

    alignment_var.set(True)
    angle_var.set(False)
    enrichment_var.set(False)
    enrichment_cutoff_var.set("0.3")
    distance_cutoff_var.set("0.3")
    reference_pair_var.set("")
    alignment_tolerance_var.set("0.2")

    protein_A_var.set("")
    protein_B_var.set("")
    protein_C_var.set("")
    angle_a_var.set("180")
    angle_b_var.set("180")
    angle_c_var.set("180")

    global available_pairs, all_proteins
    available_pairs, all_proteins = [], []

    try:
        ref_pair_cb.configure(values=[], state="disabled")
    except Exception:
        pass

    rebuild_location_widgets()
    update_reference_pairs_from_locations()
    update_angle_widgets()

refresh_btn = tb.Button(footer, text="Refresh / Reset", command=reset_app_state, bootstyle="warning")
refresh_btn.grid(row=0, column=2, sticky="ew")

def rebuild_location_widgets():
    global protein_location_vars
    for child in list(protein_location_frame.winfo_children()):
        child.destroy()

    protein_location_vars = {}

    if not all_proteins:
        tb.Label(protein_location_frame, text="Select a data folder to load proteins.").grid(row=0, column=0, sticky="w")
        return

    for r, prot in enumerate(all_proteins):
        tb.Label(protein_location_frame, text=prot, width=12).grid(row=r, column=0, sticky="w", padx=(0, 8), pady=2)
        v = tk.StringVar(value=LOCATION_OPTIONS[0])
        protein_location_vars[prot] = v
        cb = tb.Combobox(protein_location_frame, textvariable=v, values=LOCATION_OPTIONS, state="readonly", width=26, height=8)
        cb.grid(row=r, column=1, sticky="ew", pady=2)

    def _on_loc_change(*_):
        update_reference_pairs_from_locations()

    for v in protein_location_vars.values():
        v.trace_add("write", _on_loc_change)


def _directed_pair_or_none(a: str, b: str) -> Optional[str]:
    """Return a directed pair string that exists in available_pairs, in either direction."""
    avail = set(available_pairs)
    if f"{a}-{b}" in avail:
        return f"{a}-{b}"
    if f"{b}-{a}" in avail:
        return f"{b}-{a}"
    return None


def _compute_longest_reference_candidates() -> List[str]:
    """
    Choose the "longest" reference-pair candidates based on synapse order:
    Presynaptic cytosol -> Presynaptic membrane -> Postsynaptic membrane -> Postsynaptic cytosol

    Priority for endpoints:
      1) pre-cytosol <-> post-cytosol (if both exist)
      2) cytosol side <-> opposite membrane (if only one cytosol side exists)
      3) pre-membrane <-> post-membrane (if no cytosol proteins exist)

    Returns a list of directed pair strings that exist in available_pairs.
    """
    if not protein_location_vars:
        return list(available_pairs)

    pre_cyt = [p for p, v in protein_location_vars.items() if v.get() == "Presynaptic cytosol"]
    pre_mem = [p for p, v in protein_location_vars.items() if v.get() == "Presynaptic membrane"]
    post_mem = [p for p, v in protein_location_vars.items() if v.get() == "Postsynaptic membrane"]
    post_cyt = [p for p, v in protein_location_vars.items() if v.get() == "Postsynaptic cytosol"]

    pre_all = pre_cyt + pre_mem
    post_all = post_mem + post_cyt

    # If everything is on one side, alignment doesn't make sense.
    if not pre_all or not post_all:
        return []

    endpoints: List[tuple[str, str]] = []

    # 1) cytosol-to-cytosol (outermost endpoints)
    if pre_cyt and post_cyt:
        endpoints = [(a, b) for a in pre_cyt for b in post_cyt]

    # 2) only one cytosol side populated -> cytosol to opposite membrane
    elif pre_cyt and post_mem:
        endpoints = [(a, b) for a in pre_cyt for b in post_mem]
    elif post_cyt and pre_mem:
        endpoints = [(a, b) for a in pre_mem for b in post_cyt]

    # 3) no cytosol proteins -> membrane-to-membrane across synapse
    elif pre_mem and post_mem:
        endpoints = [(a, b) for a in pre_mem for b in post_mem]

    # Convert to directed candidates present in available_pairs
    out: List[str] = []
    for a, b in endpoints:
        pair = _directed_pair_or_none(a, b)
        if pair and pair not in out:
            out.append(pair)

    return out


def update_reference_pairs_from_locations():
    if not alignment_var.get():
        return
    if not available_pairs:
        ref_pair_cb.configure(values=[], state="disabled")
        reference_pair_var.set("")
        return

    # If we don't have location info yet, fall back to showing all pairs.
    if not protein_location_vars:
        ref_pair_cb.configure(values=available_pairs, state="readonly")
        if not reference_pair_var.get() and available_pairs:
            reference_pair_var.set(available_pairs[0])
        return

    candidates = _compute_longest_reference_candidates()

    # If alignment isn't applicable (e.g., all proteins presynaptic OR all postsynaptic),
    # disable the selector so the user isn't forced to pick a meaningless pair.
    if not candidates:
        ref_pair_cb.configure(values=["(Alignment not applicable: all proteins on one side)"], state="disabled")
        reference_pair_var.set("")
        return

    # If multiple "longest" candidates exist (e.g., >1 protein in a cytosol location),
    # add a combined option that will run BOTH alignments and merge the results.
    values: List[str] = []
    if len(candidates) > 1:
        values.append(" OR ".join(candidates))
    values.extend(candidates)

    ref_pair_cb.configure(values=values, state="readonly", height=min(10, max(5, len(values))))
    if reference_pair_var.get() not in values:
        reference_pair_var.set(values[0])


def update_angle_widgets():
    vals = list(all_proteins) if all_proteins else []

    # Enable only when Angle Analysis is on AND we actually have proteins
    new_state = "readonly" if (angle_var.get() and vals) else "disabled"

    angleA_cb.configure(values=vals, state=new_state)
    angleB_cb.configure(values=vals, state=new_state)
    angleC_cb.configure(values=vals, state=new_state)

    if not vals:
        return

    # Set defaults if current selections aren't valid
    if protein_A_var.get() not in vals:
        protein_A_var.set(vals[0])
    if protein_B_var.get() not in vals:
        protein_B_var.set(vals[1] if len(vals) > 1 else vals[0])
    if protein_C_var.get() not in vals:
        protein_C_var.set(vals[2] if len(vals) > 2 else vals[0])


def toggle_alignment_ui(*_):
    if alignment_var.get():
        alignment_frame.grid()
        rebuild_location_widgets()
        update_reference_pairs_from_locations()
    else:
        alignment_frame.grid_remove()
    _update_scrollregion()

def toggle_angle_ui(*_):
    if angle_var.get():
        angle_frame.grid()
        update_angle_widgets()
    else:
        angle_frame.grid_remove()
    _update_scrollregion()

alignment_var.trace_add("write", toggle_alignment_ui)
angle_var.trace_add("write", toggle_angle_ui)

def refresh_pairs_and_locations():
    global available_pairs, all_proteins

    folder = data_folder_var.get().strip()
    if not folder:
        available_pairs, all_proteins = [], []
        ref_pair_cb.configure(values=[])
        update_angle_widgets()
        rebuild_location_widgets()
        _update_scrollregion()
        return

    # Optional: disable the button while loading so user can’t click twice
    # (only if you saved the button widget to a variable)
    # data_folder_btn.configure(state="disabled")

    def worker():
        try:
            pairs, proteins = get_protein_pairs_with_distances(folder)
        except Exception as e:
            pairs, proteins = [], []
            err = str(e)

            def show_err():
                messagebox.showerror("Data folder error", f"Could not read files in:\n{folder}\n\n{err}")

            app.after(0, show_err)

        def apply_updates():
            global available_pairs, all_proteins
            available_pairs = pairs
            all_proteins = proteins

            # your existing UI updates:
            update_angle_widgets()
            rebuild_location_widgets()
            update_reference_pairs_from_locations()
            _update_scrollregion()

            # data_folder_btn.configure(state="normal")

        app.after(0, apply_updates)

    threading.Thread(target=worker, daemon=True).start()

# -----------------------------
# Run analysis
# -----------------------------

def run_clicked():
    folder = data_folder_var.get().strip()
    exp = expansion_file_var.get().strip()

    if not folder:
        messagebox.showerror("Error", "Please select a Data Folder.")
        return
    if not exp:
        messagebox.showerror("Error", "Please select an Expansion Factor file.")
        return

    try:
        n_prot = int(num_proteins_var.get())
    except ValueError:
        messagebox.showerror("Error", "Number of Proteins must be 3 or 4.")
        return
    # Cluster distance cutoff (µm)
    dist_cutoff_str = distance_cutoff_var.get().strip() or "0.3"
    try:
        dist_cutoff = float(dist_cutoff_str)
    except ValueError:
        messagebox.showerror("Error", "Cluster distance cutoff must be a number (µm), e.g. 0.3.")
        return
    if dist_cutoff <= 0:
        messagebox.showerror("Error", "Cluster distance cutoff must be > 0.")
        return


    use_align = bool(alignment_var.get())
    use_angle = bool(angle_var.get())
    use_enrich = bool(enrichment_var.get())

    ref_pair = reference_pair_var.get().strip() if use_align else None

    tol = None
    if use_align:
        try:
            tol = float(alignment_tolerance_var.get())
        except ValueError:
            messagebox.showerror("Error", "Alignment tolerance must be a number like 0.1 (10%).")
            return
        if tol < 0:
            messagebox.showerror("Error", "Alignment tolerance must be >= 0.")
            return
        if not ref_pair:
            messagebox.showerror("Error", "Please choose a reference pair (longest pair).")
            return

    prot_locs = None
    if use_align:
        prot_locs = {p: v.get() for p, v in protein_location_vars.items()}
        if not prot_locs:
            messagebox.showerror("Error", "Protein locations are missing. Select a data folder first.")
            return

    protA = protB = protC = None
    angA = angB = angC = 180.0
    if use_angle:
        protA = protein_A_var.get().strip()
        protB = protein_B_var.get().strip()
        protC = protein_C_var.get().strip()
        if not (protA and protB and protC):
            messagebox.showerror("Error", "Angle analysis is ON but Protein A/B/C were not selected.")
            return
        if len({protA, protB, protC}) < 3:
            messagebox.showerror("Error", "Angle analysis needs 3 different proteins (A, B, C).")
            return
        try:
            angA = float(angle_a_var.get())
            angB = float(angle_b_var.get())
            angC = float(angle_c_var.get())
        except ValueError:
            messagebox.showerror("Error", "Angle limits must be numbers (degrees), e.g. 180.")
            return

    enrich_cutoff = None
    if use_enrich:
        try:
            enrich_cutoff = float(enrichment_cutoff_var.get())
        except ValueError:
            messagebox.showerror("Error", "Enrichment cutoff must be a number (µm), e.g. 0.3.")
            return
        if enrich_cutoff <= 0:
            messagebox.showerror("Error", "Enrichment cutoff must be > 0.")
            return

    run_btn.configure(state="disabled")
    progress.grid()
    progress.start(10)

    def worker():
        try:
            result = run_analysis(
                data_folder=folder,
                expansion_factor_file=exp,
                number_of_proteins=n_prot,
                distance_cutoff=dist_cutoff,
                use_enrichment_analysis=use_enrich,
                enrichment_cutoff=enrich_cutoff,
                use_alignment_analysis=use_align,
                reference_pair=ref_pair,
                protein_locations=prot_locs,
                alignment_tolerance=float(tol) if tol is not None else 0.1,
                use_angle_analysis=use_angle,
                protein_A=protA,
                protein_B=protB,
                protein_C=protC,
                angle_a=angA,
                angle_b=angB,
                angle_c=angC,
            )
            results_dir_var.set(result.get("results_dir", ""))
            summary_path = result.get("summary_path", "")
            enrich_path = result.get("enrichment_summary_path", "")
            msg = f"Analysis complete.\n\nSummary:\n{summary_path}"
            if enrich_path:
                msg += f"\n\nEnrichment summary:\n{enrich_path}"
            app.after(0, lambda: messagebox.showinfo("Done", msg))
            app.after(0, open_results_btn.grid)
        except Exception as e:
            app.after(0, lambda msg=str(e): messagebox.showerror("Error", msg))
        finally:
            def done_ui():
                progress.stop()
                progress.grid_remove()
                run_btn.configure(state="normal")
            app.after(0, done_ui)

    threading.Thread(target=worker, daemon=True).start()

run_btn.configure(command=run_clicked)


# -----------------------------
# Splash (logo + clickable start, drawn ONE way to avoid jumping)
# -----------------------------

splash_canvas = tk.Canvas(app, bg="white", highlightthickness=0)
splash_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
splash_canvas.tk.call("raise", splash_canvas._w)

splash_item = None
start_btn_item = None
splash_logo_photo = None
splash_start_photo = None
splash_credit_item = None
splash_start_hover_photo = None

def redraw_splash(event=None):
    global splash_item, start_btn_item, splash_logo_photo, splash_start_photo, splash_start_hover_photo, splash_credit_item
    if not splash_canvas.winfo_exists():
        return

    w = (event.width if event else splash_canvas.winfo_width())
    h = (event.height if event else splash_canvas.winfo_height())
    if w < 50 or h < 50:
        # don't draw until we have a real size -> prevents "jump"
        return
    # --- Credit at the bottom ---
    credit_y = h - 20  # distance from bottom of the window

    # logo
    logo_img = resize_image_keep_ratio(_logo_pil, int(w * 0.5), int(h * 0.5))
    splash_logo_photo = ImageTk.PhotoImage(logo_img)

    if splash_item:
        splash_canvas.itemconfig(splash_item, image=splash_logo_photo)
        splash_canvas.coords(splash_item, w // 2, h // 2 - 90)
    else:
        splash_item = splash_canvas.create_image(w // 2, h // 2 - 90, image=splash_logo_photo, anchor="center")

    # start button (normal + hover; scaled with window, consistent calc)
    max_w = int(w * 0.15)
    w0, h0 = _start_normal_pil.size
    scale = max_w / max(1, w0)
    new_size = (max(1, int(w0 * scale)), max(1, int(h0 * scale)))

    normal_img = _start_normal_pil.resize(new_size, Image.LANCZOS)
    hover_img  = _start_hover_pil.resize(new_size, Image.LANCZOS)

    splash_start_photo = ImageTk.PhotoImage(normal_img)
    splash_start_hover_photo = ImageTk.PhotoImage(hover_img)

    # keep refs to avoid garbage collection
    splash_canvas._start_normal_ref = splash_start_photo
    splash_canvas._start_hover_ref = splash_start_hover_photo

    y = (h // 2 - 40) + (logo_img.height // 2) - 20 + int(h * 0.1)

    if start_btn_item:
        splash_canvas.itemconfig(start_btn_item, image=splash_start_photo)
        splash_canvas.coords(start_btn_item, w // 2, y)
    else:
        start_btn_item = splash_canvas.create_image(
            w // 2, y, image=splash_start_photo, anchor="n", tags=("start_btn",)
        )

        def on_enter(_=None):
            splash_canvas.config(cursor="hand2")
            splash_canvas.itemconfig(start_btn_item, image=splash_start_hover_photo)

        def on_leave(_=None):
            splash_canvas.config(cursor="")
            splash_canvas.itemconfig(start_btn_item, image=splash_start_photo)

        splash_canvas.tag_bind("start_btn", "<Button-1>", lambda e: fade_to_background())
        splash_canvas.tag_bind("start_btn", "<Enter>", on_enter)
        splash_canvas.tag_bind("start_btn", "<Leave>", on_leave)

    # Created-by credit (bottom of splash)
    credit_y = h - 18
    if splash_credit_item:
        splash_canvas.coords(splash_credit_item, w // 2, credit_y)
    else:
        splash_credit_item = splash_canvas.create_text(
            w // 2,
            credit_y,
            text="Created by: Rebecca E. Twilley",
            fill="#9AA0A6",
            font=("Segoe UI", 10),
            anchor="s"
        )


_splash_is_fading = False
_fade_item = None
_fade_photo = None

def fade_to_background(steps: int = 20, delay_ms: int = 40):
    global _splash_is_fading, _fade_item, _fade_photo, start_btn_item, splash_item

    if _splash_is_fading:
        return
    _splash_is_fading = True

    w = splash_canvas.winfo_width()
    h = splash_canvas.winfo_height()
    if w < 50 or h < 50:
        w = app.winfo_width()
        h = app.winfo_height()

    # Build the CURRENT splash frame (white bg + logo + start) using SAME sizing math as redraw_splash()
    splash_frame = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    # Logo (match redraw_splash: 0.5 of window)
    logo_img = resize_image_keep_ratio(_logo_pil, int(w * 0.5), int(h * 0.5))
    logo_x = w // 2
    logo_y = h // 2 - 90
    splash_frame.paste(
        logo_img,
        (logo_x - logo_img.width // 2, logo_y - logo_img.height // 2),
        logo_img
    )

    # Start button (match redraw_splash placement)
    start_img = _start_normal_pil.copy()
    max_w = int(w * 0.15)
    w0, h0 = start_img.size
    scale = max_w / max(1, w0)
    start_img = start_img.resize(
        (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
        Image.LANCZOS
    )

    y = (h // 2 - 40) + (logo_img.height // 2) - 20 + int(h * 0.1)
    start_x = w // 2
    start_y = y  # anchor "n" in redraw_splash
    splash_frame.paste(
        start_img,
        (start_x - start_img.width // 2, start_y),
        start_img
    )

    # Target background frame
    bg_frame = resize_image_cover(_bg_pil, w, h).convert("RGBA")

    # Remove interactive start item so it can't be clicked during fade
    try:
        splash_canvas.config(cursor="")
        if start_btn_item:
            splash_canvas.delete(start_btn_item)
            start_btn_item = None
    except Exception:
        pass

    # Draw a dedicated fullscreen fade item anchored top-left (prevents bars/jumps)
    if _fade_item is None:
        _fade_photo = ImageTk.PhotoImage(splash_frame)
        _fade_item = splash_canvas.create_image(0, 0, image=_fade_photo, anchor="nw")
        # Optional: hide the old logo item if it exists
        try:
            if splash_item:
                splash_canvas.itemconfig(splash_item, state="hidden")
        except Exception:
            pass

    def step(i: int):
        global _fade_photo
        t = i / steps
        blended = Image.blend(splash_frame, bg_frame, t)
        _fade_photo = ImageTk.PhotoImage(blended)
        splash_canvas.itemconfig(_fade_item, image=_fade_photo)

        if i < steps:
            app.after(delay_ms, lambda: step(i + 1))
        else:
            splash_canvas.destroy()
            show_main_ui()

            # let Tk draw the UI first (so the Run button appears instantly)
            app.after_idle(redraw_background)

    step(0)


def start_splash_sequence():
    # Show window now that everything is built
    app.deiconify()
    app.update_idletasks()

    # Ensure splash canvas is top-most
    splash_canvas.tk.call("raise", splash_canvas._w)

    # Draw once, then bind resize updates
    redraw_splash()
    splash_canvas.bind("<Configure>", redraw_splash)

app.after_idle(start_splash_sequence)

# Ensure correct visibility at start
toggle_alignment_ui()
toggle_angle_ui()

app.mainloop()

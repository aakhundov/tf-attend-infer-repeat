import tkinter as tk
import tkinter.ttk as ttk

from .pixel_canvas import PixelCanvas


class DemoWindow(ttk.Frame):
    def __init__(self, master, **kw):
        ttk.Frame.__init__(self, master=master, **kw)

        self.frm_canvas = ttk.Frame(self, padding=(10, 10, 10, 10))
        self.cnv_demo = PixelCanvas(
            self.frm_canvas, 50, 50,
            highlightthickness=0, borderwidth=0,
            width=400, height=400
        )

        self.frm_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_demo.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.columnconfigure(0, weight=1, minsize=200)
        self.rowconfigure(0, weight=1, minsize=200)

        self.frm_canvas.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas.rowconfigure(0, weight=1, minsize=200)

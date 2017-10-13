import tkinter as tk
import tkinter.ttk as ttk

from .pixel_canvas import PixelCanvas


class DemoWindow(ttk.Frame):

    def __init__(self, master, model_wrapper,
                 canvas_size=50, refresh_period=50, **kw):

        ttk.Frame.__init__(self, master=master, **kw)

        self.master = master
        self.model_wrapper = model_wrapper
        self.canvas_size = canvas_size
        self.refresh_period = refresh_period

        self.frm_canvas_orig = ttk.Frame(self, padding=(10, 10, 5, 10))
        self.cnv_orig = PixelCanvas(
            self.frm_canvas_orig, self.canvas_size, self.canvas_size, drawable=True,
            highlightthickness=0, borderwidth=0, width=400, height=400
        )

        self.frm_canvas_orig.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_orig.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_canvas_orig.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas_orig.rowconfigure(0, weight=1, minsize=200)

        self.frm_canvas_rec = ttk.Frame(self, padding=(5, 10, 10, 10))
        self.cnv_rec = PixelCanvas(
            self.frm_canvas_rec, self.canvas_size, self.canvas_size, drawable=False,
            highlightthickness=0, borderwidth=0, width=400, height=400
        )

        self.frm_canvas_rec.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_rec.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_canvas_rec.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas_rec.rowconfigure(0, weight=1, minsize=200)

        self.columnconfigure(0, weight=1, minsize=200)
        self.columnconfigure(1, weight=1, minsize=200)
        self.rowconfigure(0, weight=1, minsize=200)

        self.master.after(100, self._reconstruct)

    def _reconstruct(self):
        dig, pos, rec, win, lat = self.model_wrapper.infer(
            [self.cnv_orig.get_image()]
        )

        self.cnv_rec.set_image(rec[0])
        self.cnv_rec.set_bbox_positions(pos[0])
        self.cnv_orig.set_bbox_positions(pos[0])

        self.master.after(self.refresh_period, self._reconstruct)

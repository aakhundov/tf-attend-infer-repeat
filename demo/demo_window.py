import tkinter as tk
import tkinter.ttk as ttk

from .pixel_canvas import PixelCanvas


class DemoWindow(ttk.Frame):

    def __init__(self, master, model_wrapper,
                 canvas_size=50, window_size=28,
                 refresh_period=50, **kw):

        ttk.Frame.__init__(self, master=master, **kw)

        self.master = master
        self.model_wrapper = model_wrapper
        self.canvas_size = canvas_size
        self.window_size = window_size
        self.refresh_period = refresh_period

        self._create_controls()

        self.columnconfigure(0, weight=410, minsize=215)
        self.columnconfigure(1, weight=410, minsize=210)
        self.columnconfigure(2, weight=140, minsize=65)
        self.rowconfigure(0, weight=1, minsize=220)

        self.master.after(50, lambda: master.focus_force())
        self.master.after(100, self._reconstruct)

    def _create_controls(self):
        self.frm_canvas_orig = ttk.Frame(self, padding=(10, 10, 5, 10))
        self.lbl_orig = ttk.Label(self.frm_canvas_orig, text="Original image [draw here]:")
        self.cnv_orig = PixelCanvas(
            self.frm_canvas_orig, self.canvas_size, self.canvas_size, drawable=True,
            highlightthickness=0, borderwidth=0, width=400, height=400
        )

        self.frm_canvas_orig.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.lbl_orig.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_orig.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_canvas_orig.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas_orig.rowconfigure(0, weight=0, minsize=20)
        self.frm_canvas_orig.rowconfigure(1, weight=1, minsize=200)

        self.frm_canvas_rec = ttk.Frame(self, padding=(5, 10, 5, 10))
        self.lbl_rec = ttk.Label(self.frm_canvas_rec, text="Reconstructed image:")
        self.cnv_rec = PixelCanvas(
            self.frm_canvas_rec, self.canvas_size, self.canvas_size, drawable=False,
            highlightthickness=0, borderwidth=0, width=400, height=400
        )

        self.frm_canvas_rec.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.lbl_rec.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_rec.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_canvas_rec.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas_rec.rowconfigure(0, weight=0, minsize=20)
        self.frm_canvas_rec.rowconfigure(1, weight=1, minsize=200)

        self.frm_windows = ttk.Frame(self, padding=(0, 0, 0, 0))
        self.frm_windows.grid(row=0, column=2, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_windows.columnconfigure(0, weight=1)

        self.frm_canvas_win, self.lbl_win, self.cnv_win = [], [], []

        for i in range(3):
            frm_canvas_win = ttk.Frame(
                self.frm_windows,
                padding=(5, 10 if i == 0 else 0, 10, 10 if i == 2 else 0)
            )
            lbl_win = ttk.Label(
                frm_canvas_win, text="VAE rec. {0}:".format(i+1)
            )
            cnv_win = PixelCanvas(
                frm_canvas_win, self.window_size, self.window_size, drawable=False,
                highlightthickness=0, borderwidth=0, width=120, height=120
            )

            frm_canvas_win.grid(row=i, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
            lbl_win.grid(row=0, column=0, sticky=(tk.S, tk.W))
            cnv_win.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
            frm_canvas_win.columnconfigure(0, weight=1, minsize=50)
            frm_canvas_win.rowconfigure(0, weight=0, minsize=20)
            frm_canvas_win.rowconfigure(1, weight=1, minsize=50)

            self.frm_windows.rowconfigure(i, weight=1)
            self.frm_canvas_win.append(frm_canvas_win)
            self.lbl_win.append(lbl_win)
            self.cnv_win.append(cnv_win)

    def _reconstruct(self):
        dig, pos, rec, win, lat = self.model_wrapper.infer(
            [self.cnv_orig.get_image()]
        )

        self.cnv_rec.set_image(rec[0])
        self.cnv_rec.set_bbox_positions(pos[0])
        self.cnv_orig.set_bbox_positions(pos[0])

        for i in range(len(self.cnv_win)):
            if i < len(win[0]):
                self.cnv_win[i].set_image(win[0][i])
                self.cnv_win[i].set_bbox_positions(
                    [[0.0, -2.0, -2.0]] * i + [[0.98, 0.0, 0.0]]
                )
            else:
                self.cnv_win[i].clear_image()
                self.cnv_win[i].set_bbox_positions([])

        self.master.after(self.refresh_period, self._reconstruct)

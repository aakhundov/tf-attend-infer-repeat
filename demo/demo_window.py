import tkinter as tk
import tkinter.ttk as ttk

from .pixel_canvas import PixelCanvas


class DemoWindow(ttk.Frame):

    def __init__(self, master, model_wrapper,
                 canvas_size=50, window_size=28,
                 refresh_period=50, test_image=None, **kw):

        ttk.Frame.__init__(self, master=master, **kw)

        self.master = master
        self.model_wrapper = model_wrapper
        self.canvas_size = canvas_size
        self.window_size = window_size
        self.refresh_period = refresh_period

        self._create_interface()

        if test_image is not None:
            self.cnv_orig.set_image(test_image)

        self.columnconfigure(0, weight=410, minsize=215)
        self.columnconfigure(1, weight=410, minsize=210)
        self.columnconfigure(2, weight=140, minsize=65)
        self.rowconfigure(0, weight=0, minsize=50)
        self.rowconfigure(1, weight=1, minsize=220)
        self.rowconfigure(2, weight=0, minsize=0)

        self.master.after(50, lambda: master.focus_force())
        self.master.after(100, self._reconstruct_image)

    def _create_interface(self):
        self.frm_controls = ttk.Frame(self, padding=(10, 15, 10, 10))
        self.frm_controls.grid(row=0, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.lbl_draw_mode = ttk.Label(self.frm_controls, text="Drawing Mode:")
        self.lbl_line_width = ttk.Label(self.frm_controls, text="Line Width:")
        self.lbl_refresh_rate = ttk.Label(self.frm_controls, text="Refresh (ms):")
        self.var_draw_mode = tk.IntVar(value=1)
        self.rad_draw = ttk.Radiobutton(self.frm_controls, text="Draw", variable=self.var_draw_mode, value=1)
        self.rad_erase = ttk.Radiobutton(self.frm_controls, text="Erase", variable=self.var_draw_mode, value=0)
        self.btn_clear = ttk.Button(
            self.frm_controls, text="Clear Image",
            command=lambda: self.cnv_orig.clear_image()
        )
        self.var_width = tk.StringVar(self.frm_controls)
        self.spn_width = tk.Spinbox(
            self.frm_controls, values=(1, 2, 3, 4, 5), width=10,
            state="readonly", textvariable=self.var_width
        )
        self.var_rate = tk.StringVar(self.frm_controls)
        self.spn_rate = tk.Spinbox(
            self.frm_controls, values=(10, 20, 50, 100, 200, 500, 1000), width=10,
            state="readonly", textvariable=self.var_rate
        )
        self.var_bbox = tk.IntVar(value=1)
        self.cbx_bbox = ttk.Checkbutton(self.frm_controls, text="Bounding Boxes", variable=self.var_bbox)

        self.lbl_draw_mode.grid(row=0, column=0, columnspan=2, sticky=(tk.N, tk.W))
        self.lbl_line_width.grid(row=0, column=3, sticky=(tk.N, tk.W))
        self.lbl_refresh_rate.grid(row=0, column=4, sticky=(tk.N, tk.W))
        self.rad_draw.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.rad_erase.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 20))
        self.btn_clear.grid(row=1, column=2, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 20))
        self.spn_width.grid(row=1, column=3, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 20))
        self.spn_rate.grid(row=1, column=4, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 20))
        self.cbx_bbox.grid(row=1, column=5, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.var_draw_mode.trace("w", lambda *_: self._set_draw_mode(self.var_draw_mode.get() == 1))
        self.var_width.trace("w", lambda *_: self.cnv_orig.set_line_width(int(self.var_width.get())))
        self.var_rate.trace("w", lambda *_: self._set_refresh_period(int(self.var_rate.get())))
        self.var_bbox.trace("w", lambda *_: self._set_bbox_visibility(self.var_bbox.get() == 1))

        self.frm_canvas_orig = ttk.Frame(self, padding=(10, 10, 5, 10))
        self.frm_canvas_orig.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_canvas_orig.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas_orig.rowconfigure(0, weight=0, minsize=20)
        self.frm_canvas_orig.rowconfigure(1, weight=1, minsize=200)

        self.lbl_orig = ttk.Label(self.frm_canvas_orig, text="Original Image (draw here):")
        self.cnv_orig = PixelCanvas(
            self.frm_canvas_orig, self.canvas_size, self.canvas_size, drawable=True,
            highlightthickness=0, borderwidth=0, width=400, height=400
        )
        self.lbl_orig.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_orig.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.frm_canvas_rec = ttk.Frame(self, padding=(5, 10, 5, 10))
        self.frm_canvas_rec.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_canvas_rec.columnconfigure(0, weight=1, minsize=200)
        self.frm_canvas_rec.rowconfigure(0, weight=0, minsize=20)
        self.frm_canvas_rec.rowconfigure(1, weight=1, minsize=200)

        self.lbl_rec = ttk.Label(self.frm_canvas_rec, text="Reconstructed Image:")
        self.cnv_rec = PixelCanvas(
            self.frm_canvas_rec, self.canvas_size, self.canvas_size, drawable=False,
            highlightthickness=0, borderwidth=0, width=400, height=400
        )
        self.lbl_rec.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.cnv_rec.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.frm_windows = ttk.Frame(self, padding=(0, 0, 0, 0))
        self.frm_windows.grid(row=1, column=2, sticky=(tk.N, tk.S, tk.W, tk.E))
        self.frm_windows.columnconfigure(0, weight=1)

        self.frm_canvas_win, self.lbl_win, self.cnv_win = [], [], []

        for i in range(3):
            self.frm_windows.rowconfigure(i, weight=1)

            frm_canvas_win = ttk.Frame(
                self.frm_windows,
                padding=(5, 10 if i == 0 else 0, 10, 10 if i == 2 else 0)
            )
            frm_canvas_win.grid(row=i, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
            frm_canvas_win.columnconfigure(0, weight=1, minsize=50)
            frm_canvas_win.rowconfigure(0, weight=0, minsize=20)
            frm_canvas_win.rowconfigure(1, weight=1, minsize=50)

            lbl_win = ttk.Label(
                frm_canvas_win, text="VAE Rec. #{0}:".format(i+1)
            )
            cnv_win = PixelCanvas(
                frm_canvas_win, self.window_size, self.window_size, drawable=False,
                highlightthickness=0, borderwidth=0, width=120, height=120
            )
            lbl_win.grid(row=0, column=0, sticky=(tk.S, tk.W))
            cnv_win.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

            self.frm_canvas_win.append(frm_canvas_win)
            self.lbl_win.append(lbl_win)
            self.cnv_win.append(cnv_win)

        self.lbl_status = ttk.Label(self, borderwidth=1, relief="sunken", padding=(5, 2))
        self.lbl_status.grid(row=2, column=0, columnspan=3, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.cnv_orig.bind("<Button-2>", lambda *_: self.cnv_orig.clear_image())
        self.cnv_orig.bind("<Button-3>", lambda *_: self.cnv_orig.clear_image())

        self.var_draw_mode.set(1)
        self.var_width.set("3")
        self.var_rate.set("50")
        self.var_bbox.set(1)

    def _reconstruct_image(self):
        dig, pos, rec, win, lat, loss = self.model_wrapper.infer(
            [self.cnv_orig.get_image()]
        )

        self.cnv_rec.set_image(rec[0])
        self.cnv_rec.set_bbox_positions(pos[0])
        self.cnv_orig.set_bbox_positions(pos[0])

        for i in range(len(self.cnv_win)):
            if i < len(win[0]):
                self.cnv_win[i].set_image(win[0][i])
                self.cnv_win[i].set_bbox_positions(
                    [[0.0, -2.0, -2.0]] * i + [[0.99, 0.0, 0.0]]
                )
            else:
                self.cnv_win[i].clear_image()
                self.cnv_win[i].set_bbox_positions([])

        self.lbl_status.configure(
            text="Reconstruction loss (negative log-likelihood): {0:.3f}".format(
                abs(loss[0])
            )
        )

        self.master.after(self.refresh_period, self._reconstruct_image)

    def _set_refresh_period(self, value):
        self.refresh_period = value

    def _set_bbox_visibility(self, visible):
        self.cnv_orig.set_bbox_visibility(visible)
        self.cnv_rec.set_bbox_visibility(visible)

    def _set_draw_mode(self, draw):
        self.cnv_orig.set_erasing_mode(not draw)
        self.cnv_orig.config(cursor=("cross" if draw else "icon"))

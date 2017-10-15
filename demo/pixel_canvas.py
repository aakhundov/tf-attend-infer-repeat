import math
import threading

import numpy as np
import tkinter as tk


class PixelCanvas(tk.Canvas):
    def __init__(self, master, w, h, image=None,
                 drawable=True, line_width=3, **kw):

        tk.Canvas.__init__(self, master=master, **kw)

        self.master = master
        self.w, self.h = w, h

        if image is not None:
            self.image = image.copy()
        else:
            self.image = np.zeros((w, h), dtype=np.float32)

        self.last_drawn_image = None
        self.redraw_lock = threading.Lock()
        self.photo = tk.PhotoImage(width=0, height=0, format='PPM')
        self.photo_id = self.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.bind("<Configure>", lambda e: self._redraw_canvas(resize=True))

        if drawable:
            self.erasing = False
            self.line_width = line_width
            self.bind("<Button-1>", self._left_click_event)
            self.bind("<B1-Motion>", self._mouse_drag_event)

        self.bbox_ids = self._create_bboxes()
        self.bbox_visibility = True
        self.bbox_positions = []

    def _redraw_canvas(self, resize=False):
        with self.redraw_lock:
            if not resize and \
               self.last_drawn_image is not None and \
               np.all(self.last_drawn_image == self.image):
                    return
            self.last_drawn_image = self.image.copy()

        self.cw = self.winfo_width()
        self.ch = self.winfo_height()
        self.rw = self.cw / self.w
        self.rh = self.ch / self.h

        arr = np.kron(
            self.image,
            np.ones((
                int(math.ceil(self.rh)),
                int(math.ceil(self.rw))
            ))
        )

        if arr.shape != (self.ch, self.cw):
            if arr.shape[0] != self.ch:
                diff, ceil = arr.shape[0] - self.ch, int(math.ceil(self.rh))
                idx = [i for i in range(arr.shape[0]) if i // ceil >= diff or i % ceil != 0]
                arr = arr[idx, :]
            if arr.shape[1] != self.cw:
                diff, ceil = arr.shape[1] - self.cw, int(math.ceil(self.rw))
                idx = [i for i in range(arr.shape[1]) if i // ceil >= diff or i % ceil != 0]
                arr = arr[:, idx]

        data = ("P5 {0} {1} 255 ".format(self.cw, self.ch)).encode()
        data += np.ndarray.astype(arr * 255, dtype=np.uint8).tobytes()
        self.photo.configure(width=self.cw, height=self.ch, data=data)

        if resize:
            self._redraw_bboxes()

    def _get_image_coordinates(self, cx, cy):
        ceil_w = int(math.ceil(self.rw))
        if self.cw % ceil_w == 0:
            j = int(math.floor(cx / self.rw))
        else:
            diff = self.w * ceil_w - self.cw
            if cx < diff * (ceil_w - 1):
                j = int(math.floor(cx / (ceil_w - 1)))
            else:
                j = diff + int(math.floor((cx - diff * (ceil_w - 1)) / ceil_w))

        ceil_h = int(math.ceil(self.rh))
        if self.ch % ceil_h == 0:
            i = int(math.floor(cy / self.rh))
        else:
            diff = self.h * ceil_h - self.ch
            if cy < diff * (ceil_h - 1):
                i = int(math.floor(cy / (ceil_h - 1)))
            else:
                i = diff + int(math.floor((cy - diff * (ceil_h - 1)) / ceil_h))

        return i, j

    def _coordinates_are_in_image(self, i, j):
        return self.w > i >= 0 and self.h > j >= 0

    def _draw_thin_line(self, cx1, cy1, cx2, cy2, length):
        i1, j1 = self._get_image_coordinates(cx1, cy1)
        i2, j2 = self._get_image_coordinates(cx2, cy2)

        if not self._coordinates_are_in_image(i1, j1) and \
           not self._coordinates_are_in_image(i2, j2):
            return

        coordinates = set([])
        steps = int(math.ceil(length / (min(self.rw, self.rh) / 2)))
        dx, dy = (cx2 - cx1) / steps, (cy2 - cy1) / steps

        x, y = cx1, cy1
        for i in range(steps + 1):
            coordinates.add(
                self._get_image_coordinates(x, y)
            )
            x, y = x + dx, y + dy

        updated = False
        for c in coordinates:
            if self._coordinates_are_in_image(*c):
                if self.erasing:
                    if self.image[c] != 0.0:
                        self.image[c] = 0.0
                        updated = True
                else:
                    if self.image[c] != 1.0:
                        self.image[c] = 1.0
                        updated = True

        return updated

    def _draw_line(self, cx1, cy1, cx2, cy2):
        length = math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        line_width = self.line_width if not self.erasing else 3

        if length < 1.0:
            return

        if line_width == 1:
            if self._draw_thin_line(cx1, cy1, cx2, cy2, length):
                self._redraw_canvas()
        else:
            slope = ((cx2 - cx1) / length, (cy2 - cy1) / length)
            increment = slope[1] * self.rw, -slope[0] * self.rh
            running = cx1 - increment[0] * (line_width - 1) / 2, cy1 - increment[1] * (line_width - 1) / 2, \
                cx2 - increment[0] * (line_width - 1) / 2, cy2 - increment[1] * (line_width - 1) / 2

            updated = False
            for i in range(line_width):
                updated = self._draw_thin_line(*running, length) or updated
                running = running[0] + increment[0], running[1] + increment[1], \
                    running[2] + increment[0], running[3] + increment[1]

            if updated:
                self._redraw_canvas()

    def _create_bboxes(self, num=10):
        bbox_ids = []
        colors = ["#F00", "#0F0", "#00F",
                  "#0FF", "#F0F", "#FF0", "#FFF"]

        for i in range(num):
            c = colors[i % len(colors)]
            bbox_ids.append(
                self.create_rectangle(
                    (0, 0, 0, 0), width=2, outline=c, fill=None,
                    tags=("bbox", "bbox_{0}".format(i))
                )
            )

        return bbox_ids

    def _redraw_bboxes(self):
        for i in range(len(self.bbox_ids)):
            if self.bbox_visibility and len(self.bbox_positions) > i:
                self.coords(self.bbox_ids[i], self._get_bbox_coordinates(self.bbox_positions[i]))
                self.itemconfig(self.bbox_ids[i], state="normal")
            else:
                self.itemconfig(self.bbox_ids[i], state="hidden")

    def _get_bbox_coordinates(self, position):
        scale, shift_x, shift_y = position

        sx, sy = scale * self.cw / 2.0, scale * self.ch / 2.0
        cx, cy = (1.0 + shift_x) * self.cw / 2.0, (1.0 + shift_y) * self.ch / 2.0
        lx, ly, rx, ry = cx - sx, cy - sy, cx + sx, cy + sy

        return lx, ly, rx, ry

    def _left_click_event(self, e):
        self.last_x, self.last_y = e.x, e.y

    def _mouse_drag_event(self, e):
        self._draw_line(self.last_x, self.last_y, e.x, e.y)
        self.last_x, self.last_y = e.x, e.y

    def get_image(self):
        return self.image.copy()

    def set_image(self, image):
        self.image = image.copy()
        self._redraw_canvas()

    def clear_image(self):
        self.image.fill(0.0)
        self._redraw_canvas()

    def set_erasing_mode(self, erasing=True):
        self.erasing = erasing

    def set_bbox_positions(self, positions):
        self.bbox_positions = positions[:]
        self._redraw_bboxes()

    def set_bbox_visibility(self, visible=True):
        self.bbox_visibility = visible
        self._redraw_bboxes()

    def set_line_width(self, width):
        self.line_width = width

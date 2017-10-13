import tkinter as tk
import tkinter.ttk as ttk

from demo.demo_window import DemoWindow


root = tk.Tk()

root.title("AIR Demo")
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

ttk.Style().theme_use("default")
ttk.Style().map("TCombobox", fieldbackground=[("readonly", "#ffffff")])

window = DemoWindow(root)
window.grid(sticky=(tk.N, tk.S, tk.W, tk.E))

root.mainloop()

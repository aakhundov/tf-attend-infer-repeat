import tkinter as tk
import tensorflow as tf

from air.air_model import AIRModel
from demo.demo_window import DemoWindow
from demo.model_wrapper import ModelWrapper


CANVAS_SIZE = 50
WINDOW_SIZE = 28

MODEL_PATH = "./model/air-model"


print("Creating placeholders...")
test_data = tf.placeholder(tf.float32, shape=[None, CANVAS_SIZE ** 2])
test_targets = tf.placeholder(tf.int32, shape=[None])

print("Creating model...")
air_model = AIRModel(
    test_data, test_targets,
    max_steps=3, rnn_units=256, canvas_size=CANVAS_SIZE, windows_size=WINDOW_SIZE,
    vae_latent_dimensions=50, vae_recognition_units=(512, 256), vae_generative_units=(256, 512),
    vae_likelihood_std=0.3, scale_hidden_units=64, shift_hidden_units=64, z_pres_hidden_units=64,
    z_pres_temperature=1.0, stopping_threshold=0.99, cnn=False,
    train=False, reuse=False, scope="air",
)

with tf.Session() as sess:
    print("Restoring model...")
    tf.train.Saver().restore(sess, MODEL_PATH)
    wrapper = ModelWrapper(air_model, sess, test_data)

    print("Creating window...")
    master = tk.Tk()
    master.title("Attend Infer Repeat - Live Demo")
    master.columnconfigure(0, weight=1)
    master.rowconfigure(0, weight=1)
    window = DemoWindow(master, wrapper, CANVAS_SIZE)
    window.grid(sticky=(tk.N, tk.S, tk.W, tk.E))
    master.mainloop()

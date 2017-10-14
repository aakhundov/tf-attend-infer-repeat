import numpy as np


class ModelWrapper:

    def __init__(self, model, session, data_placeholder, canvas_size=50, window_size=28):

        self.model = model
        self.session = session
        self.data_placeholder = data_placeholder
        self.canvas_size = canvas_size
        self.window_size = window_size

    def infer(self, images):
        all_digits, all_positions = [], []
        all_windows, all_latents = [], []
        all_reconstructions, all_loss = [], []

        rec_digits, rec_scales, rec_shifts, reconstructions, \
            rec_windows, rec_latents, rec_loss = self.session.run(
                [
                    self.model.rec_num_digits, self.model.rec_scales,
                    self.model.rec_shifts, self.model.reconstruction,
                    self.model.rec_windows, self.model.rec_latents,
                    self.model.reconstruction_loss
                ],
                feed_dict={
                    self.data_placeholder: [np.ravel(img) for img in images]
                }
            )

        for i in range(len(rec_digits)):
            digits = int(rec_digits[i])
            reconstruction = np.reshape(
                reconstructions[i], (self.canvas_size, self.canvas_size)
            )

            positions = []
            windows, latents = [], []
            for j in range(digits):
                positions.append(np.array([rec_scales[i][j][0]] + list(rec_shifts[i][j])))
                windows.append(np.reshape(rec_windows[i][j], (self.window_size, self.window_size)))
                latents.append(rec_latents[i][j])

            all_digits.append(digits)
            all_positions.append(np.array(positions))
            all_reconstructions.append(reconstruction)
            all_windows.append(np.array(windows))
            all_latents.append(np.array(latents))
            all_loss.append(rec_loss[i])

        return all_digits, all_positions, all_reconstructions, all_windows, all_latents, all_loss

import numpy as np


class OneEuroPoseFilter:
    """Adaptive temporal filter for COCO keypoints 0..8."""

    def __init__(
        self,
        fps=25.0,
        min_cutoff=1.0,
        beta=0.08,
        derivative_cutoff=1.0,
        confidence_threshold=0.25,
    ):
        if fps <= 0 or min_cutoff <= 0 or derivative_cutoff <= 0:
            raise ValueError("FPS and cutoffs must be positive")
        if beta < 0 or not 0 <= confidence_threshold <= 1:
            raise ValueError("Invalid filter parameters")

        self.dt = 1.0 / fps
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self.confidence_threshold = confidence_threshold

        self.positions = np.zeros((9, 2), dtype=np.float64)
        self.derivatives = np.zeros((9, 2), dtype=np.float64)
        self.initialized = np.zeros(9, dtype=bool)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def update(self, pose, dt=None):
        pose = np.asarray(pose, dtype=np.float32)

        if pose.shape != (9, 3):
            raise ValueError("Pose must have shape (9, 3)")
        if not np.isfinite(pose).all():
            raise ValueError("Pose must contain finite values")

        if dt is None:
            dt = self.dt
        else:
            dt = float(dt)
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError('dt must be finite and positive')

        output = pose.copy()
        derivative_alpha = self._alpha(
            self.derivative_cutoff, dt
        )

        for i in range(9):
            confidence = float(pose[i, 2])
            measurement = pose[i, :2].astype(np.float64)

            if confidence < self.confidence_threshold:
                if self.initialized[i]:
                    output[i, :2] = self.positions[i]
                    self.derivatives[i] *= 0.5
                continue

            if not self.initialized[i]:
                self.positions[i] = measurement
                self.derivatives[i] = 0.0
                self.initialized[i] = True
                continue

            raw_derivative = (
                measurement - self.positions[i]
            ) / dt

            self.derivatives[i] = (
                derivative_alpha * raw_derivative
                + (1.0 - derivative_alpha) * self.derivatives[i]
            )

            speed = float(np.linalg.norm(self.derivatives[i]))

            cutoff = self.min_cutoff + self.beta * speed
            alpha = self._alpha(cutoff, dt)

            self.positions[i] = (
                alpha * measurement
                + (1.0 - alpha) * self.positions[i]
            )

            output[i, :2] = self.positions[i]

        return output

    def reset(self):
        self.positions.fill(0)
        self.derivatives.fill(0)
        self.initialized.fill(False)

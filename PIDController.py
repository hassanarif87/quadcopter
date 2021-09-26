
class PIDController:
    def __init__(self, dt, p_gain, i_gain = 0, d_gain = 0, sat=None):
        self.dt = dt
        self.p_gain = p_gain
        self.i_gain = i_gain
        self.d_gain = d_gain
        self.sat = sat
        self.integrated_error=0
        self.prev_error=0
    def update(self, sp, fb):
        # PID
        dt = self.dt
        error = sp - fb
        self.integrated_error = self.integrated_error + error*dt

        p_term = self.p_gain * (error)
        i_term =self.p_gain * (self.integrated_error)

        d_term = self.d_gain * (error - self.prev_error) / dt

        output = p_term + i_term + d_term
        self.prev_error = error
        if (self.sat is not None):
            # set Saturation
            output = min(output, self.sat)
            output = max(output, -self.sat)
        return output

    @property
    def unwind_integral(self):
        self.integrated_error=0



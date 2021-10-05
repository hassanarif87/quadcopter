
import numpy as np
# Ref:
# https://www.semanticscholar.org/paper/An-introduction-to-inertial-navigation-Woodman/dd785cbf1cea96eade72fff03eb3d30562feca5b
# https://github.com/Aceinna/gnss-ins-sim/tree/master/gnss_ins_sim
class Gyroscope:
    def __init__(self,dt, arw=0) -> None:
        self.bias_instability_noise_mu = arw # 0.1
        self.constant_bias = np.zeros(3)
        self.noise_mu = 0 #0.1
        self.dt = dt
    def update(self, state):
        omega_body = np.dot(state.dcm_body2frame.transpose(), state.omega)
        sensor_noise = np.random.normal(0,  self.mu, omega_body.shape)
        bias_instability = np.random.normal(0,  self.bias_instability_noise_mu, omega_body.shape)
        vairing_bias = bias_instability * self.dt
        sensor_bias = vairing_bias + self.constant_bias
        omega_sensor = omega_body + sensor_bias + sensor_noise
        return omega_sensor


class Accelometer:
    def __init__(self,vrw=0) -> None:
        self.noise_mu = vrw
        self.sensor_bias = np.zeros(3)

    def update(self, state, accel):
        dcm_frame2body = state.dcm_body2frame.transpose()
        accel_body = np.dot(dcm_frame2body, accel)
        sensor_noise = np.random.normal(0,  self.mu, accel_body.shape)
        a_sensor =  accel_body + sensor_noise + self.sensor_bias
        return a_sensor

class Magnetometer:
    def __init__(self) -> None:
        pass
    def update(state):
        pass

class IMU:
    def __init__(self,dt):
        self.gyro = Gyroscope(dt)
        self.accel = Accelometer(dt)
        self.mag = Magnetometer(dt)

    def update(self,state, accel):
        self.data_out = dict(
            gyro=self.gyro.update(state),
            accel=self.accel.update(state, accel),
            mag=self.mag.update(state)
        )
    def get_data(self):
        return self.data_out

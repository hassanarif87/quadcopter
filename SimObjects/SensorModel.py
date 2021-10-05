
import numpy as np
# Ref:
# https://www.semanticscholar.org/paper/An-introduction-to-inertial-navigation-Woodman/dd785cbf1cea96eade72fff03eb3d30562feca5b
# https://github.com/Aceinna/gnss-ins-sim/tree/master/gnss_ins_sim\

# gyro_low_accuracy = {'b': np.array([0.0, 0.0, 0.0]) * D2R,
#                      'b_drift': np.array([10.0, 10.0, 10.0]) * D2R/3600.0,
#                      'b_corr':np.array([100.0, 100.0, 100.0]),
#                      'arw': np.array([0.75, 0.75, 0.75]) * D2R/60.0}
# accel_low_accuracy = {'b': np.array([0.0e-3, 0.0e-3, 0.0e-3]),
#                       'b_drift': np.array([2.0e-4, 2.0e-4, 2.0e-4]),
#                       'b_corr': np.array([100.0, 100.0, 100.0]),
#                       'vrw': np.array([0.05, 0.05, 0.05]) / 60.0}
# mag_low_accuracy = {'si': np.eye(3) + np.random.randn(3, 3)*0.0,
#                     'hi': np.array([10.0, 10.0, 10.0])*0.0,
#                     'std': np.array([0.1, 0.1, 0.1])}
class Gyroscope:
    def __init__(self,dt, arw=0) -> None:
        self.bias_instability_noise_mu = np.deg2rad(10.0)/3600.0 #arw
        self.constant_bias = np.zeros(3)
        self.noise_mu =  0.05 / 60.0 #0.1
        self.dt = dt
        self.vairing_bias = np.zeros(3)
    def update(self, state):
        omega_body = np.dot(state.dcm_body2frame.transpose(), state.omega)
        sensor_noise = np.random.normal(0,  self.noise_mu, omega_body.shape)
        bias_instability = np.random.normal(0,  self.bias_instability_noise_mu, omega_body.shape)
        self.vairing_bias += bias_instability * self.dt
        sensor_bias = self.vairing_bias + self.constant_bias
        omega_sensor = omega_body + sensor_bias + sensor_noise
        return omega_sensor


class Accelometer:
    def __init__(self,vrw=0) -> None:
        self.noise_mu = 0.05 / 60.0
        self.sensor_bias = np.zeros(3)

    def update(self, state, accel):
        dcm_frame2body = state.dcm_body2frame.transpose()
        accel_body = np.dot(dcm_frame2body, accel)
        sensor_noise = np.random.normal(0,  self.noise_mu, accel_body.shape)
        a_sensor =  accel_body + sensor_noise + self.sensor_bias
        return a_sensor

class Magnetometer:
    def __init__(self) -> None:
        self.si = np.eye(3) + np.random.randn(3, 3)*0.0 # is the soft iron matrix
        self.hi = np.array([10.0, 10.0, 10.0])*0.0 # the hard iron consta bias
        self.std =  np.array([0.1, 0.1, 0.1]) # white noise

    def update(self,state):
        dcm_frame2body = state.dcm_body2frame.transpose()
        mag_truth = np.dot(dcm_frame2body, np.array([1.,0.,0.]))
        mag_sensor = np.dot(self.si, mag_truth ) + np.random.normal(0,  self.std[0], mag_truth.shape) + self.hi
        return mag_sensor
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

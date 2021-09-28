import unittest
import numpy as np
from Controller import PIDController, quar_axis_error, thrust_tilt
from transforms3d import quaternions, euler
euler = euler.EulerFuncs('rxyz')

class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.controller = PIDController(1,10, sat= 50, name = "test")

    def test_basic_output(self):
        setpoint = 0
        feedback = -3
        output = self.controller.update(setpoint, feedback)
        self.assertEqual(self.controller.name, 'test')
        self.assertEqual(output, 3*10)


    def test_saturation(self):
        setpoint = 0
        feedback = 8
        output = self.controller.update(setpoint, feedback)
        self.assertEqual(output, -50)

        setpoint = 8
        feedback = 0
        output = self.controller.update(setpoint, feedback)
        self.assertEqual(output, 50)

    def test_pd(self):
        setpoint = 0
        feedback = 1
        controller = PIDController(1, p_gain = 10, i_gain = 0, d_gain = 2, sat = 50, name = "test")
        output = controller.update(setpoint, feedback)
        self.assertEqual(output, -12.0)
        output = controller.update(setpoint, feedback)
        self.assertEqual(output, -10.0)

    def test_pi(self):
        setpoint = 1
        feedback = 0
        controller = PIDController(1,p_gain = 10, i_gain = 4, d_gain = 0, sat= 50, name = "test")
        for ii in range(0,3):
            output = controller.update(setpoint, feedback)
            self.assertEqual(output, 10 +4/2 + 4*ii)

        controller.reset_integral
        self.assertEqual(controller.integrated_error, 0.0)


class Test_axiserror(unittest.TestCase):

    def test_angles(self):
        q_sp = quaternions.qeye()
        q_state = quaternions.qeye()
        result = quar_axis_error(q_sp, q_state)
        self.assertEqual(result[0], 0.)
        self.assertEqual(result[1], 0.)
        self.assertEqual(result[2], 0.)

        # Roll
        q_sp = euler.euler2quat(0.1,0.0,0)
        result = quar_axis_error(q_sp, q_state)
        self.assertAlmostEqual(result[0], 0.1, delta=1.e-8)
        self.assertAlmostEqual(result[1], 0.0, delta= 1.e-8)
        self.assertAlmostEqual(result[2], 0., delta=1.e-8)

        # Pitch
        q_sp = euler.euler2quat(0,0.1,0)
        result = quar_axis_error(q_sp, q_state)
        self.assertAlmostEqual(result[0], 0.0, delta = 1.e-8)
        self.assertAlmostEqual(result[1], 0.1, delta = 1.e-8)
        self.assertAlmostEqual(result[2], 0.0, delta = 1.e-8)

        # Yaw
        q_sp = euler.euler2quat(0,0,0.1)
        result = quar_axis_error(q_sp, q_state)
        self.assertAlmostEqual(result[0], 0.0, delta = 1.e-8)
        self.assertAlmostEqual(result[1], 0.0, delta = 1.e-8)
        self.assertAlmostEqual(result[2], 0.1, delta = 1.e-8)

    def test_composite_angles(self):
        pass

if __name__ == '__main__':
    unittest.main()
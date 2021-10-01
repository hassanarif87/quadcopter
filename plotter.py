
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from Helper import rot_matrix3d

import numpy as np

def plot_logs(time, logger, plot, plot_traj):
    tend = time[-1]
    f = logger.f
    X = logger.log['X']
    eulAng = logger.log['eulAng']
    vel = logger.log['Xdot']
    angvel = logger.log['omega']
    u = logger.log['u']
    PWM = logger.log['PWM']
    velDot = logger.log['velDot']
    eulAngSP = logger.log['eulAngSP']
    rateSP = logger.log['rateSP']
    aero_force = logger.log['aero_force']
    if (plot == True):
        plt.figure(1)
        plt.plot(time, X[0,:],'r',label='X',linewidth= 0.5)
        plt.plot(time, X[1,:],'g',label='Y',linewidth= 0.5)
        plt.plot(time, X[2,:],'b',label='Z',linewidth= 0.5)
        plt.legend()
        plt.title('Pos')

        plt.figure(2)
        plt.plot(time, vel[0,:],'r',label='X',linewidth= 0.5)
        plt.plot(time, vel[1,:],'g',label='Y',linewidth= 0.5)
        plt.plot(time, vel[2,:],'b',label='Z',linewidth= 0.5)
        plt.legend()
        plt.ylabel("Velocity(m/s)")
        plt.xlabel("Time(s)")
        plt.title('Vel')

        plt.figure(3)
        plt.plot(time, eulAng[0,:],'r',label='Roll',linewidth= 0.5)
        plt.plot(time, eulAng[1,:],'g',label='Pitch',linewidth= 0.5)
        plt.plot(time, eulAng[2,:],'b',label='Yaw',linewidth= 0.5)
        plt.plot(time, eulAngSP[0,:],'r--',label='Roll SP',linewidth= 0.5)
        plt.plot(time, eulAngSP[1,:],'g--',label='Pitch SP',linewidth= 0.5)
        plt.plot(time, eulAngSP[2,:],'b--',label='Yaw SP',linewidth= 0.5)
        plt.ylabel("Angle(rad)")
        plt.xlabel("Time(s)")
        plt.legend(loc='upper right')
        plt.title('Euler Angles')

        plt.figure(4)
        plt.plot(time, u[0,:],label='Z',linewidth= 0.5)
        plt.plot(time, u[1,:],label='R',linewidth= 0.5)
        plt.plot(time, u[2,:],label='P',linewidth= 0.5)
        plt.plot(time, u[3,:],label='Y',linewidth= 0.5)
        plt.legend()
        plt.title('Cmd')

        plt.figure(5)
        plt.plot(time, PWM[0,:])
        plt.plot(time, PWM[1,:])
        plt.plot(time, PWM[2,:])
        plt.plot(time, PWM[3,:])
        plt.title('PWM')

        plt.figure(6)
        plt.plot(time, velDot[0,:])
        plt.plot(time, velDot[1,:])
        plt.plot(time, velDot[2,:])
        plt.title('Linear Accel')

        plt.figure(7)
        plt.plot(time, rateSP[0,:],label='Roll',linewidth= 0.5)
        plt.plot(time, rateSP[1,:],label='Pitch',linewidth= 0.5)
        plt.plot(time, rateSP[2,:],label='Yaw',linewidth= 0.5)
        plt.legend()
        plt.title('Rate Sp')

        plt.figure(8)
        plt.plot(time, aero_force[0,:],'r',label='X',linewidth= 0.5)
        plt.plot(time, aero_force[1,:],'g',label='Y',linewidth= 0.5)
        plt.plot(time, aero_force[2,:],'b',label='Z',linewidth= 0.5)
        plt.legend()
        plt.ylabel("Force(N)")
        plt.xlabel("Time(s)")
        plt.title('Aero Forces')

    if (plot_traj == True):
        ## 3d path
        wing1o = np.array([0, 0, 0])
        wing1edge =  np.array([[.5],[-.5],[0.]])
        wing2o = np.array([0, 0, 0])
        wing2edge = np.array([[.5],[.5],[0.]])
        wing3o = np.array([0, 0, 0])
        wing3edge = np.array([[-.5],[.5],[0.]])
        wing4o = np.array([0, 0, 0])
        wing4edge = np.array([[-.5],[-.5],[0.]])

        def update_traj(num, dat, lines):
            for line in lines:
                # NOTE: there is no .set_data() for 3 dim data...
                line.set_data(dat[0:2, :num])
                line.set_3d_properties(dat[2, :num])
            return lines

        def update_wings(num, dat, lines):
            for line in lines:
                # NOTE: there is no .set_data() for 3 dim data...
                x = np.array(dat[0:2,0,num])
                y = np.array(dat[0:2,1,num])
                line.set_data(x, y)
                line.set_3d_properties(dat[0:2,2,num])
            return lines
        # Attaching 3D axis to the figure
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        data = X

        data = np.array(data)

        n_steps = (tend* f)
        plot_frames = (int)(n_steps/50)
        traj_data = np.zeros([3,plot_frames])
        wing_data1 = np.zeros([2,3,plot_frames])
        wing_data2 = np.zeros([2,3,plot_frames])
        i = 0
        for j in range(0,plot_frames):
            traj_data[:,j] = data[:,i]
            trag_date_n = np.reshape(traj_data[:,j],(3,1))
            eul = np.reshape(eulAng[:,i],(3,1))
            wing1 = trag_date_n + np.dot(rot_matrix3d(eul),wing1edge)
            wing2 = trag_date_n + np.dot(rot_matrix3d(eul),wing2edge)
            wing3 = trag_date_n + np.dot(rot_matrix3d(eul),wing3edge)
            wing4 = trag_date_n + np.dot(rot_matrix3d(eul),wing4edge)
            wing_data1[0,:, j] = wing1.reshape(3)
            wing_data1[1,:, j] = wing3.reshape(3)
            wing_data2[0,:, j] = wing2.reshape(3)
            wing_data2[1,:, j] = wing4.reshape(3)
            i = i + 50

        lines = ax.plot(traj_data[0, 0:1], traj_data[1, 0:1], traj_data[2, 0:1], 'b--')
        lines1 = ax.plot(wing_data1[0:2, 0, 0], wing_data1[0:2, 1, 0], wing_data1[0:2, 2, 0], 'r')
        lines2 = ax.plot(wing_data1[0:2, 0, 0], wing_data1[0:2, 1, 0], wing_data1[0:2, 2, 0], 'g')

        # Setting the axes properties
        ax.set_xlim3d([-1.0, 8.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-4.0, 4.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-4.0, 4.0])
        ax.set_zlabel('Z')

        ax.set_title('Trajectory')

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig, update_traj, plot_frames, fargs=(traj_data, lines),
                                        interval=1, blit=False)
        line_ani1 = animation.FuncAnimation(fig, update_wings, plot_frames, fargs=(wing_data1, lines1),
                                        interval=1, blit=False)
        line_ani2 = animation.FuncAnimation(fig, update_wings, plot_frames, fargs=(wing_data2, lines2),
                                        interval=1, blit=False)
    #########################

    plt.show()
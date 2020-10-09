import numpy as np
import matplotlib.pyplot as plt

steering_angle = 0.5
max_accel = 1.6


class TrajectorySets:

    def __init__(self, velocity=2, wheelbase=2.9):
        self.x = 0
        self.y = 0
        self.v = velocity
        self.v0 = velocity
        self.theta = 0
        self.u_steer = -steering_angle
        self.u_accel = 0.1
        self.b = wheelbase
        self.trajectory_sets_x = []
        self.trajectory_sets_y = []

    def state_equation(self, delta_t=0.01):
        self.v = self.v + self.u_accel * delta_t
        self.theta = self.theta + self.v / self.b * np.tan(self.u_steer) * delta_t
        self.x = self.x + self.v * np.cos(self.theta) * delta_t
        self.y = self.y + self.v * np.sin(self.theta) * delta_t
        # print(self.v)

    def generate_trajectory(self):
        trajectory_x = []
        trajectory_y = []
        for i in range(600):
            self.state_equation()
            trajectory_x.append(self.x)
            trajectory_y.append(self.y)
            # print(self.u_accel)
            # if i == 599:
                # print(self.v)
        self.trajectory_sets_x.append(trajectory_x)
        self.trajectory_sets_y.append(trajectory_y)
        self.initialization()

    def generate_trajectory_sets(self):
        delta_steer = 20
        delta_accel = 1
        for i in range(delta_accel):
            for j in range(delta_steer * 2):
                self.generate_trajectory()
                self.u_steer += steering_angle/delta_steer

            self.u_accel += max_accel/delta_accel
            self.u_steer = -steering_angle
            # print(self.u_accel)

    def initialization(self):
        self.x = 0
        self.y = 0
        self.v = self.v0
        self.theta = 0

    def print_trajectory(self):
        for y, x in zip(self.trajectory_sets_x, self.trajectory_sets_y):
            plt.plot(x, y)

        # plt.xlim(-3.5, 3.5)
        # plt.ylim(0, 14)
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title("pedestrian, v = {} m/s, wheelbase = {} m, steering angle = {} rad".format(self.v0, self.b, steering_angle))
        plt.show()


# def state_equation_test():
#     test = TrajectorySets(10)
# assert

if __name__ == '__main__':
    test1 = TrajectorySets(velocity=2, wheelbase=0.6)
    test1.generate_trajectory_sets()
    test1.print_trajectory()

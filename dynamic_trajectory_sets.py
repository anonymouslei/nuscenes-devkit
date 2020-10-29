# Code written by Lei Ge, 2020.
# KIT-MRT
import numpy as np
import matplotlib.pyplot as plt

# steering_angle = 0.5
max_accel = 1.6
delta_steer = 100
delta_accel = 1
category_v = "vehicle"
category_v = "pedestrian"

class TrajectorySets:

    def __init__(self, velocity=2, steering_angle=0.5, wheelbase=2.9, acceleration=0.1):
        self.x = 0
        self.y = 0
        self.v = velocity
        self.v0 = velocity
        self.theta = 0
        self.u_steer = -steering_angle
        self.steering_angle = steering_angle
        self.u_accel = acceleration
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
        self.trajectory_sets_x.append(trajectory_x)
        self.trajectory_sets_y.append(trajectory_y)
        self.initialization()

    def generate_trajectory_sets(self):
        # delta_steer = 100
        # delta_accel = 1

        # u_steer_sets = self.generate_normal_random(self.steering_angle/5, delta_steer*2)

        for i in range(delta_accel):
            u_steer_sets = self.generate_normal_random(self.steering_angle/5, delta_steer * 2)
            for j in range(delta_steer * 2):
                self.generate_trajectory()
                # self.u_steer += self.steering_angle/delta_steer
                self.u_steer = u_steer_sets[j]

            self.u_accel += max_accel/delta_accel
            # self.u_steer = -self.steering_angle
            # print(self.u_accel)

    def initialization(self):
        self.x = 0
        self.y = 0
        self.v = self.v0
        self.theta = 0

    def print_trajectory(self, category):
        i = 0
        for y, x in zip(self.trajectory_sets_x, self.trajectory_sets_y):
            if category == 'pedestrian' or category == "bicycle":
                plt.plot(x, y)
            elif i < delta_steer * 2:
                plt.plot(x, y, color="b")
            elif i < 2 * delta_steer * 2:
                plt.plot(x, y, color='g')
            elif i < 3 * delta_steer * 2:
                plt.plot(x, y, color='r')
            elif i < 4 * delta_steer * 2:
                plt.plot(x, y, color='c')
            elif i < 5 * delta_steer * 2:
                plt.plot(x, y, color='y')
            i += 1

        # plt.xlim(-3.5, 3.5)
        # plt.ylim(0, 14)
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title("{}, v = {} m/s, wheelbase = {} m, steering angle = {} rad".
                  format(category, self.v0, self.b, self.steering_angle))
        plt.show()

    def generate_normal_random(self, sigma, sampleNo):
        mu = 0
        if category_v == "vehicle":
            r = np.random.rand(1)
            np.random.seed(int(10 * r))
        else:
            np.random.seed(0)
        s = np.random.normal(mu, sigma, sampleNo)
        return s

# def state_equation_test():
#     test = TrajectorySets(10)
# assert

if __name__ == '__main__':
    pedestrian = TrajectorySets(velocity=1, wheelbase=0.6, acceleration=0.1, steering_angle=0.5)
    pedestrian.generate_trajectory_sets()
    pedestrian.print_trajectory("pedestrian")

    # vehicle = TrajectorySets(velocity=10, wheelbase=3, acceleration=0.5, steering_angle=0.2)
    # vehicle.generate_trajectory_sets()
    # vehicle.print_trajectory("vehicle")

    # bicycle = TrajectorySets(velocity=4, wheelbase=1.5, acceleration=1, steering_angle=0.4)
    # bicycle.generate_trajectory_sets()
    # bicycle.print_trajectory("bicycle")

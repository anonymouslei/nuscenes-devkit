import numpy as np
import matplotlib.pyplot as plt

# steering_angle = 0.5
max_accel = 1.6


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
        delta_steer = 20
        delta_accel = 1

        u_steer_sets = self.generate_normal_random(self.steering_angle/5, delta_steer*2)

        for i in range(delta_accel):
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
        for y, x in zip(self.trajectory_sets_x, self.trajectory_sets_y):
            plt.plot(x, y)

        # plt.xlim(-3.5, 3.5)
        # plt.ylim(0, 14)
        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title("{}, v = {} m/s, wheelbase = {} m, steering angle = {} rad".
                  format(category, self.v0, self.b, self.steering_angle))
        plt.show()

    def generate_normal_random(self, sigma, sampleNo):
        mu = 0
        np.random.seed(0)
        s = np.random.normal(mu, sigma, sampleNo)
        return s

# def state_equation_test():
#     test = TrajectorySets(10)
# assert

if __name__ == '__main__':
    pedestrian = TrajectorySets(velocity=1, wheelbase=0.6, acceleration=0.1)
    pedestrian.generate_trajectory_sets()
    pedestrian.print_trajectory("pedestrain")

    vehicle = TrajectorySets(velocity=1, wheelbase=0.6)
    vehicle.generate_trajectory_sets()
    vehicle.print_trajectory("vehicle")

    bicycle = TrajectorySets(velocity=1, wheelbase=0.6)
    bicycle.generate_trajectory_sets()
    bicycle.print_trajectory("bicycle")

# Code written by Lei Ge, 2020.
# KIT-MRT
import numpy as np
import matplotlib.pyplot as plt

# max_accel = 2
# num_of_steer_angle = 100
# delta_accel = 1


class TrajectorySets:
    """
    Generates trajectory sets with different steering angles and acceleration according to different category,
    e.g. vehicle, bicycle and pedestrian. Every trajectory has the constant steering angle.
    Using normal distribution to generate different steering angles.

    """

    def __init__(self, velocity=2, steering_angle=0.5, wheelbase=2.9, acceleration=0.1, category="vehicle",
                 num_of_steer_angle=100, num_of_accel=1, max_accel=2):
        """
        steering_angle: 68.2% of trajectories will be in [-steering_angle, steering_angle],
                        95.4% of trajectories will be in [-2*steering_angle, 2*steering_angle],
                        99.6% of trajectories will be in [-3*steering_angle, 3*steering_angle].
                        If there are too few trajectories, anomalies may occur. The above distribution is satisfied only
                        if there are plenty of trajectories.
        num_of_steer_angle: the number of trajectories for each acceleration
        num_of_accel: the number of different accelerations

        """

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
        self.category = category
        self.num_of_steer_angle = num_of_steer_angle
        self.num_of_accel = num_of_accel
        self.max_accel = max_accel

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
        for i in range(self.num_of_accel):
            u_steer_sets = self.generate_normal_random(self.steering_angle, self.num_of_steer_angle * 2)
            for j in range(self.num_of_steer_angle * 2):
                self.generate_trajectory()
                self.u_steer = u_steer_sets[j]

            self.u_accel += self.max_accel / self.num_of_accel

    def initialization(self):
        self.x = 0
        self.y = 0
        self.v = self.v0
        self.theta = 0

    def print_trajectory(self):
        i = 0
        for y, x in zip(self.trajectory_sets_x, self.trajectory_sets_y):
            if self.num_of_accel == 1:
                plt.plot(x, y)
            elif i < self.num_of_steer_angle * 2:
                plt.plot(x, y, color="b")
            elif i < 2 * self.num_of_steer_angle * 2:
                plt.plot(x, y, color='g')
            elif i < 3 * self.num_of_steer_angle * 2:
                plt.plot(x, y, color='r')
            elif i < 4 * self.num_of_steer_angle * 2:
                plt.plot(x, y, color='c')
            elif i < 5 * self.num_of_steer_angle * 2:
                plt.plot(x, y, color='y')
            i += 1

        plt.xlabel("x(m)")
        plt.ylabel("y(m)")
        plt.title("{}, v = {} m/s, wheelbase = {} m, sigma = {} rad".
                  format(self.category, self.v0, self.b, self.steering_angle))
        plt.show()

    def generate_normal_random(self, sigma, sampleNo):
        mu = 0
        if self.category == "vehicle":
            r = np.random.rand(1)
            np.random.seed(int(10 * r))
        else:
            np.random.seed(0)
        s = np.random.normal(mu, sigma, sampleNo)
        return s


if __name__ == '__main__':
    # recommended parameter for pedestrian
    # the speed of a pedestrian is generally 1.5m/s
    # steering_angle = 0.15
    pedestrian = TrajectorySets(velocity=1, wheelbase=0.6, acceleration=0.1, steering_angle=0.15, category="pedestrian")
    pedestrian.generate_trajectory_sets()
    pedestrian.print_trajectory()

    # recommended parameters for vehicle:
    # wheelbase = 3
    vehicle = TrajectorySets(velocity=10, wheelbase=3, acceleration=0.5, steering_angle=0.04, category="vehicle",
                             num_of_accel=5, num_of_steer_angle=50)
    vehicle.generate_trajectory_sets()
    vehicle.print_trajectory()

    # recommended parameter for bicycle
    # the speed of a bicycle is generally between 10-30km/h, that is between 2.8m/s - 8.3m/s
    # wheelbase = 1.5
    # steering_angle = 0.08
    bicycle = TrajectorySets(velocity=4, wheelbase=1.5, acceleration=1, steering_angle=0.08, category="bicycle")
    bicycle.generate_trajectory_sets()
    bicycle.print_trajectory()

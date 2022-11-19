# CSE4088 HW3
# Berk Kırtay
# 150118043

import numpy as np
import matplotlib.pyplot as plt


class Gradient_Descent():
    learning_rate = 0.1

    # To calculate error:
    def func(self, u, v):
        return (u*np.exp(v) - 2*v*np.exp(-u))**2

    # Calculating the derivatives of squared residuals:
    def apply_gradient_descent(self, u, v):
        u_err = np.float64(2*(u*np.exp(v) - 2*v*np.exp(-u))
                            * (np.exp(v) + 2*v*np.exp(-u)))
        v_err = np.float64(2*(u*np.exp(v) - 2*v*np.exp(-u))
                            * (u*np.exp(v) - 2*np.exp(-u)))
        return u_err * self.learning_rate, v_err * self.learning_rate

    def gradient_descent_algorithm(self, is_plot=False):
        err_rate = 1
        latest_u = 1
        latest_v = 1
        u_arr = [latest_u]
        v_arr = [latest_v]
        for i in range(100):
            u_err, v_err = self.apply_gradient_descent(
                latest_u, latest_v)
            latest_u = latest_u - u_err
            latest_v = latest_v - v_err

            # Putting every (u,v) pair in a list to plot them later.
            u_arr.append(latest_u)
            v_arr.append(latest_v)

            err_rate = self.func(latest_u, latest_v)
            iterations = i + 1
            if err_rate <= 10 ** -14:
                break

        print("Gradient Descent:")
        print(f"Number of iterations: {iterations}")
        print(f"Final(u,v): ({latest_u}, {latest_v})")
        print(f"Error rate after completion: {err_rate}")

        if is_plot == True:
            self.plot(u_arr, v_arr, iterations,
                      "Gradient Descent: Convergence of (u,v)")

    def coordinate_descent(self, is_plot=False):
        err_rate = 1
        latest_u = 1
        latest_v = 1
        u_arr = [latest_u]
        v_arr = [latest_v]
        for i in range(15):
            u_err, _ = self.apply_gradient_descent(
                latest_u, latest_v)
            latest_u = latest_u - u_err
            u_arr.append(latest_u)
            v_arr.append(latest_v)

            _, v_err = self.apply_gradient_descent(
                latest_u, latest_v)
            latest_v = latest_v - v_err
            u_arr.append(latest_u)
            v_arr.append(latest_v)

            err_rate = self.func(latest_u, latest_v)
            iterations = i + 1
            if err_rate <= 10 ** -14:
                break

        print("Coordinate Descent:")
        print(f"Number of iterations: {iterations}")
        print(f"Final(u,v): ({latest_u}, {latest_v})")
        print(f"Error rate after completion: {err_rate}")

        if is_plot == True:
            self.plot(u_arr, v_arr, iterations,
                      "Coordinate Descent: Convergence of (u,v)")

    def plot(self, u_arr, v_arr, iterations, title):
        plt.plot(u_arr, v_arr, '.-')
        plt.ylabel('u')
        plt.xlabel('v')
        plt.title(f'{title} for N={iterations} iterations')
        plt.legend()
        plt.show()


gd = Gradient_Descent()
gd.gradient_descent_algorithm()
gd.coordinate_descent()

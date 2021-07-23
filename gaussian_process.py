import numpy as np
import matplotlib.pyplot as plt


class Visualizer:

    def visualize(x, y, mean, var, plot_X):
        plt.scatter(x, y, color='red')
        plt.plot(plot_X, mean, color='blue')
        plt.fill_between(plot_X.squeeze(), (mean-np.sqrt(var)).squeeze(), (mean+np.sqrt(var)).squeeze(), alpha=0.5)
        plt.xlabel('$x$', fontsize=16)
        plt.ylabel('$y$', fontsize=16)
        plt.xlim(-8, 8)
        plt.ylim(-4, 4)
        plt.legend(['Predicted mean', 'Observed values', 'S.D.'], loc='upper left', fontsize=10)
        plt.show()


class GaussianProcess:

    def __init__(self, filename):
        self.x, self.y = GetData.file_open(filename)

    def kernel_operate(self, s, n, alpha, beta):
        k_size = len(self.x)
        K = np.zeros((k_size, k_size))
        for i in range(k_size):
            for j in range(k_size):
                K[i, j] = GetKernel.kernel(self.x[i], self.x[j], s)
        K = alpha * K + beta * np.eye(k_size)

        self.plot_X = np.linspace(-7, 7, n)
        self.mean = np.zeros(n)
        self.var = np.zeros(n)

        for i in range(n):
            k_st = np.zeros(len(self.x))
            for k in range(len(self.x)):
                k_st[k] = GetKernel.kernel(self.plot_X[i], self.x[k], s)
            k_st = np.reshape(alpha * k_st, (1, -1))
            k_stst = alpha * GetKernel.kernel(self.plot_X[i], self.plot_X[i], s) + beta
            self.mean[i] = np.dot(np.dot(k_st, np.linalg.inv(K)), self.y)
            self.var[i] = k_stst - np.dot(np.dot(k_st, np.linalg.inv(K)), k_st.T)

    def show_figure(self):
        Visualizer.visualize(x=self.x, y=self.y, mean=self.mean, var=self.var, plot_X=self.plot_X)


class GetData:

    def file_open(filename):
        f = open(filename, 'r')
        datalist = f.readlines()
        x = []
        y = []
        for data in datalist:
            data_split = data.split()
            x.append(float(data_split[0]))
            y.append(float(data_split[1]))

        f.close()
        return x, y


class GetKernel:

    def kernel(x, x_prime, s):
        return np.exp(-((x - x_prime)**2 / (2 * s)))


def main(filename, s, n, alpha, beta):
    gp_ins = GaussianProcess(filename)
    gp_ins.kernel_operate(s, n, alpha, beta)
    gp_ins.show_figure()

if __name__ == '__main__':
    filename = 'data_test.txt'
    s = 0.25
    n = 300
    alpha = 1 / 0.16
    beta = 1
    main(filename, s, n, alpha, beta)

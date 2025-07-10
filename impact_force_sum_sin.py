import numpy as np
import matplotlib.pyplot as plt

try:
    num_pieces = 10
    num_points = 100

    if num_pieces <= 0 or num_points <= 0:
        raise ValueError("num_pieces 和 num_points 必须为正整数")

    delta_t = 2 * np.pi / (num_pieces + 1)
    ft = np.zeros((num_pieces, num_points))

    for i in range(num_pieces):
        t = np.linspace(i * delta_t, (i + 1) * delta_t, num_points)
        ft[i] = np.sum([np.sin(t - j * delta_t) for j in range(i + 1)], axis=0)

        plt.plot(t, ft[i], '-*')
        plt.xlabel('Time (s)')
        plt.ylabel('Function value')
        plt.title('Function value over time')

    plt.show()
except ValueError as e:
    print(f"输入错误: {e}")
except Exception as e:
    print(f"发生错误: {e}")

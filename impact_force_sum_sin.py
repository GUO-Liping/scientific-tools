'''
Author: Liping0_0 liping0_0@my.swjtu.edu.cn
Date: 2025-07-10 22:12:50
LastEditors: Liping0_0 liping0_0@my.swjtu.edu.cn
LastEditTime: 2025-07-10 22:57:07
FilePath: \GitHub_files\scientific-tools\impact_force_sum_sin.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
        for j in range(i + 1):
            ft[i] = ft[i]  + np.sin(t - j * delta_t)

        plt.plot(t, np.cos(t - j * delta_t), '-*')
        plt.plot(t, ft[i], '-*')
        plt.xlabel('Time (s)')
        plt.ylabel('Function value')
        plt.title('Function value over time')

    plt.show()
except ValueError as e:
    print(f"输入错误: {e}")
except Exception as e:
    print(f"发生错误: {e}")

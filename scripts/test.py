import numpy as np
import matplotlib.pyplot as plt

# 定义能量函数
theta2 = np.linspace(-np.pi, np.pi, 500)
E_p = 1 + np.cos(theta2)

plt.figure(figsize=(8, 5))
plt.plot(theta2, E_p, label=r'$E_p = 1 + \cos(\theta_2)$', color='blue', linewidth=2)

# 标注非凸区域 (Hessian < 0)
plt.axvspan(-np.pi/2, np.pi/2, color='red', alpha=0.1, label='Non-convex Region (Hessian < 0)')

# 标注关键点
plt.scatter([0], [2], color='red', s=100, zorder=5) # 局部极大值
plt.text(0.1, 2.05, 'Local Maximum (Peak)', fontsize=10, color='red')

plt.scatter([-np.pi, np.pi], [0, 0], color='green', s=100, zorder=5) # 全局极小值
plt.text(np.pi-1.2, 0.1, 'Global Minima (Valleys)', fontsize=10, color='green')

# 装饰图表
plt.title('Energy Landscape of Motion Retargeting (Planar Two-link Arm)')
plt.xlabel(r'Joint Angle $\theta_2$ (rad)')
plt.ylabel('Position Error Cost $E_p$')
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], 
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
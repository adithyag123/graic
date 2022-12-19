import numpy as np
dx = 3
dy = 3
lst = np.arange(-0.5,0.5,0.1)
for yaw in lst:
    rx = np.cos(-yaw) * dx - np.sin(-yaw) * dy
    ry = np.cos(-yaw) * dy + np.sin(-yaw) * dx

    psi = np.arctan(ry / rx)
    print(yaw, psi, rx, ry)

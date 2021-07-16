import numpy as np
import scipy
import scipy.optimize as sop


def main():
    img_size = 400
    offset = img_size/2
    kp_2 = np.array([150, 180])
    kp_3 = np.array([-5, -1.23, 30])
    kp_x, kp_y, kp_z = kp_3
    fov = 50
    fx = offset / (np.tan(np.deg2rad(fov / 2)) * kp_z)
    fy = fx
    pi_t = np.array(
        [[fx, 0, 0],
         [0, fy, 0],
         [0, 0, 1]],
    )
    projected_kp = pi_t@kp_3
    projected_kp[:2] += offset
    print(projected_kp)
    residuals = kp_2-projected_kp[:2]
    print(residuals)


if __name__ == '__main__':
    main()

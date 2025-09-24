import numpy as np
import matplotlib.pyplot as plt


plot_color = ["r", "b", "m", "k", "g", "y", "k", "r--", "b--", "m--", "k--", "g--", "y--", "k--"]
c = 342 


f = np.linspace(0, 1e5, 300, dtype=complex)
omega = 2*np.pi*f
k = omega/c

print(f"k_max = {max(k)}")
theta = 60*np.pi/180
k_x_0 = k*np.sin(theta) # wave number in x direction
D = 2e-2 # Period of the grating

plt.figure()
for i, d in enumerate(range(-2,2)):
    k_x = k_x_0+d*2*np.pi/D
    k_y = np.sqrt(k**2 - k_x**2)
    if d <0:
        eps = -1
    else:
        eps = 1

    f_d = eps*d*c /(D*(1-eps*np.sin(theta)))
    # plt.figure(d)
    plt.plot(f, np.real(k_y), plot_color[i], label=f"real(k_y) n={d}")
    # plt.plot(f, np.imag(k_y), plot_color[i]+"--", label=f"imag(k_y) n={d}")
    # plt.plot(np.real(k_y), np.imag(k_y), plot_color[i], label=f"n={i}")
    plt.plot(f_d, 0, plot_color[i]+"o")
    plt.legend()
plt.show()

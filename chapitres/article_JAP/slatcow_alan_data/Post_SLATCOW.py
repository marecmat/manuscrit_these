import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from mpl_toolkits.mplot3d import Axes3D
import sys

# Clear variables and close figures
plt.close('all')

# Load data
spectral_data = scipy.io.loadmat('bicouche_spectral.mat')
freqs = spectral_data['freqs'].flatten()
k1_values = spectral_data['k1_values'].squeeze()

#slatcow_data = scipy.io.loadmat('slatcow.mat')
#freq_slatcow = slatcow_data['freq_slatcow'].squeeze()
#k_slatcow = slatcow_data['k_slatcow'].squeeze()
#A_slatcow = slatcow_data['A_slatcow'].squeeze()

svd_data = scipy.io.loadmat('Retrivied_SVD_V2.mat')
freq_slatcow = svd_data['freq'].squeeze()
k1_7 = svd_data['kk_com_SVD1'][:, 0]
k2_7 = svd_data['kk_com_SVD1'][:, 1]
k3_7 = svd_data['kk_com_SVD1'][:, 2]

k1_8 = svd_data['kk_com_SVD2'][:, 0]
k2_8 = svd_data['kk_com_SVD2'][:, 1]
k3_8 = svd_data['kk_com_SVD2'][:, 2]

A1_7 = svd_data['A_com_SVD1'][:, 0]
A2_7 = svd_data['A_com_SVD1'][:, 1]
A3_7 = svd_data['A_com_SVD1'][:, 2]

A1_8 = svd_data['A_com_SVD2'][:, 0]
A2_8 = svd_data['A_com_SVD2'][:, 1]
A3_8 = svd_data['A_com_SVD2'][:, 2]

k_slatcow = k1_7
A_slatcow = A1_7

ind_freq = 20

file_name = f'F_mes_freq_{ind_freq}.mat'
LAP_MES = scipy.io.loadmat(file_name)

F_mes = LAP_MES['F_mes']

freq_data = scipy.io.loadmat('freq.mat')
freq= freq_data['freq'].squeeze()

Kap_rel_data = scipy.io.loadmat('Kap_rel.mat')
Kap_rel= Kap_rel_data['Kap_rel']

Kap_img_data = scipy.io.loadmat('Kap_img.mat')
Kap_img= Kap_img_data['Kap_img']

kappa_rel_x = Kap_rel[0,:].squeeze()
kappa_img_x = Kap_img[:,0].squeeze()

x_data = scipy.io.loadmat('x.mat')
x= x_data['x'].squeeze()
lx = x[-1] - x[0]

# Define Lap_an_1D
def Lap_an_1D(kappa_rel_x, kappa_img_x, lx,k11_7,k12_7,k13_7,A11_7,A12_7,A13_7):

     kx_n = np.concatenate([np.array([k11_7]),np.array([k12_7]),np.array([k13_7])  ])
     A = np.concatenate([np.array([A11_7]),np.array([A12_7]),np.array([A13_7])  ])
     M, N = len(kappa_rel_x), len(kappa_img_x)
     Kappa_x = kappa_rel_x[:, None] - 1j * kappa_img_x
     delta_kx_pl = kx_n[:, None] - Kappa_x.flatten()
     L_x_pl = lx * np.exp(1j * delta_kx_pl * lx / 2) * np.sinc(delta_kx_pl * lx / (2 * np.pi))
     LAP_W_an = np.sum(A[:, None] * L_x_pl, axis=0).reshape(M, N)
     

     return LAP_W_an



F_Lap_an = Lap_an_1D(kappa_rel_x, kappa_img_x, lx,k1_7[ind_freq],k2_7[ind_freq],k3_7[ind_freq],A1_7[ind_freq],A2_7[ind_freq],A3_7[ind_freq])
F_Lap_an = F_Lap_an.T

# Frequency range
fr_mi = np.min(freq_slatcow)
fr_ma = np.max(freq_slatcow)

# Create figure
fig = plt.figure(1, figsize=(15, 10))
fig.text(0.5, 0.95, f'Freq = {freq_slatcow[ind_freq]}', color='black', 
         fontsize=16, ha='center')
fig.text(0.7, 0.95, r'$\ast$', color='cyan', fontsize=16, ha='center')
ax1 = fig.add_subplot(2, 3, 1)
# Plot real part of k
ax1.plot(np.real(k1_values), freqs, '.k', 
            np.real(k_slatcow), freq_slatcow, '.b',np.real(k_slatcow[ind_freq]), freq_slatcow[ind_freq], '*c',  markersize=12)
ax1.grid(True)
ax1.set_ylabel(r'$f \; [Hz]$', fontsize=16)
ax1.set_ylabel(r'$f \; [Hz]$', fontsize=16)
ax1.set_xlabel(r'$k_r$', fontsize=16)
ax1.set_xlim([np.min(np.real(k_slatcow)), np.max(np.real(k_slatcow))])
ax1.set_ylim([fr_mi, fr_ma])

# Plot imaginary part of k
ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(np.imag(k1_values), freqs, '.k', 
            np.imag(k_slatcow), freq_slatcow, '.b',np.imag(k_slatcow[ind_freq]), freq_slatcow[ind_freq], '*c', markersize=12)
ax2.grid(True)
ax2.set_xlabel(r'$k_i$', fontsize=16)
#ax2.set_xlim([np.min(np.imag(k_slatcow)), np.max(np.imag(k_slatcow))])
#ax2.set_ylim([fr_mi, fr_ma])

# Plot magnitude of A_slatcow
ax3 = fig.add_subplot(2, 3, 3)
ax3.plot(np.abs(A_slatcow), freq_slatcow, '-b',np.abs(A_slatcow[ind_freq]), freq_slatcow[ind_freq], '*c', markersize=12, linewidth=3)
ax3.grid(True)
ax3.set_xlabel(r'$A$', fontsize=16)
#ax3.set_xlim([np.min(np.abs(A_slatcow)), np.max(np.abs(A_slatcow))])
#ax3.set_ylim([fr_mi, fr_ma])

# Subplot 1: Surface plot of abs(F_mes)
#ax1 = fig.add_subplot(6, 6, [1, 2, 7, 8], projection='3d')
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
ax4.plot_surface(Kap_rel, Kap_img, np.abs(F_mes), cmap='jet', shade=True)
ax4.set_xlabel(r'$\kappa_{r}$', fontsize=16, labelpad=10)
ax4.set_ylabel(r'$\kappa_{i}$', fontsize=16, labelpad=10)
ax4.set_zlabel(r'$\mid L(z)_{Mes}\mid$', fontsize=16, labelpad=10)

ax5 = fig.add_subplot(2, 3, 5, projection='3d')
ax5.plot_surface(Kap_rel, Kap_img, np.abs(F_Lap_an), cmap='jet', shade=True)
ax5.set_xlabel(r'$\kappa_{r}$', fontsize=16, labelpad=10)
ax5.set_ylabel(r'$\kappa_{i}$', fontsize=16, labelpad=10)
ax5.set_zlabel(r'$\mid L(z)_{Recon}\mid$', fontsize=16, labelpad=10)

plt.show()

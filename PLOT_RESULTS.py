import os
import sys
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib        as mpl
import matplotlib.cm as cm
from scipy.stats import norm
from matplotlib import rc
from Chain2xz import Chain2xz
from GridModel import GridModel
from scipy.spatial import Voronoi, voronoi_plot_2d
#######################################################################

file_name = '2022_03_11-05_00_37_PM'

fpath_Restart = os.getcwd()+'//'+file_name+'//'+'Restart'

dg_obs = np.load(fpath_Restart+'//'+'dg_obs.npy')
dT_obs = np.load(fpath_Restart+'//'+'dT_obs.npy')
XnZn = np.load(fpath_Restart+'//'+'XnZn.npy')
TrueDensityModel = np.load(fpath_Restart+'//'+'TrueDensityModel.npy')
TrueSUSModel = np.load(fpath_Restart+'//'+'TrueSUSModel.npy')
AR_parameters_original_g = np.load(fpath_Restart+'//'+'AR_parameters_original_g.npy')
AR_parameters_original_T = np.load(fpath_Restart+'//'+'AR_parameters_original_T.npy')
globals_par = np.load(fpath_Restart+'//'+'globals_par.npy')
globals_xyz = np.load(fpath_Restart+'//'+'globals_xyz.npy')
AR_bounds = np.load(fpath_Restart+'//'+'AR_bounds.npy')
# Chain_raw = np.load(fpath_Restart+'//'+'Chain_raw.npy')
Chain_All = np.load(fpath_Restart+'//'+'ChainAll.npy')
WhiteBlueGreenYellowRed = np.loadtxt(os.getcwd()+'//'+'WhiteBlueGreenYellowRed.txt')
WBGR = np.ones((WhiteBlueGreenYellowRed.shape[0],4))
WBGR[:,:-1] = WhiteBlueGreenYellowRed.copy()
WBGR = mpl.colors.ListedColormap(WBGR, name='WBGR', N=WBGR.shape[0])
########################################################################
fpath_output = os.getcwd()+'//'+file_name+'//'+'Output'

LogLkeep = np.load(fpath_output+'//'+'LogLkeep.npy')
Nnode = np.load(fpath_output+'//'+'Nnode.npy')
NARg = np.load(fpath_output+'//'+'NARg.npy')
NART = np.load(fpath_output+'//'+'NART.npy')
ARgkeep = np.load(fpath_output+'//'+'ARgkeep.npy')
ARTkeep = np.load(fpath_output+'//'+'ARTkeep.npy')
PMD_g = np.load(fpath_output+'//'+'PMD_g.npy')
STD_g = np.load(fpath_output+'//'+'STD_g.npy')
PMD_T = np.load(fpath_output+'//'+'PMD_T.npy')
STD_T = np.load(fpath_output+'//'+'STD_T.npy')
rho_keep = np.load(fpath_output+'//'+'rho_keep.npy')
PMD_data_g = np.load(fpath_output+'//'+'PMD_data_g.npy')
STD_data_g = np.load(fpath_output+'//'+'STD_data_g.npy')
PMD_data_T = np.load(fpath_output+'//'+'PMD_data_T.npy')
STD_data_T = np.load(fpath_output+'//'+'STD_data_T.npy')
sigma_g = np.load(fpath_output+'//'+'sigma_g.npy')
sigma_T = np.load(fpath_output+'//'+'sigma_T.npy')
PMD_Cov_g = np.load(fpath_output+'//'+'PMD_Cov_g.npy')
PMD_Cov_T = np.load(fpath_output+'//'+'PMD_Cov_T.npy')
PMD_Cov_g_0 = np.load(fpath_output+'//'+'PMD_Cov_g_0.npy')
PMD_Cov_T_0 = np.load(fpath_output+'//'+'PMD_Cov_T_0.npy')
PMD_Cov_g_1 = np.load(fpath_output+'//'+'PMD_Cov_g_1.npy')
PMD_Cov_T_1 = np.load(fpath_output+'//'+'PMD_Cov_T_1.npy')
PMD_Cov_g_2 = np.load(fpath_output+'//'+'PMD_Cov_g_2.npy')
PMD_Cov_T_2 = np.load(fpath_output+'//'+'PMD_Cov_T_2.npy')
PMD_Cov_g_3 = np.load(fpath_output+'//'+'PMD_Cov_g_3.npy')
PMD_Cov_T_3 = np.load(fpath_output+'//'+'PMD_Cov_T_3.npy')
PMD_autocorr_rg = np.load(fpath_output+'//'+'PMD_autocorr_rg.npy')
PMD_autocorr_rT = np.load(fpath_output+'//'+'PMD_autocorr_rT.npy')
PMD_autocorr_stand_rg = np.load(fpath_output+'//'+'PMD_autocorr_stand_rg.npy')
PMD_autocorr_stand_rT = np.load(fpath_output+'//'+'PMD_autocorr_stand_rT.npy')
PMD_standardized_rg = np.load(fpath_output+'//'+'PMD_standardized_rg.npy')
PMD_standardized_rT = np.load(fpath_output+'//'+'PMD_standardized_rT.npy')


Ndata = PMD_data_g.size

CI_g_Low  = PMD_g - 1.96 * STD_g
CI_g_High = PMD_g + 1.96 * STD_g
CI_g_Width = abs(CI_g_High-CI_g_Low)

CI_T_Low  = PMD_T - 1.96 * STD_T
CI_T_High = PMD_T + 1.96 * STD_T
CI_T_Width = abs(CI_T_High-CI_T_Low)

data_g_error = 1.96 * STD_data_g
data_T_error = 1.96 * STD_data_T
###############################################################################
fpath_plots = os.getcwd()+'//'+file_name+'//'+'Plots'
if os.path.exists(fpath_plots) and os.path.isdir(fpath_plots):
    shutil.rmtree(fpath_plots)
os.mkdir(fpath_plots)

##############################################################################

Kmin = int(globals_par[0,0])
Kmax = int(globals_par[0,1])
rho_salt_min = globals_par[1,0]
rho_salt_max = globals_par[1,1]
rho_base_min = globals_par[2,0]
rho_base_max = globals_par[2,1]
KminAR = int(globals_par[3,0])
KmaxAR = int(globals_par[3,1])

AR0_min = AR_bounds[0,0]
AR0_max = AR_bounds[0,1]
AR1_min = AR_bounds[1,0]
AR1_max = AR_bounds[1,1]
AR2_min = AR_bounds[2,0]
AR2_max = AR_bounds[2,1]
AR3_min = AR_bounds[3,0]
AR3_max = AR_bounds[3,1]

dis_min = globals_xyz[0,0]
dis_max = globals_xyz[0,1]
z_min = globals_xyz[1,0]
z_max = globals_xyz[1,1]

###############################################################################
### Plot LogL
fig, axe = plt.subplots(2, 1)
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

axe[0].plot(LogLkeep, 'k-',linewidth=2) #row=0, col=0
axe[0].set_ylabel('Log Likelihood',fontweight="bold", fontsize = 10)
axe[0].set_xlabel('(a) rjMCMC STEP',fontweight="bold", fontsize = 10)

axe[0].xaxis.set_label_position("top")
axe[0].xaxis.tick_top()
# axe[0].get_xaxis().get_major_formatter().set_scientific(False)
plt.show()

axe[1].hist(Nnode, 13, density=True, color='0.5') 
axe[1].set_ylabel('pdf',fontweight="bold", fontsize = 10)
axe[1].set_xlabel('(b) Number of Nodes',fontweight="bold", fontsize = 10)

# axe[1].yaxis.set_label_position("right")
# axe[1].yaxis.tick_right()
plt.show()

figname = 'LogL_NNode'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window


#########################################################################################
## Plot Sigma
fig, axe = plt.subplots(2, 1)

axe[0].plot(sigma_g, 'k-',linewidth=2) #row=0, col=0
axe[0].set(xlabel='rjMCMC STEP', ylabel='STD of gravity residuals (mGal)')
plt.show()

axe[1].plot(sigma_T, 'k-',linewidth=2) #row=1, col=0
axe[1].set(xlabel='rjMCMC STEP', ylabel='STD of magnetic residuals (nT)')
axe[1].yaxis.set_label_position("right")
axe[1].yaxis.tick_right()
plt.show()
figname = 'Sigma'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#########################################################################################
### Plot rho

rho_salt = rho_keep[rho_keep < 0]
rho_base = rho_keep[rho_keep > 0]

fig, axe = plt.subplots(2, 1)
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

axe[0].hist(rho_salt, 8, density=True, color='0.5') 
axe[0].set_ylabel('pdf',fontweight="bold", fontsize = 10)
axe[0].set_xlabel('(a) density contrast of salt',fontweight="bold", fontsize = 10)
axe[0].locator_params(axis='x', nbins=5)

axe[0].xaxis.set_label_position("top")
axe[0].xaxis.tick_top()
axe[0].get_xaxis().get_major_formatter().set_scientific(False)
plt.show()

axe[1].hist(rho_base, 8, density=True, color='0.5') 
axe[1].set_ylabel('pdf',fontweight="bold", fontsize = 10)
axe[1].set_xlabel('(b) density contrast of basement',fontweight="bold", fontsize = 10)
axe[1].locator_params(axis='x', nbins=5)

# axe[1].yaxis.set_label_position("right")
# axe[1].yaxis.tick_right()
plt.show()

figname = 'rho_hist'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig) 


#############################################################################################
### Plot Gravity AR and Cov

TF0 = np.squeeze(ARgkeep[:,0]==0)
AR0 = ARgkeep[TF0,0:1].copy()

TF1 = np.squeeze((NARg==1) & (ARgkeep[:,0]!=0))
AR1 = ARgkeep[TF1,0:1].copy()
    
TF2 = np.squeeze(NARg==2)
AR2 = ARgkeep[TF2,0:2].copy() 
    
TF3 = np.squeeze(NARg==3)
AR3 = ARgkeep[TF3,0:3].copy()

barval_g = [np.sum(TF0), np.sum(TF1), np.sum(TF2), np.sum(TF3)]/(np.sum(TF0)+np.sum(TF1)+np.sum(TF2)+np.sum(TF3)).copy()

n_bins = 40

fig, axs = plt.subplots(4,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0.05, 'hspace': 0},figsize=(10, 10))
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

axs[0,0].hist(AR0, n_bins, density=True, histtype='step', fill=False, linewidth=2) 
axs[0,0].axvline(x=AR_parameters_original_g[0], color='black', ls='--', lw=2)
axs[0,0].set_xlim([-1, 1])
axs[0,0].set_xticklabels([])
axs[0,0].set_ylabel('pdf AR(0)',fontweight="bold", fontsize = 12)
axs[0,0].text(-0.9,35,'(a)', fontsize=20)

axs[1,0].hist(AR1, n_bins, density=True, histtype='step', fill=False, linewidth=2) 
axs[1,0].axvline(x=AR_parameters_original_g[0], color='black', ls='--', lw=2)
axs[1,0].set_xlim([-1, 1])
axs[1,0].set_xticklabels([])
axs[1,0].set_ylabel('pdf AR(1)',fontweight="bold", fontsize = 12)
axs[1,0].text(-0.9,1.6,'(b)', fontsize=20)

axs[2,0].hist(AR2, 12, density=True, histtype='step', fill=False, linewidth=2) 
axs[2,0].axvline(x=AR_parameters_original_g[0], color='black', ls='--', lw=2)
axs[2,0].axvline(x=AR_parameters_original_g[1], color='black', ls='--', lw=2)
axs[2,0].set_xlim([-1, 1])
axs[2,0].set_xticklabels([])
axs[2,0].set_ylabel('pdf AR(2)',fontweight="bold", fontsize = 12)
axs[2,0].text(-0.9,1.6,'(c)', fontsize=20)


axs[3,0].hist(AR3, 20, density=True, histtype='step', fill=False, linewidth=2) 
axs[3,0].axvline(x=AR_parameters_original_g[0], color='black', ls='--', lw=2)
axs[3,0].axvline(x=AR_parameters_original_g[1], color='black', ls='--', lw=2)
axs[3,0].set_xlim([-1, 1])
axs[3,0].set_xlabel('AR Coefficent',fontweight="bold", fontsize = 20)
axs[3,0].set_ylabel('pdf AR(3)',fontweight="bold", fontsize = 12)
axs[3,0].text(-0.9,2.3,'(d)', fontsize=20)

axs[0,1].plot(PMD_Cov_g_0[14,:], color='black', lw=2)
axs[0,1].yaxis.tick_right()
axs[0,1].set_xticklabels([])
axs[0,1].set_ylabel('Covariance ($\mathregular{mGal^{2}}$)',fontweight="bold", fontsize = 12)
axs[0,1].yaxis.set_label_position("right")
axs[0,1].text(0.7,0.03,'(e)', fontsize=20)

axs[1,1].plot(PMD_Cov_g_1[14,:], color='black', lw=2)
axs[1,1].yaxis.tick_right()
axs[1,1].set_xticklabels([])
axs[1,1].set_ylabel('Covariance ($\mathregular{mGal^{2}}$)',fontweight="bold", fontsize = 12)
axs[1,1].yaxis.set_label_position("right")
axs[1,1].text(0.7,0.06,'(f)', fontsize=20)


axs[2,1].plot(PMD_Cov_g_2[14,:], color='black', lw=2)
axs[2,1].yaxis.tick_right()
axs[2,1].set_xticklabels([])
axs[2,1].set_ylabel('Covariance ($\mathregular{mGal^{2}}$)',fontweight="bold", fontsize = 12)
axs[2,1].yaxis.set_label_position("right")
axs[2,1].text(0.7,0.055,'(g)', fontsize=20)

axs[3,1].plot(PMD_Cov_g_3[14,:], color='black', lw=2)
axs[3,1].yaxis.tick_right()
axs[3,1].set_xlabel('lag',fontweight="bold", fontsize = 20)
axs[3,1].set_ylabel('Covariance ($\mathregular{mGal^{2}}$)',fontweight="bold", fontsize = 12)
axs[3,1].yaxis.set_label_position("right")
axs[3,1].text(0.7,0.06,'(h)', fontsize=20)

plt.show()
figname = 'AR_Cov_gravity'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

##########################################################################################
### Plot Magnetic AR and Cov

TF0 = np.squeeze(ARTkeep[:,0]==0)
AR0 = ARTkeep[TF0,0:1].copy()

TF1 = np.squeeze((NART==1) & (ARTkeep[:,0]!=0))
AR1 = ARTkeep[TF1,0:1].copy()
    
TF2 = np.squeeze(NART==2)
AR2 = ARTkeep[TF2,0:2].copy() 
    
TF3 = np.squeeze(NART==3)
AR3 = ARTkeep[TF3,0:3].copy()
    
barval_T = [np.sum(TF0), np.sum(TF1), np.sum(TF2), np.sum(TF3)]/(np.sum(TF0)+np.sum(TF1)+np.sum(TF2)+np.sum(TF3)).copy()

n_bins = 40

fig, axs = plt.subplots(4,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0.05, 'hspace': 0},figsize=(10, 10))
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

axs[0,0].hist(AR0, n_bins, density=True, histtype='step', fill=False, linewidth=2) 
axs[0,0].axvline(x=AR_parameters_original_T[0], color='black', ls='--', lw=2)
axs[0,0].set_xlim([-1, 1])
axs[0,0].set_xticklabels([])
axs[0,0].set_ylabel('pdf AR(0)',fontweight="bold", fontsize = 12)
axs[0,0].text(-0.9,35,'(a)', fontsize=20)

axs[1,0].hist(AR1, n_bins, density=True, histtype='step', fill=False, linewidth=2) 
axs[1,0].axvline(x=AR_parameters_original_T[0], color='black', ls='--', lw=2)
axs[1,0].set_xlim([-1, 1])
axs[1,0].set_xticklabels([])
axs[1,0].set_ylabel('pdf AR(1)',fontweight="bold", fontsize = 12)
axs[1,0].text(-0.9,1,'(b)', fontsize=20)


axs[2,0].hist(AR2, 13, density=True, histtype='step', fill=False, linewidth=2) 
axs[2,0].axvline(x=AR_parameters_original_T[0], color='black', ls='--', lw=2)
axs[2,0].set_xlim([-1, 1])
axs[2,0].set_xticklabels([])
axs[2,0].set_ylabel('pdf AR(2)',fontweight="bold", fontsize = 12)
axs[2,0].text(-0.9,2,'(c)', fontsize=20)

axs[3,0].hist(AR3, 15, density=True, histtype='step', fill=False, linewidth=2) 
axs[3,0].axvline(x=AR_parameters_original_T[0], color='black', ls='--', lw=2)
axs[3,0].set_xlim([-1, 1])
axs[3,0].set_xlabel('AR Coefficent',fontweight="bold", fontsize = 20)
axs[3,0].set_ylabel('pdf AR(3)',fontweight="bold", fontsize = 12)
axs[3,0].text(-0.9,2.4,'(d)', fontsize=20)


axs[0,1].plot(PMD_Cov_T_0[14,:], color='black', lw=2)
axs[0,1].yaxis.tick_right()
axs[0,1].set_xticklabels([])
axs[0,1].set_ylabel('Covariance ($\mathregular{nT^{2}}$)',fontweight="bold", fontsize = 12)
axs[0,1].yaxis.set_label_position("right")
axs[0,1].text(0.7,0.8,'(e)', fontsize=20)

axs[1,1].plot(PMD_Cov_T_1[14,:], color='black', lw=2)
axs[1,1].yaxis.tick_right()
axs[1,1].set_xticklabels([])
axs[1,1].set_ylabel('Covariance ($\mathregular{nT^{2}}$)',fontweight="bold", fontsize = 12)
axs[1,1].yaxis.set_label_position("right")
axs[1,1].text(0.7,1,'(f)', fontsize=20)

axs[2,1].plot(PMD_Cov_T_2[14,:], color='black', lw=2)
axs[2,1].yaxis.tick_right()
axs[2,1].set_xticklabels([])
axs[2,1].set_ylabel('Covariance ($\mathregular{nT^{2}}$)',fontweight="bold", fontsize = 12)
axs[2,1].yaxis.set_label_position("right")
axs[2,1].text(0.7,1,'(g)', fontsize=20)

axs[3,1].plot(PMD_Cov_T_3[14,:], color='black', lw=2)
axs[3,1].yaxis.tick_right()
axs[3,1].set_xlabel('lag',fontweight="bold", fontsize = 20)
axs[3,1].set_ylabel('Covariance ($\mathregular{nT^{2}}$)',fontweight="bold", fontsize = 12)
axs[3,1].yaxis.set_label_position("right")
axs[3,1].text(0.7,1,'(h)', fontsize=20)


plt.show()
figname = 'AR_Cov_Magnetic'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
#############################################################################################

### Plot AR Order Hist

fig, axs = plt.subplots(1,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 5))
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

bars = ('AR(0)', 'AR(1)', 'AR(2)', 'AR(3)')
y_pos = np.arange(len(bars))
axs[0].bar(y_pos, barval_g,color=['red', 'blue', 'purple', 'black'])
axs[0].set_xticks(y_pos)
axs[0].set_xticklabels(bars)
axs[0].set_xlabel('(a) AR Gravity Orders',fontweight="bold", fontsize = 10)
axs[0].set_ylabel('pdf',fontweight="bold", fontsize = 10)

axs[1].bar(y_pos, barval_T,color=['red', 'blue', 'purple', 'black'])
axs[1].set_xticks(y_pos)
axs[1].set_xticklabels(bars)
axs[1].set_xlabel('(b) AR Magnetic Orders',fontweight="bold", fontsize = 10)
axs[1].set_ylabel('pdf',fontweight="bold", fontsize = 10)
axs[1].yaxis.tick_right()
axs[1].yaxis.set_label_position("right")

plt.show()
figname = 'AR_Order'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
##########################################################################################
### Plot Model

x1 = dis_min/1000
x2 = dis_max/1000
z1 = z_min/1000
z2 = z_max/1000
     
fig, axs = plt.subplots(3,2, sharex=True, sharey=True ,gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 8))
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

pos00 = axs[0,0].get_position() # get the original position
pos10 = axs[1,0].get_position() # get the original position 
pos20 = axs[2,0].get_position() # get the original position 
pos01 = axs[0,1].get_position() # get the original position
pos11 = axs[1,1].get_position() # get the original position
pos21 = axs[2,1].get_position() # get the original position
pos00.x0 += 0.1  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
pos10.x0 += 0.1
pos20.x0 += 0.1
pos01.x1 -= 0.1
pos11.x1 -= 0.1
pos21.x1 -= 0.1
axs[0,0].set_position(pos00) # set a new position
axs[1,0].set_position(pos10) # set a new position
axs[2,0].set_position(pos20) # set a new position
axs[0,1].set_position(pos01) # set a new position
axs[1,1].set_position(pos11) # set a new position
axs[2,1].set_position(pos21) # set a new position

im00 = axs[0,0].imshow(TrueDensityModel,interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[0,0].text(0.1,.9,'(a)',horizontalalignment='center',transform=axs[0,0].transAxes, fontweight="bold", fontsize = 12)
axs[0,0].set_ylabel('Depth (km)',fontweight="bold", fontsize = 12)
# axs[0,0].set_xticklabels([])
axs[0,0].yaxis.tick_left()
axs[0,0].tick_params(axis="x",direction="in")
# axs[0,0].tick_params(axis='both', labelsize=15)


im10 = axs[1,0].imshow(PMD_g,interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap='seismic')
axs[1,0].text(0.1,.9,'(b)',horizontalalignment='center',transform=axs[1,0].transAxes, fontweight="bold", fontsize = 12)
axs[1,0].set_ylabel('Depth (km)',fontweight="bold", fontsize = 12)
# axs[1,0].set_xticklabels([])
axs[1,0].yaxis.tick_left()
axs[1,0].tick_params(axis="x",direction="in")

CI_g_Width_NaN = CI_g_Width.copy()
CI_g_Width_NaN[CI_g_Width==0] = np.NaN
NewJet = cm.jet
NewJet.set_bad("white")
im20 = axs[2,0].imshow(CI_g_Width_NaN,interpolation='none',
       vmin=0, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect='auto', cmap=NewJet)
axs[2,0].text(0.1,.9,'(c)',horizontalalignment='center',transform=axs[2,0].transAxes, fontweight="bold", fontsize = 12)
axs[2,0].set_ylabel('Depth (km)',fontweight="bold", fontsize = 12)
axs[2,0].set_xlabel('Distance (km)',fontweight="bold", fontsize = 12)
axs[2,0].yaxis.tick_left()

im01 = axs[0,1].imshow(TrueSUSModel,interpolation='none',
       vmin=0, vmax=rho_base_max/50., extent=(x1,x2,z2,z1), aspect='auto', cmap=WBGR)
axs[0,1].text(0.1,.9,'(d)',horizontalalignment='center',transform=axs[0,1].transAxes, fontweight="bold", fontsize = 12)
# axs[0,1].set_xticklabels([])
axs[0,1].tick_params(axis="y",direction="in")
axs[0,1].tick_params(axis="x",direction="in")

im11 = axs[1,1].imshow(PMD_T,interpolation='none',
       vmin=0, vmax=rho_base_max/50., extent=(x1,x2,z2,z1), aspect='auto', cmap=WBGR)
axs[1,1].text(0.1,.9,'(e)',horizontalalignment='center',transform=axs[1,1].transAxes, fontweight="bold", fontsize = 12)
# axs[1,1].set_xticklabels([])
axs[1,1].tick_params(axis="y",direction="in")
axs[1,1].tick_params(axis="x",direction="in")

im21 = axs[2,1].imshow(CI_T_Width,interpolation='none',
       vmin=0, vmax=rho_base_max/50., extent=(x1,x2,z2,z1), aspect='auto', cmap=WBGR)
axs[2,1].text(0.1,.9,'(f)',horizontalalignment='center',transform=axs[2,1].transAxes, fontweight="bold", fontsize = 12)
axs[2,1].set_xlabel('Distance (km)',fontweight="bold", fontsize = 12)
axs[2,1].tick_params(axis="y",direction="in")

cbar_pos_density = fig.add_axes([0.1, 0.4, 0.03, 0.45]) 
cbar_density = plt.colorbar(im00, ax=axs[0,0] ,shrink=0.3, cax = cbar_pos_density,
                    orientation='vertical', ticklocation = 'left')
cbar_density.ax.tick_params(labelsize=12)
cbar_density.set_label(label = 'Density Contrast ($\mathregular{g/cm^{3}}$)', weight='bold')


cbar_pos_jet = fig.add_axes([0.1, 0.1, 0.03, 0.2]) 
cbar_jet = plt.colorbar(im20, ax=axs[2,0] ,shrink=0.3,  cax = cbar_pos_jet,
                    orientation='vertical', ticklocation = 'left')
cbar_jet.ax.tick_params(labelsize=12)
cbar_jet.set_label(label='95% CI Widths ($\mathregular{g/cm^{3}}$)', weight='bold')


cbar_pos_sus = fig.add_axes([0.85, 0.4, 0.03, 0.45]) 
cbar_sus = plt.colorbar(im11, ax=axs[1,1] ,shrink=0.3, cax = cbar_pos_sus,
                    orientation='vertical', ticklocation = 'right')
cbar_sus.ax.tick_params(labelsize=12)
cbar_sus.set_label(label='Susceptibility (SI)', weight='bold')

cbar_pos_95_sus = fig.add_axes([0.85, 0.1, 0.03, 0.2]) 
cbar_95_sus = plt.colorbar(im11, ax=axs[1,1] ,shrink=0.3, cax = cbar_pos_95_sus,
                    orientation='vertical', ticklocation = 'right')
cbar_95_sus.ax.tick_params(labelsize=12)
cbar_95_sus.set_label(label='95% CI Widths (SI)', weight='bold')

# for ax in axs.flat:
#     ax.label_outer()

plt.show()
figname = 'True_PMD_Model'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
#########################################################################################
### Plot Data Fit
dis = np.linspace(dis_min,dis_max,Ndata)
dis = dis/1000.
fig, axe = plt.subplots(1, 2)
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

axe[0].fill_between(dis, PMD_data_g-data_g_error, PMD_data_g+data_g_error,facecolor='0.5')
# axe[0].errorbar(dis, PMD_data_g , yerr=data_g_error, fmt='.', markersize=6, capsize=4, linewidth=3, c='0.5')
axe[0].plot(dis,dg_obs, 'k.-',linewidth=2) #row=0, col=0
# axe[0].set(xlabel='Distance (km)', ylabel='Gravity (mGal)')
axe[0].set_xlabel('Distance (km)',fontweight="bold", fontsize = 8)
axe[0].set_ylabel('Gravity (mGal)',fontweight="bold", fontsize = 8)
axe[0].text(-5,5.4,'(a)', fontweight="bold", fontsize = 12)
plt.show()

axe[1].fill_between(dis, PMD_data_T-data_T_error, PMD_data_T+data_T_error,facecolor='0.5')
# axe[1].errorbar(dis, PMD_data_T , yerr=data_T_error, fmt='.', markersize=6, capsize=4, linewidth=3, c='0.5')
axe[1].plot(dis,dT_obs, 'k.-',linewidth=1) #row=0, col=0
# axe[1].set(xlabel='Distance (km)', ylabel='Magnetic (nT)')
axe[1].set_xlabel('Distance (km)',fontweight="bold", fontsize = 8)
axe[1].set_ylabel('Magnetic (nT)',fontweight="bold", fontsize = 8)
axe[1].yaxis.set_label_position("right")
axe[1].yaxis.tick_right()
axe[1].text(-5,52,'(b)', fontweight="bold", fontsize = 12)

plt.show()
figname = 'Data_Fit'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
#######################################################################################
### Plot PMD of Cov
x1 = 1
x2 = len(PMD_Cov_g)
z1 = 1
z2 = len(PMD_Cov_g)
fig, axs = plt.subplots(2, 2)
im0 = axs[0, 0].imshow(PMD_Cov_g,interpolation='none',
       extent=(x1,x2,z2,z1), aspect=(x2-x1)/(z2-z1), cmap='jet')
axs[0, 0].set_title('(a)')
axs[0, 0].set(xlabel='lag', ylabel='lag')
cbar0 = fig.colorbar(im0, ax=axs[0, 0], shrink=1, label='Covariance ($\mathregular{mGal^{2}}$)')

im1 = axs[0, 1].imshow(PMD_Cov_T,interpolation='none',
       extent=(x1,x2,z2,z1), aspect=(x2-x1)/(z2-z1), cmap='jet')
axs[0, 1].set_title('(b)')
axs[0, 1].set(xlabel='lag', ylabel='lag')
cbar0 = fig.colorbar(im1, ax=axs[0, 1], shrink=1, label='Covariance ($\mathregular{nT^{2}}$)')

axs[1, 0].plot(PMD_Cov_g[15,:],'k-',linewidth=2)
axs[1, 0].set_title('(c)')
axs[1, 0].set(xlabel='lag', ylabel='Covariance ($\mathregular{mGal^{2}}$)')

axs[1, 1].plot(PMD_Cov_T[15,:],'k-',linewidth=2)
axs[1, 1].set_title('(d)')
axs[1, 1].set(xlabel='lag', ylabel='Covariance ($\mathregular{nT^{2}}$)')
plt.tight_layout()

# for ax in axs.flat:
#     ax.set(xlabel='lag', ylabel='Covariance')

# Hide x labels and tick labels for top plots and y ticks for right plots.
# for ax in axs.flat:
#     ax.label_outer()
    
plt.show()
figname = 'PMD_Cov'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#####################################################################################
### Plot ACF

n_bins = 10
ynormal = norm.rvs(size=1000)

fig, axs = plt.subplots(2,2, sharex=False, sharey=False ,gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 10))
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)


axs[0,0].plot(PMD_autocorr_stand_rg,'k-',linewidth=3)
axs[0,0].plot(PMD_autocorr_rg, '--' ,linewidth=2, color = 'gray')
axs[0,0].set_ylabel('ACF (gravity residuals)',fontweight="bold", fontsize = 20)
axs[0,0].text(1,0.9,'(a)', fontsize=20)


axs[1,0].plot(PMD_autocorr_stand_rT,'k-',linewidth=3)
axs[1,0].plot(PMD_autocorr_rT, '--',linewidth=2, color = 'gray')
axs[1,0].set_xlabel('lag',fontweight="bold", fontsize = 20)
axs[1,0].set_ylabel('ACF (magnetic residuals)',fontweight="bold", fontsize = 20)
axs[1,0].text(1,0.9,'(c)', fontsize=20)

axs[0,1].hist(ynormal, n_bins , density=True, histtype='step', fill=False, hatch="++" , linewidth=2, color=('black')) 
axs[0,1].hist(PMD_data_g-dg_obs, n_bins , density=True, histtype='step', fill=False, linestyle=('dashed'), linewidth=2) 
axs[0,1].hist(PMD_standardized_rg, n_bins , density=True, histtype='step', fill=False, linestyle=('solid'), linewidth=2)
axs[0,1].set_ylabel('pdf (gravity residuals)',fontweight="bold", fontsize = 20)
axs[0,1].yaxis.tick_right()
axs[0,1].yaxis.set_label_position("right")
axs[0,1].text(-2.6,1.6,'(b)', fontsize=20)
axs[0,1].set_xlim([-3, 3])

axs[1,1].hist(ynormal, n_bins , density=True, histtype='step', fill=False, hatch="++" , linewidth=2, color=('black')) 
axs[1,1].hist(PMD_data_T-dT_obs, n_bins , density=True, histtype='step', fill=False, linestyle=('dashed'), linewidth=2) 
axs[1,1].hist(PMD_standardized_rT, n_bins , density=True, histtype='step', fill=False, linestyle=('solid'), linewidth=2)
axs[1,1].set_xlabel('residuals',fontweight="bold", fontsize = 20)
axs[1,1].set_ylabel('pdf (magnetic residuals)',fontweight="bold", fontsize = 20)
axs[1,1].yaxis.tick_right()
axs[1,1].yaxis.set_label_position("right")
axs[1,1].text(-2.6,0.7,'(d)', fontsize=20)
axs[1,1].set_xlim([-3, 3])

plt.show()
figname = 'ACF'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window

#######################################################################
### Plot MAXL model
x1 = dis_min/1000
x2 = dis_max/1000
z1 = z_min/1000
z2 = z_max/1000
ind = np.argsort(Chain_All[:,0])[::-1]
Chain_maxL = Chain_All[ind[0]].copy()
[xm, zm, x, z, rho, ARg, ART]= Chain2xz(Chain_maxL).copy()


CX = PMD_g.shape[0]
CZ = PMD_g.shape[1]
DensityModel = GridModel(x,z,rho,XnZn,CX,CZ)

fig, ax = plt.subplots(gridspec_kw={'wspace': 0, 'hspace': 0},figsize=(10, 10))
plt.rc('font', weight='bold')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)


pos00 = ax.get_position() # get the original position
pos00.x0 += 0.2  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
# pos00.top -= 0.2  # or use: pos00 = [pos00.x0 + 0.1, pos00.y0 ,  pos00.width, pos00.height] 
plt.subplots_adjust(left=0.3, right=0.9, bottom=0.2, top=0.7)
# ax.set_position(pos00) # set a new position

xbins = 5
zbins = 5
Xticklabels = np.around(np.linspace(x1,x2,xbins), 2)
Zticklabels = np.around(np.linspace(z1,z2,zbins), 2)


im00 = ax.imshow(DensityModel,interpolation='none',
       vmin=rho_salt_min, vmax=rho_base_max, extent=(0,1,1,0), aspect='auto', cmap='seismic')

# plt.locator_params(nbins=4)
plt.locator_params(axis='y', nbins=zbins-1)
plt.locator_params(axis='x', nbins=xbins-1)


ax.set_xticklabels(Xticklabels)
ax.set_yticklabels(Zticklabels)
# plt.xticks(Xticklabels)
# plt.yticks(Zticklabels)

plt.xlabel("Distance (km)",fontweight="bold", fontsize = 20)
plt.ylabel("Depth (km)",fontweight="bold", fontsize = 20)

cbar_pos_density = fig.add_axes([0.1, 0.2, 0.03, 0.4]) 
cbar_density = plt.colorbar(im00, ax=ax ,shrink=0.3, cax = cbar_pos_density,
                    orientation='vertical', ticklocation = 'left')
cbar_density.ax.tick_params(labelsize=15)
cbar_density.set_label(label = 'density contrast ($\mathregular{g/cm^{3}}$)', weight='bold')

ax.plot(x,z,'ko')

TrainPoints = np.column_stack((x,z)).copy()
#TrainPoints = np.vstack((TrainPoints, [0.2, 1.3]))

vor = Voronoi(TrainPoints)
voronoi_plot_2d(vor,  show_vertices=False, line_colors='black',
                line_width=3, line_alpha=1.0, point_size=0, ax=ax)

ax.plot(xm,zm,'mx')
ParentNodes = np.column_stack((xm,zm)).copy()
vorm = Voronoi(ParentNodes)
voronoi_plot_2d(vorm,  show_vertices=False, line_colors='m',
                line_width=3, line_alpha=1.0, point_size=0, ax=ax)

ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

ax.invert_yaxis()

plt.show()
figname = 'MaxL_Model'
fig.savefig(fpath_plots+'/'+figname+'.pdf')
plt.close(fig)    # close the figure window
###########################################################################

# fig, axs = plt.subplots(2,2, sharex=True, sharey=True, gridspec_kw={'wspace': 0})
# x1 = 1
# x2 = len(PMD_Cov_g)
# z1 = 1
# z2 = len(PMD_Cov_g)
# im0 = axs[0,0].imshow(PMD_Cov_g,interpolation='none',
#        extent=(x1,x2,z2,z1), aspect=(x2-x1)/(z2-z1), cmap='jet')
# im1 = axs[0,1].imshow(PMD_Cov_T,interpolation='none',
#        extent=(x1,x2,z2,z1), aspect=(x2-x1)/(z2-z1), cmap='jet')

# im2 = axs[1,0].plot(PMD_Cov_g[15,:],'k-',linewidth=2)
# #axe[1,0].set(xlabel='lag', ylabel='Covariance ($\mathregular{mGal^{2}}$)')

# im3 = axs[1,1].plot(PMD_Cov_T[15,:],'k-',linewidth=2)
# #axe[1,1].set(xlabel='lag', ylabel='Covariance ($\mathregular{nT^{2}}$)')

# # axs[0].title.set_text('(a)')
# # axs[1].title.set_text('(b)')
# # axs[0].set(xlabel='lag', ylabel='lag')
# # axs[1].set(xlabel='lag')
# ### Hide x labels and tick labels for all but bottom plot.
# for ax in axs:
#     ax.label_outer()
# cbar0 = fig.colorbar(im0, ax=axs, shrink=0.5, label='Covariance ($\mathregular{mGal^{2}}$)')
# cbar1 = fig.colorbar(im1, ax=axs, shrink=0.5, label='Covariance ($\mathregular{nT^{2}}$)')
# plt.show()
# figname = 'PMD_Cov'
# fig.savefig(fpath_plots+'/'+figname+'.pdf')
# plt.close(fig)    # close the figure window

##############################################################################################
## Plot PMD
  
# fig = plt.figure()
# x1 = dis_min/1000
# x2 = dis_max/1000
# z1 = z_min/1000
# z2 = z_max/1000
# Xticklabels = np.linspace(x1,x2,5)
# Zticklabels = np.linspace(z2,z1,5)

# plt.imshow(PMD_density,interpolation='none',
#        vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect=(x2-x1)/(z2-z1))
# plt.xticks(Xticklabels)
# plt.yticks(Zticklabels)

# plt.xlabel("X Profile (km)")
# plt.ylabel("Depth (km)")

# cbar = plt.colorbar()
# plt.set_cmap('bwr')
# cbar.ax.set_ylabel('density contrast ($\mathregular{g/cm^{3}}$)')
# plt.show()
# figname = 'PMD_Density'
# fig.savefig(fpath_plots+'/'+figname+'.pdf')
# plt.close(fig)    # close the figure window

# #########################################################################################
# ## Plot True Model
  
# fig = plt.figure()
# x1 = dis_min/1000
# x2 = dis_max/1000
# z1 = z_min/1000
# z2 = z_max/1000
# Xticklabels = np.linspace(x1,x2,5)
# Zticklabels = np.linspace(z2,z1,5)

# plt.imshow(TrueDensityModel,interpolation='none',
#        vmin=rho_salt_min, vmax=rho_base_max, extent=(x1,x2,z2,z1), aspect=(x2-x1)/(z2-z1))
# plt.xticks(Xticklabels)
# plt.yticks(Zticklabels)

# plt.xlabel("X Profile (km)")
# plt.ylabel("Depth (km)")

# cbar = plt.colorbar()
# plt.set_cmap('bwr')
# cbar.ax.set_ylabel('density contrast ($\mathregular{g/cm^{3}}$)')
# plt.show()
# figname = 'True_Model'
# fig.savefig(fpath_plots+'/'+figname+'.pdf')
# plt.close(fig)    # close the figure window



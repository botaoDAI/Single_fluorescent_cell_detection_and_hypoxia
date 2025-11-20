# =============== DESCRIPTION =========================================================================================
#
# This file fits our model of cell growth IN RESPONSE TO RADIATION to our experiments for a given initial cell density
# and display the result. An explicit forward Runge-Kutta 4-formulation with a time step dt was used for the resolution.
#
# =============== REQUIRED PACKAGES =========================================================================================

import numpy as np # to use arrays
import matplotlib.pyplot as plt # to display fitting result
from scipy.optimize import curve_fit # to perform the non linear fit
import random # to use random initial parameter values
import time
from scipy.integrate import solve_ivp

# =============== FUNCTIONS =========================================================================================

# Function computing the chi2 value of a model. Here, no std values are taking into account in the computation
# because we do not possess "true" error bars in our case.
# Arguments:
#   - exp: numpy array of experimental values (floats)
#   - fit: numpy array of fit/model values (floats)
# Output:
#   - chi2: float corresponding to the chi2 value of the model
#   - chi2: float corresponding to the chi2 value of the model
def Chi2(exp,fit,std,k):
    n=np.size(exp)
    # print(n)
    # print(np.size(fit))
    # print(np.size(std))
    chi2=(1/(n-k))*np.sum(((exp-fit)/std)**2)
    return chi2

# Function that determines the best set of model parameters by performing fits on several random initial 
# sets of model parameters and comparing their respective chi2
# Arguments:
#   - model: numpy array of model values (floats)
#   - time: numpy array of time values (floats)
#   - exp: numpy array of experimental values (floats)
#   - bounds: tuple of lists (a,b) where a corresponds to the list of the lower bounds of the set of parameters
#             and b corresponds to the list of the upper bounds of the set of parameters
#   - ndraws: int corresponding to the total number of random draws for the initial values of model parameters
#             from whoich a fit is performed
#   - method: string designating the type of method wished for choosing the initial set of parameters
# Output:
#   - chi2: float corresponding to the chi2 value of the model
#   - best_popt: float corresponding to the chi2 value of the model
def Find_best_popt(model,time,exp, std, k,bounds,ndraws,method):
    npara=len(bounds[0]) # we retrieve the total number of parameters
    chi2=10000000 # we initialize the chi2 at a very high value
    best_popt=np.zeros(npara) 
    # we loop on the number of random draws
    for j in range(ndraws):
        if j%100==0:
            print("Draw ",j) # to evaluate the progress of the algorithm
        p0=np.zeros(npara) 
        if method=="random":
            # we initialize the initial set of parameters by choosing random values for each parameter within its bounds
            for i in range(npara):
                b0=bounds[0][i]
                b1=bounds[1][i]
                p0[i]=np.random.uniform(b0,b1)
        elif method=="randrange": # here we only choose values at regular interval within the bounds
            p01=random.randrange(2,7,2)*0.01  # lambda_s
            p02=random.randrange(1,10,2)*0.0001 # lambda_r
            p03=random.randrange(1,30,2)       # Ta
            p04=random.randrange(1,15,2)*0.01  # lambda_u
            p0=np.array([p01,p02,p03,p04])
        # we perform the fit with this initial set of parameters
        popt,pcov=curve_fit(model,time,exp,maxfev=200000000,p0=p0,bounds=bounds)
        fit=model(time,*popt)
        # then we compute its chi2 value
        chi2_draw=Chi2(exp, fit,std,k)
        # if this chi2 value if lower than the one saved before then we replace it
        # and we also keep iun memory the best set of parameters associated to the best chi2 
        if chi2_draw <= chi2:
            chi2=chi2_draw
            best_popt=popt
    return chi2,best_popt


# =============== SETTINGS =========================================================================================

# We fit here our model in the case of only one initial cell density. The latter is determined by the numeration of 
# the well considered in the experiment, the lowest cell density corresponding to the first well. Hence, we specify the 
# numeration of the well we are interested in
num_well=6
dose=20
# as well as the path where are saved the mean cell densities values at each time point with their standard deviation values in a .txt file
#path_exp="/Users/billoir/Desktop/Fichiers Thèse/Projet 1 Irradiations BB/Expériences/Données expérimentales/Averaged_curves/"+str(dose)+"_Gy_Well"+str(num_well)+"_3exps.txt"
path_exp="/Users/billoir/Desktop/Fichiers Thèse/Projet 1 Irradiations BB/Expériences/Données expérimentales/Well"+str(num_well)+"_Incucyte_F98_20Gy_2024_09_24_smooth=35.txt"
# we precise the interval of time in HOURS between two cell density values
interval=1.5
# we indicate how many random draws we wish to perform
ndraws=int(1e4)
# and the methods used to do the draws
method="random"
# we also specify the frame to which we wish to stop
last_frame=130
# Finally, we give two list of colors: one for the mean cell densities values and another one for the error bars. The numeration of the well
# will correspond to the color associated to its corresponding initial cell density. 
colors_av=["#4B0082","#000080","#008080","#006400","r","#800000"]
colors_std=["#DDA0DD","#00BFFF","#7FFFD4","#ADFF2F","#FFD700","#F08080"]

# =============== INITIALIZATION =========================================================================================

# we retrieve the experimental values of the inital cell density as well as the standard deviation values
mean_celldensity=np.loadtxt(path_exp,usecols=0)[0:last_frame]
std_celldensity=0.2*mean_celldensity            #np.loadtxt(path_exp,usecols=1)
# we stock the value of the initial cell density in the variable C0 which will be used in the model
C0=mean_celldensity[0]
# we deduce the total number of frames in the experiment
nframes=np.size(mean_celldensity)
# we define the array of experimental time values
time_points=(np.arange(nframes))*interval

ind_cut=np.where(mean_celldensity<1.1e-3)[0][-1]
mean_celldensity=mean_celldensity[0:ind_cut+1]
std_celldensity=std_celldensity[0:ind_cut+1]
time_points=time_points[0:ind_cut+1]

# Fixed parameters
k0=0.045
gamma=0.06
Cmax=2.4e-3
k=3
tr=0
ts=0.02
tu=0.08
if dose==20:
    Ta=23
if dose==15:
    Ta=21
if dose==10 or dose==12.5:
    Ta=18
if dose==5 or dose==7.5:
    Ta=16.5
if dose==0:
    Ta=0

# Bounds of the unfixed parameters ts, tr, tu and Cmax
bounds=([0,0],[0.06,0.2])
#bounds=([0],[0.2])

# =============== MODEL =========================================================================================

def equa_diff(t,y,ts,tu):
    Cd,Cr,Cs,Cu,C=y
    kd=tr+tu-gamma
    if t<Ta:
        dCd=(kd/Ta)*t*C*(1-C/Cmax)
        dCu=0
        dCs=0
        dCr=0
        dC=dCd+dCr+dCu+dCs
    if t>= Ta:
        dCd=kd*Cd*(1-C/Cmax)-(tu+tr)*Cd
        dCu=(tu*Cd-gamma*Cu)
        dCs=ts*Cu
        dCr=(k0*(1-C/Cmax)*Cr+tr*Cd)
        dC=dCd+dCr+dCu+dCs
    return [dCd,dCr,dCs,dCu,dC]

def model(t,ts,tu):
    sol=solve_ivp(equa_diff,[time_points[0],time_points[-1]],[C0,0,0,0,C0],t_eval=time_points,args=[ts,tu],method='DOP853')
    return sol.y[4]

def subpopulations(t,ts,tu):
    sol=solve_ivp(equa_diff,[time_points[0],time_points[-1]],[C0,0,0,0,C0],t_eval=time_points,args=[ts,tu],method='DOP853')
    return sol.y[0],sol.y[1],sol.y[2],sol.y[3]



# =============== FITTING =========================================================================================

#### 1- NUMERICAL RESOLUTION AND FIT ######################################
start=time.time()
chi2,popt=Find_best_popt(model, time_points, mean_celldensity, std_celldensity,k,bounds, ndraws,method)
end=time.time()
print("Time taken = ", (end-start)/60)

#popt=[tr,ts,tu]
#popt=np.asarray([0, 1.9e-02, 	0.8e-01])
#popt=np.asarray([3.30841570e-06,1.56032614e-02,1.11340108e-01])
density=model(time_points[0:last_frame],*popt)
    
#### 2- FIGURE ############################################################
    
plt.rcParams["font.family"]="serif"
plt.figure(figsize=(35,10))
value=1e3
# Experiment
plt.errorbar(time_points[0:last_frame],mean_celldensity[0:last_frame]*value,xerr=None, yerr=std_celldensity[0:last_frame]*value,color=colors_av[num_well-1], ecolor=colors_std[num_well-1],fmt="o",
        markersize=6,linewidth=5,label="Experiment")
# Fit
plt.plot(time_points[0:last_frame],density*value,
      label=r'$C=C_d+C_{s}+C_u+C_r$', c='b',
      linewidth=13)

# Evolution of the compartments
index=np.where(time_points>=Ta)[0][0]
Cd,Cr,Cs,Cu=subpopulations(time_points,*popt)
plt.plot(time_points[index:last_frame],Cs[index:last_frame]*value,
      label=r'$C_{s}$', c='#FF8C00',
      linewidth=13)
plt.plot(time_points[index:last_frame],Cu[index:last_frame]*value,
      label=r'$C_{u}$', c='r',
      linewidth=13)
plt.plot(time_points[index:last_frame],Cr[index:last_frame]*value,
      label=r'$C_{r}$', c='g',
      linewidth=13)
plt.plot(time_points[index:last_frame],Cd[index:last_frame]*value,
      label=r'$C_{d}$', c='k',
      linewidth=13)

plt.xlabel("Temps ",fontsize=50,fontweight="bold")
plt.ylabel("Densité cellulaire ",fontsize=50,fontweight="bold")
plt.tick_params(axis='both',labelsize=40,width=4)
# plt.title(r'Well ' +str(num_well)+": "
#           + r', $\lambda_{s}=$ '+ str((np.round(popt[0],6)))
#           +"\n"+ r' $\lambda_{r}=$ '+ str((np.round(popt[1],8)))
#             #+r', $T=$ '+ str((np.round(popt[2],6)))
#             +"\n"+r' $\lambda_{u}=$ '+ str((np.round(popt[2],6)))
#             ,fontsize=40,fontweight='bold')
plt.legend(loc='lower right', bbox_to_anchor=(1.7, 0.46),fontsize=50,markerscale=3)
plt.tight_layout()
plt.show()

print(popt)


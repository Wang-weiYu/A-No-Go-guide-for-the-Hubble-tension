"""
 ____   _              
|  _ \ / \   __ _  ___ 
| |_) / _ \ / _` |/ _ \
|  __/ ___ \ (_| |  __/
|_| /_/   \_\__, |\___|
            |___/      
author: Rong-Gen Cai, Zong-Kuan Guo, Shao-Jiang Wang, Wang-Wei Yu, Yong Zhou
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy import constants as C
from scipy.optimize import fsolve
import emcee
import os
from multiprocessing import Pool
os.environ["OMP_NUM_THREADS"] = "1"



def zofH0t(H0t,z0,p,eta):
    return -z0-1+np.exp(-(H0t-p)*(3*p**2+H0t*eta-p*(2+eta))/(3*p**2))*(p/H0t)**(2/3)

def E(z0,H0,p,eta):
    rootH0t = fsolve(zofH0t,p/1e6,args=(z0,p,eta,))
    return 1/(H0*(1+(2/3)*(1-eta*rootH0t/p)*(1/rootH0t-1/p)))

def Hz(z0,H0,p,eta):
    rootH0t = fsolve(zofH0t,p/1e6,args=(z0,p,eta,))
    return H0*(1+(2/3)*(1-eta*rootH0t/p)*(1/rootH0t-1/p))

def DL(zhel,zcmb,H0,p,eta):
    v,err = integrate.quad(E,0.0,zcmb,args=(H0,p,eta,))
    return (1+zhel)*v*C.c/(10**3)

def muth(zhel,zcmb,H0,p,eta):
    return 5*np.log10(DL(zhel,zcmb,H0,p,eta))+25

def mth(zhel,zcmb,H0,p,eta,M):
    return muth(zhel,zcmb,H0,p,eta)+M

# SN Data
zcmb,zhel,mb,dmb = np.genfromtxt('./data/lcparam_full_long_zhel.txt', skip_header=1, delimiter=' ', dtype=None,usecols = (1,2,4,5),unpack=True)

covdata = np.loadtxt('./data/sys_full_long.txt')
covariance = covdata.reshape(1048,1048)
merr=covariance+np.diag(dmb**2)
inv=np.linalg.inv(merr)

# BAO Data
zbaow = np.array([0.31,0.31,0.36,0.36,0.40,0.40,0.44,0.44,0.48,0.48,0.52,0.52,0.56,0.56,0.59,0.59,0.64,0.64])
baoobw =np.array([6.29,11550,7.09,11810,7.70,12120,8.20,12530,8.64,12970,8.90,13940,9.16,13790,9.45,14550,9.62,14600])
invbaow = np.loadtxt('./data/tomographic_BAO_invcov.txt')

zbaoh = np.array([0.106,0.15,1.52,2.34,2.34,2.35,2.35,0.845,0.85,0.85,2.33,2.33,0.835])
msurh = np.array([0.336,4.47,26.00,8.86,37.41,9.20,36.3,18.33,19.6,19.5,8.99,37.5,18.92])
sh = np.array([0.015,0.17,0.99,0.29,1.86,0.36,1.8,0.595,2.15,1.0,0.19,1.1,0.51])
ish = np.diag(sh**2)
invsh = np.linalg.inv(ish)

zbao2 = np.array([0.698,0.698,1.48,1.48])
baoob2 = np.array([17.646,19.770,30.21,13.23])
covbao2 = np.loadtxt('./data/bao698.txt')
invbao2 = np.linalg.inv(covbao2)

# HOD Data
# BC03
# zhod =np.array([0.3802,0.4004,0.4247,0.4497,0.4783,0.4293,0.1791,0.1993,0.3519,0.5929,0.6797,0.7812,0.8754,1.037,0.07,0.12,0.20,0.28,0.47,0.1,0.17,0.27,0.4,0.48,0.88,0.9,1.3,1.43,1.53,1.75])
# Hhod =np.array([83.0,77.0,87.1,92.8,80.9,85.7,75,75,83,104,92,105,125,154,69.0,68.6,72.9,88.8,89,69,83,77,95,97,90,117,168,177,140,202])
# shod = np.array([13.5,10.2,11.2,12.9,9,5.2,4,5,14,13,8,12,17,20,19.6,26.2,29.6,36.6,23,12,8,14,17,62,40,23,17,18,14,40])
# ihod = np.diag(shod**2)
# invhod=np.linalg.inv(ihod)

# MS11 Data
zhod =np.array([0.3802,0.4004,0.4247,0.4497,0.4783,0.4293,0.1791,0.1993,0.3519,0.5929,0.6797,0.7812,0.8754,1.037])
Hhod =np.array([89.3,82.8,93.7,99.7,86.6,91.8,81,81,88,110,98,88,124,113])
shod = np.array([14.1,10.6,11.7,13.4,8.7,5.3,5,6,16,15,10,11,17,15])
ihod = np.diag(shod**2)
invhod=np.linalg.inv(ihod)

# Initialization
deltamu=np.zeros(np.size(zcmb))
deltabaoh=np.zeros(np.size(zbaoh))
deltahod=np.zeros(np.size(zhod))
deltabaow=np.zeros(np.size(zbaow))
deltabao2=np.zeros(np.size(zbao2))

pr1=np.array([0.15,2.0])
pr2=np.array([-2,2])
pr3=np.array([60.0,80.0])
pr4=np.array([-20,-19])
pr5=np.array([120,160])


def log_probability(theta):
    p, eta, H0, M ,rd= theta
    if pr1[0] < p < pr1[1] and pr2[0] < eta < pr2[1] and pr3[0] < H0 < pr3[1] and pr4[0] < M < pr4[1] and pr5[0] < rd < pr5[1]:
        deltamu=[mb[i]-mth(zhel[i],zcmb[i],H0,p,eta,M) for i in range(np.size(zcmb))]
        for i in range(np.size(zhod)):
            deltahod[i]=Hhod[i]-Hz(zhod[i],H0,p,eta) 
        for i in range(0,18,2):
            deltabaow[i]=baoobw[i]-DL(zbaow[i],zbaow[i],H0,p,eta)/((1+zbaow[i])**2*rd)
        for i in range(1,18,2):
            deltabaow[i]=baoobw[i]-Hz(zbaow[i],H0,p,eta)*rd


        deltabaoh[0]=msurh[0]-rd/(C.c/10**3*DL(zbaoh[0],zbaoh[0],H0,p,eta)**2*zbaoh[0]/((1+zbaoh[0])**2*Hz(zbaoh[0],H0,p,eta)))**(1/3)
        for i in np.array([1,2,7]):
            deltabaoh[i]=msurh[i]-(C.c/10**3*DL(zbaoh[i],zbaoh[i],H0,p,eta)**2*zbaoh[i]/((1+zbaoh[i])**2*Hz(zbaoh[i],H0,p,eta)))**(1/3)/rd
        for i in np.array([4,6,9,11,12]): #DM
            deltabaoh[i] = msurh[i] - DL(zbaoh[i],zbaoh[i],H0,p,eta)/((1+zbaoh[i])*rd)
        for i in np.array([3,5,8,10]): #DH
            deltabaoh[i] = msurh[i] - C.c/(Hz(zbaoh[i],H0,p,eta)*rd*(10**3))
        
        for i in range(0,4,2):
            deltabao2[i] = baoob2[i] - DL(zbao2[i],zbao2[i],H0,p,eta)/((1+zbao2[i])*rd)
        for i in range(1,4,2):
            deltabao2[i] = baoob2[i] - C.c/(Hz(zbao2[i],H0,p,eta)*rd*(10**3))
        log_likelihood=-0.5 * np.dot(deltamu, np.dot(deltamu,inv))-0.5 * np.dot(deltabaow,np.dot(deltabaow,invbaow))-0.5 * np.dot(deltabaoh,np.dot(deltabaoh,invsh))-0.5 * np.dot(deltabao2,np.dot(deltabao2,invbao2))-0.5 * np.dot(deltahod,np.dot(deltahod,invhod))

        return log_likelihood
    return -np.inf

ndim = 5
nwalkers = 10
pos1=pr1[0]+(pr1[1]-pr1[0])*np.random.rand(nwalkers, 1)
pos2=pr2[0]+(pr2[1]-pr2[0])*np.random.rand(nwalkers, 1)
pos3=pr3[0]+(pr3[1]-pr3[0])*np.random.rand(nwalkers, 1)
pos4=pr4[0]+(pr4[1]-pr4[0])*np.random.rand(nwalkers, 1)
pos5=pr5[0]+(pr5[1]-pr5[0])*np.random.rand(nwalkers, 1)
pos = np.hstack((pos1,pos2,pos3,pos4,pos5))

filename = "./chains/filename.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)


if __name__ ==  '__main__': 

    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, pool=pool,backend=backend)
        sampler.run_mcmc(pos, 2000, progress=True);

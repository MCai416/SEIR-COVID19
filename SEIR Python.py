# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 16:06:14 2020

@author: Ming Cai
"""

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import scipy.interpolate as itp

font = {'family':'DejaVu Sans',
        'weight':'normal',
        'size'   : 30}

mpl.rc('font', **font)

"""
This model tries to model the spread of COVID-19 
With lockdown as an exogenous shock to reduce the transmission and death rate

parameters calibrated
1) effectiveness of lockdown 
2) transmission rate "bE"

Features 
1) Virus can be transmitted in incubation stage, and patients with symptoms will be isolated
immediately, meaning that it hardly infect anyone 
2) Incubation stage patients can heal as well, with the same rate as patients with symptoms
3) Some people can die, but dead people will not transmit virus 

Findings 
1) If uncontrolled, eventually everyone will be infected
and 1% of the population will die!
2) China's tough measure reduces the transmission and death rate by 96%! 
This is highly remarkable since there has been no vaccines 
3) On Jan 1, approximately 8 people are infected with unknown disease, 23 days later
with approximately 1000 infections, China locked down cities to prevent further transmission
which eventually slowed down the whole process 
4) Although the WHO suggested a 3.4% death rate, this model suggested a 2% death rate 
5) This model also suggested a 18 day incubation period before patients show any symptoms 
6) Despite patients with symptoms will be isolated, many will be infected if no measures are taken

Limitations 
1) Discrete approximations (when uncontrolled, the susceptible population went negative)
2) Death rate is still climbing 180 days after breakout, 
although the growth of death toll is much slower now in China 
3) Does not feature tests, i.e. incubation period people are get tested, and they are only tested
once they have symptoms 
4) Parameter of rate of transmission should decrease over time as people realise the problem 
In this model this parameter is constant, only subject to an external shock 
5) No lags, no suppression of announcement
if people have symptoms then they are tested immediately
6) It is still not known whether if people are immune to the virus once they caught the disease
here we assumed true 

"""

# Model Params 

lockdown = 0 #dummy, 1 implies lockdown 
T = 1000 # iteration period

#Population 
N0 = 1368000000 #Population 
#UK 66.44 million 
#China: 1.368 billion

#death rates 
muS = 0 # susceptible death rate COVID-19: N/A
muE = 0 # exposed death rate COVID-19: N/A
muI = 0.018 # infected death rate WHO: 3.2%, China: 2%
muR = 0 # resistant death rate COVID-19: no figure available 

# Recovery 
nu = 0 #COVID-19: N/A suscepted becomes immune 
gamma = 1/14 # average days of recovery, China: 14 days 

# symptom rate 
sigma = 1/18 # average days of incubation period to exposed, China: 18 days

# infection rate 
bE0 = 3 # rate of transmission by patients over the incubation period, China: (calibrated) 3
bE1 = 0.5
eta = 0.0011

bE = np.zeros(T)
for i in range(T):
    bE[i] = bE0*np.exp(-eta*i)+bE1*(1-np.exp(-eta*i))

bI = 0.01 # rate of transmission by patients with symptoms, China: 0.01 
# those who are tested positive would be isolated 
# and has a low chance of infecting susceptible 

delta = 0.0001

#initial number of infections 
ini = 8 #China on early January 2020: 8

# lock down threshold (the bE rates drop after # infected reaches this number) 
c = 1000 # China: approx 1000
ef1 = 28 #lockdown effectiveness on transmission China: (calibrated) 28

t = np.arange(0, T)

N = np.zeros(T)
S = np.zeros(T) # susceptible = people who are not infected 
E = np.zeros(T) # exposed = incubation period 
I = np.zeros(T) # infected 
R = np.zeros(T) # recovered 

# initial condition 

N[0] = N0
E[0] = ini/N0
S[0] = N0 - I[0]

# model calculation

for i in range(1,T):
    dS = muS*(N[i-1]-S[i-1])-S[i-1]/N[i-1]*(bE[i-1]*E[i-1]+bI*I[i-1])-nu*S[i-1]
    S[i] = S[i-1] + dS
    dE = S[i-1]/N[i-1]*(bE[i-1]*E[i-1]+bI*I[i-1]) - (muE + sigma)*E[i-1] - gamma*E[i-1] + delta*R[i-1]/N[i-1]*(bE[i-1]*E[i-1]+bI*I[i-1])
    E[i] = E[i-1] + dE
    dI = sigma*E[i-1] - (muI+gamma)*I[i-1]
    I[i] = I[i-1] + dI
    dR = gamma*I[i-1] - muR*R[i-1] + gamma*E[i-1] - delta*R[i-1]/N[i-1]*(bE[i-1]*E[i-1]+bI*I[i-1])
    R[i] = R[i-1] + dR 
    N[i] = S[i] + E[i] + I[i] + R[i]
    if I[i] >= c and lockdown == 0: 
        bE[i:T] = bE[i:T]/ef1
        muI = muI/ef1
        lockdown = 1
        print("lockdown triggered at day: %d"%(i))

# some interesting paths 
D = N0 - N # death toll 

vecdI = I[1:T] - I[0:T-1] # change of infections daily 
vecdR = R[1:T] - R[0:T-1] # 
vecdD = D[1:T] - D[0:T-1]

#plot 

fig, axs = plt.subplots(2, 2)
grid = plt.GridSpec(2, 2)
ax1 = plt.subplot(grid[:, 1])

ax1.plot(t, E, color = 'orange', linewidth = 5, label = 'Patients in incubation period')
ax1.plot(t, I, color = 'red', linewidth = 5, label = 'Patients infected with symptoms')
ax1.plot(t, D, color = 'black', linewidth = 5, label = 'Deceased')
ax1.set_xlim([0, 180])
ax1.grid()
ax1.set_title('Incubated, Infected with Symptoms, Deceased')
ax1.legend()

axs[1, 0].plot(t[0:T-1], vecdI, color = 'red', linewidth = 5, label = 'Infected')
axs[1, 0].plot(t[0:T-1], vecdD, color = 'black', linewidth = 5, label = 'Deceased')
axs[1, 0].plot(t[0:T-1], vecdR, color = 'green', linewidth = 5, label = 'Recovered')
axs[1, 0].set_xlim([0, 180])
axs[1, 0].grid()
axs[1, 0].set_title('Changes')
axs[1, 0].legend()

axs[0, 0].set_xlim([0, 180])
axs[0, 0].grid()
axs[0, 0].set_title('Population aggregates')
axs[0, 0].plot(t, N, color = 'blue', linewidth = 5, label = 'Population')
axs[0, 0].plot(t, S, color = 'purple', linewidth = 5, label = 'Uninfected')
axs[0, 0].plot(t, R, color = 'green', linewidth = 5, label = 'Recovered')
axs[0, 0].legend()

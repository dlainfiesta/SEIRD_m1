# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:02:00 2020

@author: diego
"""

#%% Importing packages

from scipy.integrate import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import os

#%% Setting paths

print(os.getcwd())
path= 'C:\\Users\\diego\\Desktop\\DL\\Covid_Guatemala\\SEIRD_Guatemala_m1_complete\\'
os.chdir(path)
print(os.getcwd())

path_input= path+'01_input\\'
path_code= path+'02_code\\'
path_output= path+'03_output\\'

#%% Defining equations

def deriv(y, t, N, beta, gamma, delta, alpha, rho):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - delta * E
    dIdt = delta * E - (1 - alpha) * gamma * I - alpha * rho * I
    dRdt = (1 - alpha) * gamma * I
    dDdt = alpha * rho * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


N = 17_915_567
D = 6.0 # infection-recovered lasts six days
gamma = 1.0 / D
delta = 1.0 / 6.0  # incubation period of six days
R_0 = 2.65
beta = R_0 * gamma  # R_0 = beta / gamma, so beta = R_0 * gamma
alpha = 0.01119  # 1% death rate
rho = 1/7.7  # 7.7 days from infection until death
S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0  # initial conditions: one exposed

t = np.linspace(0, 365, 365) # Grid of time points (in days)
y0 = S0, E0, I0, R0, D0 # Initial conditions vector

#  Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma, delta, alpha, rho))
S, E, I, R, D = ret.T

#%% Loading data Guatemala

# Loading data from Our World in Data
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(url, index_col=0, parse_dates=[0], usecols=['location' ,'date', 'new_cases', 'new_deaths'])

country= 'Guatemala'

guatemala_raw= df[df.index==country]
guatemala_raw['date']= pd.to_datetime(guatemala_raw['date'], format='%Y-%m-%d')
guatemala_raw= guatemala_raw.set_index('date', drop= 'True')

# Loading file from MSAPS
mspas_file= pd.read_excel(path_input+'22062020_MSPAS_info.xlsx')
mspas_file['date']= pd.to_datetime(mspas_file['Fecha'])
mspas_file['infectados_activos']= mspas_file['Casos por día'].cumsum()-mspas_file['Casos recuperados'].cumsum()-mspas_file['Casos fallecidos'].cumsum()
mspas_file= mspas_file.set_index('date', drop= True)

guatemala_raw= pd.concat([mspas_file['infectados_activos'],mspas_file['Casos fallecidos'], guatemala_raw], axis= 1)

start= datetime.strptime('2020-03-03', '%Y-%m-%d') # Making the cases
dates= pd.date_range(start, end= guatemala_raw.index[-1])

guatemala= pd.DataFrame({'index': dates})
guatemala= guatemala.set_index('index', drop= True)

guatemala= pd.concat([guatemala, guatemala_raw], axis= 1)

end= guatemala.shape[0]
guatemala['new_cases_SEIRD']=I[0:end]
guatemala['new_deaths_SEIRD']=D[0:end]

#%%  Plotting SRIED-overall

def plotseird(t, S, E, I, R, D, title, end):
    
  f, ax = plt.subplots(1,1,figsize=(12,8))
  
  plt.title(title, size= 14)
  
  ax.plot(t, S, 'b', alpha=0.7, linewidth=1.5, label='Susceptibles')
  ax.plot(t, E, 'gold', alpha=0.7, linewidth=1.5, label='Expuestos')
  ax.plot(t, I, 'r', alpha=0.7, linewidth=1.5, label='Infectados')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=1.5, label='Recuperados')
  ax.plot(t, D, 'k', alpha=0.7, linewidth=1.5, label='Fallecidos')
  ax.plot(t, S+E+I+R+D, 'c--', alpha=0.5, linewidth=2, label='Total')
  
  plt.annotate('Número total de muertes: '+str(int(round(np.diff(D).sum(),0))), (0, 16_500_000), size= 10, weight='bold')
  plt.annotate('Número total de infectados: '+str(int(round(np.diff(D).sum()+np.diff(R).sum(),0))), (0,15_000_000), size= 10, weight='bold')
  plt.annotate('Número total de recuperados: '+str(int(round(np.diff(R).sum(),0))), (0,13_500_000), size= 10, weight='bold')
  plt.annotate('Ratio infectados/población: '+str(round((np.diff(D).sum()+np.diff(R).sum())/N,2)), (0,12_000_000), size= 10, weight='bold')
  
  plt.axvline(x= end, label='Día de pandemia', color= 'deeppink', linestyle= '--')
  plt.arrow(end, 4_000_000, 10, 0, head_width = 600_000, head_length= 5, ec= 'deeppink')
  plt.annotate('Hoy ', (end+10, 3_000_000), size= 10, color= 'deeppink', weight='bold')

  ax.set_ylabel('Variables en el día t')
  ax.set_xlabel('Tiempo (días pandemia)')

  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
  plt.grid(color= 'lightgrey', )
  
  plt.savefig(path_output+'SEIRD_completo.jpg', dpi= 500, bbox_inches='tight')
  plt.show()

# Plotting overall 

title= 'Modelo SEIRD/Guatemala pandemia total \n actualizada al - '+datetime.strftime(guatemala.index[-1], '%d/%m/%Y')
plotseird(t, S, E, I, R, D, title, end)

#%% Plotting for today

def plotseird_t(t, E, I, R, D, title, start, end, advance):
    
  f, ax = plt.subplots(1,1,figsize=(12,8))
  
  plt.title(title, size= 14)
  
  ax.plot(t, E, 'gold', alpha=0.5, linewidth=1.5, label='Expuestos')
  ax.plot(t, I, 'r', alpha=0.7, linewidth=1.5, label='Infectados')
  ax.plot(t, D, 'k', alpha=0.7, linewidth=1.5, label='Fallecidos')
  
  plt.axvline(x= end, label='Día de pandemia', color= 'deeppink', linestyle= '--')
  plt.arrow(end, 23000, 2, 0, head_width = 5000, head_length= 2, ec= 'deeppink')
  plt.annotate('Hoy ', (end+2, 17500), size= 10, color= 'deeppink', weight='bold')
  
  ax.set_ylabel('Variables(Tiempo)')
  ax.set_xlabel('Tiempo (días pandemia)')

  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=1, ls='-')
  
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
  plt.grid(color= 'lightgrey', )
  plt.yticks(np.arange(start, max(E), 10000))
  plt.xticks(np.arange(start, end+advance, 5))
  
  """
  plt.annotate('Número total de muertes: '+str(int(round(np.diff(D).sum(),0))), (0, 16_500_000), size= 10, weight='bold')
  plt.annotate('Número total de infectados: '+str(int(round(np.diff(D).sum()+np.diff(R).sum(),0))), (0,15_000_000), size= 10, weight='bold')
  plt.annotate('Número total de recuperados: '+str(int(round(np.diff(R).sum(),0))), (0,13_500_000), size= 10, weight='bold')
  plt.annotate('Ratio infectados/población: '+str(round((np.diff(D).sum()+np.diff(R).sum())/N,2)), (0,12_000_000), size= 10, weight='bold')
  """
  
#  plt.savefig(path_output+'SEIRD_al_dia.jpg', dpi= 500, bbox_inches='tight')
  plt.show()

# Plotting

title= 'Modelo SEIRD/Guatemala pandemia al - '+datetime.strftime(guatemala.index[-1], '%d/%m/%Y')

start= 50
end= end
advance= 5

plotseird_t(t[start:end+advance], E[start:end+advance], I[start:end+advance], \
            R[start:end+advance], D[start:end+advance], title, start, end, advance)

#%% Plotting new cases

fig, ax= plt.subplots(figsize= (12,8))

ax.plot(guatemala['infectados_activos'], label= 'Reportado gobierno - Infectados activos',marker= 'x', markersize= 4, alpha=0.7, color= 'r')
ax.plot(guatemala['new_cases_SEIRD'], label= 'Predicción modelo SEIRD - Infectados activos', alpha=0.7, color= 'r')

# Set title and labels for axes
ax.set(xlabel='Tiempo (Fecha)',
       ylabel='Infectados activos',
       title='Nuevos casos reportado MSPAS y resultado modelo SEIRD')

# Define the date format
date_form = DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.xticks(rotation= 45)
plt.legend(loc= 0)
plt.grid()
plt.savefig(path_output+'infected_SEIRD.jpg', dpi= 500, bbox_inches='tight')
plt.show()

#%% Plotting new deaths

fig, ax= plt.subplots(figsize= (12,8))
ax.plot(guatemala['new_deaths'].cumsum(), label= 'Datos reportados MSPAS, suma: '+str(guatemala['new_deaths'].sum()), marker= 'x', markersize= 4, alpha=0.7, color= 'k')
ax.plot(guatemala['new_deaths_SEIRD'], label= 'Predicción modelo SEIRD, suma: '+str(round(guatemala['new_deaths_SEIRD'].diff().sum(),0)), alpha=0.7, color= 'k')

# Set title and labels for axes
ax.set(xlabel='Tiempo (Fecha)',
       ylabel='Muertes acumuladas',
       title='Muertes/día reportado MSPAS y resultado modelo SEIRD')

# Define the date format
date_form = DateFormatter("%d-%m-%Y")
ax.xaxis.set_major_formatter(date_form)

# Ensure a major tick for each week using (interval=1) 
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))

plt.xticks(rotation= 45)
plt.legend(loc= 0)
plt.grid()
plt.savefig(path_output+'new_death_reported_SEIRD.jpg', dpi= 500, bbox_inches='tight')
plt.show()



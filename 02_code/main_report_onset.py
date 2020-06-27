# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 06:56:03 2020

@author: diego
"""


import os

print(os.getcwd())
path= 'C:\\Users\\diego\\Desktop\\DL\\Covid_Guatemala\\filtracion_MSAPS\\'
os.chdir(path)
print(os.getcwd())

path_input= path+'01_input\\'
path_code= path+'02_code\\'

path_output= path+'03_output\\'

path_output_dep= path+'03_output\\dep\\'
path_output_mun= path+'03_output\\mun\\'

#%% Importing packages

PYTHONPATH=path_code+":$PYTHONPATH"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import unidecode
import urllib.request
import tarfile

from datetime import datetime

# Importing functions

import sys
sys.path.append(path_code)

from functions import delay_distribution
from functions import confirmed_to_onset
from functions import adjust_onset_for_right_censorship

#%% Importing dataset from MSPAS

# Importing information about cities Guatemala

df = pd.read_json (path_input+'gt.json', encoding='latin-1')
df= df[['city', 'lat', 'lng']].reset_index(drop= True)
df['city']= df['city'].apply(lambda x: unidecode.unidecode(x).upper())

lat_dic= dict(zip(df['city'], df['lat']))
lng_dic= dict(zip(df['city'], df['lng']))

# Informacion filtrada por el MSPAS 14/06/2020

leaked_data= pd.read_excel(path_input+'DatosCOVID_14JUNGt.xlsx', sheet_name= 'COVID')
leaked_data['MUNICIPIO']= leaked_data['MUNICIPIO'].apply(lambda x: x.upper())

leaked_data['lat']= leaked_data['MUNICIPIO'].map(lat_dic)
leaked_data['lng']= leaked_data['MUNICIPIO'].map(lng_dic)

# Dictionary for municipality to department
mun_dep_dic= dict(zip(leaked_data['MUNICIPIO'], leaked_data['DEPARTAMENTO']))

# Creating data set per day_country

raw_data_country= pd.pivot_table(leaked_data, values='CASOS', index=['FECHA DE EMISION \nDE RESULTADO LNS ', ], aggfunc=np.sum)

span= (raw_data_country.index[len(raw_data_country.index)-1]-raw_data_country.index[0]).days+1
start= datetime.strftime(raw_data_country.index[0], '%Y-%m-%d')
dates= pd.date_range(start, periods= span)

raw_data_country= raw_data_country.reindex(dates, fill_value= 0)

#%% Loading historic data Guatemala
   
url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
df = pd.read_csv(url, index_col=0, parse_dates=[0], usecols=['location' ,'date', 'new_cases', 'total_cases'])

country= 'Guatemala'
guatemala_raw= df[df.index==country]
guatemala_raw['date']= pd.to_datetime(guatemala_raw['date'], format='%Y-%m-%d')
guatemala_raw= guatemala_raw.set_index('date', drop= 'True')

#%% Datos reportados oficiales contra fecha de ejecución de pruebas

fig= plt.figure()
plt.plot(raw_data_country['CASOS'], label= 'Casos reportados MSPAS - Otra BBDD')
plt.plot(guatemala_raw['new_cases'], label= 'Casos reportados MSPAS - Datos oficiales')
plt.title('Datos MSPAS - Otra BBDD al 14/06/2020 vrs. Datos oficiales', size= 14)
plt.ylabel('Nuevos casos/día', size= 12)
plt.xlabel('Día', size= 12)
plt.xticks(rotation= 30)
plt.grid(color= 'lightgrey')
plt.legend(loc= 'best')
plt.savefig(path_output+'pruebas_realizadas_datos_oficiales.jpg', dpi= 500, bbox_inches='tight')
plt.show()

#%% Determining probable lag between time series

cut_reported= guatemala_raw['new_cases'][guatemala_raw.index <= pd.to_datetime('15-06-2020', format='%d-%m-%Y')]

joint_data_set= pd.concat([cut_reported, raw_data_country], axis= 1)

# Calculating acumulated difference
joint_data_set['Diferencia_abs']= (joint_data_set['CASOS']-joint_data_set['new_cases'])
joint_data_set['Diferencia_rel']= (joint_data_set['CASOS']-joint_data_set['new_cases'])/joint_data_set['CASOS']

# Plotting results

fig, (ax1, ax2)= plt.subplots(1, 2)
fig.suptitle('Diferencia absoluta y relativa entre pruebas positivas ejecutadas MSPAS \n y reportado Gobierno', y=1.08)

ax1.plot(joint_data_set['Diferencia_abs'])
ax1.tick_params(labelrotation=45, labelsize= 8)
ax1.set_xlabel('Días')
ax1.set_ylabel('Diferencia absoluta \n ejecutado MSPAS - reportado Gobierno')
ax1.grid(color= 'lightgrey')
      
ax2.plot(joint_data_set['Diferencia_rel']*100)
ax2.tick_params(labelrotation=45, labelsize= 8)
ax2.set_xlabel('Días')
ax2.set_ylabel('Diferencia relativa \n ejecutado MSPAS - reportado Gobierno (%)')
ax2.grid(color= 'lightgrey')

fig.tight_layout()
plt.savefig(path_output+'Diferencias_ejecutado_reportado.jpg', dpi= 500, bbox_inches='tight')
fig.show()

#%% Calculating the most probable lag

print('Variance with no lag: ', sum(((joint_data_set['CASOS']-joint_data_set['new_cases'])**2).dropna()))
print('Variance with lag-1 day: ' ,sum(((joint_data_set['CASOS'].shift(periods=1)-joint_data_set['new_cases'])**2).dropna()))
print('Variance with lag-2 days', sum(((joint_data_set['CASOS'].shift(periods=2)-joint_data_set['new_cases'])**2).dropna()))

fig= plt.figure()
plt.hist(abs(joint_data_set['CASOS'].shift(periods=0)-joint_data_set['new_cases']), bins= 'auto', label= 'Diferencia absoluta')
plt.hist(abs(joint_data_set['CASOS'].shift(periods=1)-joint_data_set['new_cases']), bins= 'auto', label= 'Con ajuste en reporte de 1 día')
plt.title('Histograma de frecuencia, fecha de: \n Ejecución de prueba MSPAS - Reporte oficial gobierno')
plt.xlabel('Días de atraso (ejecución de prueba) - Reporte oficial Gobierno')
plt.ylabel('Frecuencia de atraso al 14/06/2020')
plt.grid(color= 'lightgrey')
plt.legend()
plt.savefig(path_output+'histogram_after_before_correction.jpg', dpi= 500, bbox_inches='tight')
plt.show()

#%% Corrected reporting

fig= plt.figure()
plt.plot(raw_data_country['CASOS'], label= 'Casos reportados MSPAS - Otra BBDD')
plt.plot(guatemala_raw['new_cases'].shift(periods= -1), label= 'Casos reportados MSPAS - Datos oficiales ajustados',\
         marker= 'o', markersize= 1, linewidth= 1, linestyle= '--')
plt.title('Datos MSPAS - Otra BBDD al 14/06/2020 vrs. \n Datos oficiales ajustados', size= 14)
plt.ylabel('Nuevos casos/día', size= 12)
plt.xlabel('Día', size= 12)
plt.xticks(rotation= 30)
plt.grid(color= 'lightgrey')
plt.legend(loc= 'best')
plt.savefig(path_output+'Corrected_new_cases.jpg', dpi= 500, bbox_inches='tight')
plt.show()

#%% Plotting times series, before and after correction

fig, (ax1, ax2) = plt.subplots(1, 2)

plt.suptitle('Corrección de series de tiempo, casos nuevos', y= 1.25, size= 14)

l1 = ax1.plot(raw_data_country['CASOS'])[0]
l2 = ax1.plot(guatemala_raw['new_cases'], marker= 'o', markersize= 1, linewidth= 1, linestyle= '--')[0]

ax1.tick_params(labelrotation=45, labelsize= 8)
ax1.set_xlabel('Días')
ax1.set_ylabel('Casos nuevos')
ax1.grid(color= 'lightgrey')
      
l3 = ax2.plot(raw_data_country['CASOS'])[0]
l4 = ax2.plot(guatemala_raw['new_cases'].shift(periods= -1), marker= 'o', markersize= 0.8, linewidth= 0.8, linestyle= '-', color= 'magenta')[0]

ax2.tick_params(labelrotation=45, labelsize= 8)
ax2.set_xlabel('Días')
ax2.grid(color= 'lightgrey')

line_labels= ['Casos reportados MSPAS - Otra BBDD', 'Casos reportados MSPAS - Datos oficiales',\
              'Casos reportados MSPAS - Otra BBDD', 'Casos reportados MSPAS - Datos oficiales ajustados 1 día']

fig.legend([l1, l2, l3, l4],
           labels= line_labels, 
           loc= 'center',
           borderaxespad= 0,
           title= 'Descripción',
           bbox_to_anchor=(0.5, 1.14))

plt.savefig(path_output+'pruebas_realizadas_casos_reportados_corregidos.jpg', dpi= 500, bbox_inches='tight')
plt.show()

#%% Estimating the difference between onset and test

bucket = 'https://raw.githubusercontent.com/beoutbreakprepared/nCoV2019/master/latest_data'
key = 'latestdata.tar.gz'

thetarfile= bucket+'/'+key

ftpstream = urllib.request.urlopen(thetarfile)
thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
thetarfile.extractall(path= path_output)
thetarfile.close()

patients= pd.read_csv(path_output+'latestdata.csv', parse_dates= False,
            usecols=[ 'date_confirmation', 'date_onset_symptoms'], 
            low_memory= False)

patients.columns= ['Onset', 'Confirmed']

# There's an errant reversed date
patients = patients.replace('01.31.2020', '31.01.2020')

# Only keep if both values are present
patients = patients.dropna()

# Must have strings that look like individual dates
# "2020.03.09" is 10 chars long
is_ten_char = lambda x: x.str.len().eq(10)
patients = patients[is_ten_char(patients.Confirmed) & 
                    is_ten_char(patients.Onset)]

# Convert both to datetimes
patients.Confirmed = pd.to_datetime(
    patients.Confirmed, format='%d.%m.%Y')

# Dealing with non valid dates in Onset

indexes= list(patients.Onset.index)
Onset= []

for element in patients.Onset:
    
    try:
        Onset.append(pd.to_datetime(element, format='%d.%m.%Y'))
    
    except:
        Onset.append(np.nan)

patients.Onset= Onset
patients= patients.dropna()

# Only keep records where confirmed > onset
patients = patients[patients.Confirmed >= patients.Onset]

fig= plt.figure(figsize= (8, 8))
ax= fig.add_subplot(111)
ax.plot_date(patients.Onset, patients.Confirmed, color='blue', marker='s', markersize= 1, linewidth= 0)
ax.set_title("Síntomas vs. Caso confirmado - COVID19", fontsize= 14)
ax.set_xlabel("Día síntoma", fontsize= 10)
ax.set_ylabel("Día diagnóstico", fontsize= 10)
ax.grid(color='lightgrey', linestyle='-', linewidth= 0.5)
fig.savefig(path_output+'Diagn_Sintoma.jpg', dpi=500)
fig.show()

delay = (patients.Confirmed - patients.Onset).dt.days

#%% World probability of delay

p_delay= delay_distribution(delay)

# Show our work

fig, axes = plt.subplots(ncols=2, figsize=(9,3))
p_delay.plot(title='P(Diferencia días síntomas-diagnóstico) PMF', ax=axes[0])
p_delay.cumsum().plot(title='P(Atraso <= x) CDF', ax=axes[1])
for ax in axes:
    ax.set_xlabel('días')
    ax.grid(color='lightgrey', linestyle='-', linewidth= 0.5)
fig.tight_layout()
fig.savefig(path_output+'pmf_CDF_sint_diag.jpg', dpi=500)
fig.show()

#%% Making confirmed to onset cases

confirmed= guatemala_raw['new_cases'].shift(periods= -1)

confirmed= confirmed.fillna(0)

onset = confirmed_to_onset(confirmed, p_delay)

adjusted, cumulative_p_delay = adjust_onset_for_right_censorship(onset, p_delay)

#%% Analysis

data_set= pd.concat([confirmed, onset, adjusted], axis= 1)
data_set.columns= ['new_cases_confirmed', 'new_cases_onset', 'new_cases_adjusted']
data_set= data_set.drop(pd.Timestamp('2020-06-20'))

data_set= data_set[data_set['new_cases_adjusted']>=1] # Leaving data only when new_cases adjusted>=1

fig= plt.figure()
plt.plot(data_set['new_cases_confirmed'], \
         label= 'Nuevos casos/día oficiales Gobierno ajustado en días, total al 19/06/2020: \n '+str(round(data_set['new_cases_confirmed'].sum(), 0)))
plt.plot(data_set['new_cases_adjusted'], \
         label= 'Nuevos casos/día estimados por atraso en reporte + atraso en ejecución de prueba, \n total estimado al 19/06/2020: '+str(round(data_set['new_cases_adjusted'].sum(), 0)))
plt.xticks(rotation= 45)
plt.title('Nuevos casos positivos confirmados y estimados al 19/06/2020', size= 14)
plt.xlabel('Día', size= 11)
plt.ylabel('Nuevos casos/día', size= 11)
plt.grid(color= 'lightgrey')
plt.legend(bbox_to_anchor=(1.2, -0.35))
fig.savefig(path_output+'casos_nuevos_reales.jpg', bbox_inches='tight', dpi= 500)
plt.show()

#%% Calculating parameters of SIR model

par_db= leaked_data[~leaked_data['FECHA \nFALLECIDO/RECUPERADO'].isnull()]

leaked_data['CONDICION'].value_counts()
par_db['CONDICION'].value_counts()


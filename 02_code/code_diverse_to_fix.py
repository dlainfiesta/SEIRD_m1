# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:25:10 2020

@author: diego
"""

"""

# Making analysis for department

raw_data_dep= pd.pivot_table(leaked_data, values='CASOS', index=['FECHA DE EMISION \nDE RESULTADO LNS ', ],\
                          columns= ['DEPARTAMENTO'], aggfunc=np.sum)

span= (raw_data_dep.index[len(raw_data_dep.index)-1]-raw_data_dep.index[0]).days+1
start= datetime.strftime(raw_data_dep.index[0], '%Y-%m-%d')
dates= pd.date_range(start, periods= span)

raw_data_dep= raw_data_dep.reindex(dates, fill_value= 0)

raw_data_dep= raw_data_dep.fillna(0)

raw_data_dep.loc['Total']= raw_data_dep.sum(axis= 0)
raw_data_dep= raw_data_dep.sort_values('Total', axis=1, ascending=False)
raw_data_dep= raw_data_dep.drop('Total', axis= 0)

columns= raw_data_dep.columns

for departamento in columns:

    fig= plt.figure()
    ax= plt.subplot(111)
    plt.plot(raw_data_dep[departamento], marker= 'o', linestyle= 'None', markersize= 3, label= 'Nuevos casos/día')
    plt.title('Nuevos casos/día vrs. día departamento: \n'+departamento+' '+datetime.strftime(raw_data_dep.tail(1).index[0], '%d-%m-%Y'))
    plt.xlabel('Fecha')
    plt.ylabel('Casos')
    plt.xticks(rotation= 30, size= 7)
    plt.grid(color= 'lightgrey')
    plt.legend(loc= 'best')
    plt.tight_layout()
    plt.savefig(path_output_dep+departamento+' casos_diarios'+'.jpg', dpi= 500)
    plt.show()

# Making analysis for municipio
    
raw_data_mun= pd.pivot_table(leaked_data, values='CASOS', index=['FECHA DE EMISION \nDE RESULTADO LNS ', ],\
                          columns= ['MUNICIPIO'], aggfunc=np.sum)

span= (raw_data_mun.index[len(raw_data_mun.index)-1]-raw_data_mun.index[0]).days+1
start= datetime.strftime(raw_data_mun.index[0], '%Y-%m-%d')
dates= pd.date_range(start, periods= span)

raw_data_mun= raw_data_mun.reindex(dates, fill_value= 0)

raw_data_mun= raw_data_mun.fillna(0)

columns= raw_data_mun.columns

for municipio in columns:

    fig= plt.figure()
    ax= plt.subplot(111)
    plt.plot(raw_data_mun[municipio], marker= 'o', linestyle= 'None', markersize= 3, label= 'Nuevos casos/día')
    plt.title('Nuevos casos/día vrs. día municipio: \n'+municipio+' '+datetime.strftime(raw_data_dep.tail(1).index[0], '%d-%m-%Y'))
    plt.xlabel('Fecha')
    plt.ylabel('Casos')
    plt.xticks(rotation= 30, size= 7)
    plt.grid(color= 'lightgrey')
    plt.legend(loc= 'best')
    plt.tight_layout()
    plt.savefig(path_output_mun+municipio+' casos_diarios'+'.jpg', dpi= 500)
    plt.show()
   
"""

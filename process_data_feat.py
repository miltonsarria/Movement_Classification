import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import lfilter, resample, firwin, hilbert
from scipy.fftpack import fft
import math
import pickle
from tools import *


###############################################
###############################################
#         CODIGO PRINCIPAL ####
###
root_dir = 'data/'

folders_class = glob(root_dir+"*/",recursive=True) 

fil_dc = True
alpha  = 0.89

data_per_class = {}
classes = ['M1','M2','M3','M4','M5','M6','M7']
list_sensors = ['C1','C2','C3','C4']
data_per_class = {key: [] for key in classes}
len_window = 150
for folder in folders_class:
    clase = folder.split('/')[1][:2]
    text_files = glob(folder+"**/*.txt",recursive=True)
    names = []
    #generar los nombres de cada sesion    
    for t_file in text_files:
        names.append(t_file.split('/')[-1][:8])
    sessions = np.unique(names)
    data_sep = {key: [] for key in sessions}
    
    print(f'\n[INFO] processing {folder} de la clase : {clase}')
    i=1
    #cargar los 4 archivos por sesion y generar una matriz donde cada fila es un sensor
    for session in sessions:
        print(f'Session: {session}')
        DATA = []
        ENVS = []
        for C in list_sensors:
            data_file = folder + session + C + '.txt'
            #print(f'File: {data_file}')
            data0 = np.loadtxt(data_file)
            if fil_dc:
                data =  lfilter([1,-1], [1, -alpha], data0)
                data[:40] = 0
            else:
                data = data0
                
            env = get_env(data)
            DATA.append(data)
            ENVS.append(env)
        #promedio de envolvente de energia para cada sensor
        ave_env = np.mean(ENVS,axis=0)
        trh =ave_env.mean()*1.1   
        #detectectar sementos de eventos activos       
        if clase == 'M7':
           actv_segment = [np.arange(10,ave_env.size-10)]
        else:
           actv_segment= eventos(ave_env,trh=trh,min_duration=10) 
        #guardar informacion en diccionario
        data_sep[session]=[np.array(DATA),actv_segment]
        
        #plt.subplot(2,3,i)
        #i=i+1
        #mostrar resultados
        #seg = np.zeros(ave_env.size)
        #actv = np.hstack(actv_segment) #se agrupan en un solo vector
        #seg[actv] = 1; 
        
        #plt.plot(data_sep[session][0].T)
        #plt.plot(ave_env)
        #plt.plot(seg)
                    
#    plt.show()
    
    #para todas las sesiones de la clase, analizar cada segmento: extraer los vectores de caracteristicas
    X = analizar_segmentos(data_sep,fs=1000,min_duration=2,len_window=len_window)
    
    #agregar las caracteristicas a la clase correspondiente
    
    data_per_class[clase] = X   
        
 ######## guardar todo en un archivo binario
f = open('data_freq_'+str(len_window)+'.pickle', 'wb')
pickle.dump(data_per_class, f)
f.close()

#f = open('data.pickle', 'rb')
#unpickled = pickle.load(f)
#f.close()    

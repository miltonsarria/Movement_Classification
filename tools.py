import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import lfilter, resample, firwin, hilbert
from scipy.fftpack import fft
import statistics as stats
import math
import pickle
from librosa.filters import mel
from scipy.fftpack import dct

##### funcion para calcular envolvente
def get_env(s):
     s_h = hilbert(s) 
     s_h[s_h < np.finfo(float).eps] = np.finfo(float).eps  
     s_en =  np.log10(np.abs(s_h))         
     windowSize = 60;  #30 ms aprox using fs=1khz
     b = (1/windowSize)*np.ones(windowSize);
     a = 1;
     s_en = lfilter(b, a, s_en);
     s_en[0:windowSize+1] =s_en[windowSize+1]
     s_en=(s_en-s_en.min())/np.abs(s_en).max()
     return s_en

######## funcion para identificar eventos candidatos a ser actividad
def eventos(E,trh=0.5,min_duration=10):
    #en caso de que sea una envolvente aplicar umbral, no es necesario si ya es bool
    E = (E > trh).astype(int)
    #agregar ceros al inicio y final para garantizar segmentos cerrados
    E[E.size-2:]=0
    E[:2]=0
    Ed=np.roll(E,1)
    #calcular la derivada    
    O = E-Ed
    #detectar puntos de inicio y final
    ind_ini=np.where(O>0)[0]
    ind_fin=np.where(O<0)[0]
    #descartar los segmentos muy cortos
    length = ind_fin-ind_ini
    ind_ini = ind_ini[length>min_duration]
    ind_fin = ind_fin[length>min_duration]
    #comparar la distancia de elventos adyacentes para unirlos
    dist_ss = ind_ini-np.roll(ind_fin,1) #distancia entre segmentos adyacentes
    max_dist = 500 #aprox 500 ms, si la separacion es mayor son eventos diferentes
    long_segment=[]
    #depurar segmentos
    ini = ind_ini[0]

    for i in range(1, len(dist_ss)):
        if dist_ss[i]>max_dist:
            fin=ind_fin[i-1]
            long_segment.append(np.arange(ini-5,fin+5))
            ini=ind_ini[i]
      
    long_segment.append(np.arange(ini,ind_fin[-1]))

    #retornar los segmentos depurados
    return long_segment     
#
#############################################################
#############################################################
######### funcion para analizar los eventos de cada sesion
def analizar_segmentos(set_signal,fs=1000,min_duration=0.1,len_window=100):   
    fs = fs
    sessions = set_signal.keys()
    print('[INFO] proceso de extraer caracteristicas a cada evento sesion: ')
    FEATS = []
    for session in sessions:
        print(f'{session}')
        #data contiene la info de los 4 sensores en una matriz 4xT
        #sementos son los segmentos activos en la session
        data      = set_signal[session][0]
        segmentos = set_signal[session][1]

        #aislar la informacion de un segmento con sus 4 sensores
        for segmento in segmentos:
            chunk_data = data[:,segmento]
            if segmento.size/fs >  min_duration:
               #print("Duracion: ",segmento.size/fs)
               #tomar la senal de cada sensor para extraer caracteristicas
               feats_seg = []
               for sensor in chunk_data:
                   #### en esta parte se incluye el proceso de extraer caracteristicas
                   x = extr_caracteristicas(sensor,len_window)
                   feats_seg.append(x)
                   
                   ######
               #es necesario agregar las caracteristicas de forma horizontal  para unir los sensores 
               feats_seg=np.hstack(feats_seg)
               
               FEATS.append(feats_seg)
               
    #unir todas las caracteristicas de todos los segmentos en una sola matriz    
    FEATS = np.vstack(FEATS)

    return FEATS

#################
    
def extr_caracteristicas(s,len_window,sr=1000,n_fft=512):
    #banco de filtros 
    fil_bank=mel(sr=sr, n_fft = n_fft-1, n_mels = 20);    #nuevo de Milton
    ini = 0
    fin = len_window
    hop = len_window# int(len_window*0.25) #taslape del xx%
    
    x = []
    while fin < s.size:
        y = s[ini:fin]
        n = len(y)-1
        
        #calcular la FFT
        Y = fft(y,n=n_fft)
        # Calcular magnitud, normalizar por el num de muestras
        absY = abs(Y)/(y.size)                                      
        #reorganizar el espectro para graficar
        #numero de mxuestras hasta la mitad del espectro
        hN=int(math.floor((Y.size+1)/2))
        absY=2*absY[0:hN]
        # calcular la magnitud en dB
        # Si hay ceros, reemplazar por un valor muy pequeno, antes de aplicar el log
        absY[absY < np.finfo(float).eps] = np.finfo(float).eps    
        Ydb = 20 * np.log10(absY)
        #agrupar espectro por bandas
        fbank_e = np.dot(fil_bank, absY)
        #aplicar log
        log_fbank =  np.log10(fbank_e)
        #aplicar DCT
        ceps = dct(fbank_e,norm='ortho')
        
        
        ###Del proyecto de la órtesis###
        p = []
        vr = []
        suma_Rms = 0
        count_ZC = 0
        thresh_ZC = 0.01
        valor_Willison = 0
        thresh_Willison = 0.01 #valores modificables
        thresh_SC = 0.00003 #valores modificables
        count_SC = 0
        suma_WL = 0
        inte = []
        
        ###MIAS###
        sum_aac = 0
        sum_dasdv = 0
        sum_ssi = 0
        sum_wl = 0
        sum_rms = 0
        
        for i in range(n):
            #Mav
            p.append(abs(y[i]))
            #Var
            vr.append(abs(y[i]))
            #Rms
            suma_Rms = suma_Rms + y[i]**2
            #ZeroCrosing
            if ((y[i]>0 and y[i+1]<0) or (y[i]<0 and y[i+1]>0)) and abs(y[i]-y[i+1]) >= thresh_ZC:
                count_ZC += 1
            else:
                count_ZC = count_ZC
            #Willison    
            if (y[i]-y[i+1])>thresh_Willison:
                valor_Willison += 1
            else:
                valor_Willison = valor_Willison  
            #SlopeChange
            if((y[i]-y[i-1])*(y[i]-y[i+1]))>=thresh_SC:
                count_SC += 1
            else:
                count_SC = count_SC
            #WaveLonge
            suma_WL += abs(y[i+1]-y[i])
            #Integral
            inte.append(abs(y[i]))
            
            #aac            
            a_aac = abs(y[i+1] - y[i])
            sum_aac = sum_aac + a_aac
            #dasdv
            a_dasdv = (y[i+1] - y[i])**2
            sum_dasdv = sum_dasdv + a_dasdv
            #ssi
            a_ssi = (y[i])**2
            sum_ssi = sum_ssi + a_ssi
            #wl
            sum_wl = sum_wl + abs(y[i+1] - y[i])    
            #rms     
            sum_rms = sum_rms + (y[i])**2  
              
        
        promedio = stats.mean(p)
        varianza = stats.pvariance(vr)   
        valor_Rms = math.sqrt(suma_Rms/len(y)) 
        integral = sum(inte)
        
        aac = sum_aac/n
        dasdv = np.sqrt(sum_dasdv/(n-1))
        ssi = sum_ssi
        wl = sum_wl
        rms = np.sqrt(sum_rms/n)
        y[y < np.finfo(float).eps] = np.finfo(float).eps 
        log = np.exp((np.sum(np.log10(abs(y))))/n)
        mav = np.mean(abs(y))
         
                 
        time_feats = np.array([promedio, varianza, valor_Rms, count_ZC, valor_Willison,\
         count_SC, suma_WL, integral, aac, dasdv, log, mav, ssi, wl, rms])
        

        #feat = [aac, dasdv, log, mav, ssi, wl, rms]
        
        #incluir todas las caracteristicas de la ventana en un solo vector
        #feat = np.hstack([ceps,time_feats])
        feat = ceps   
        #agregar a la matriz de caracteristicas de ese segmento
        x.append(feat)
        
        #avanzar los indices
        ini=ini+hop
        fin = fin + hop
    
    #convertir todo a un array (matriz donde cada fila son las caracteristicas de una ventana
    x = np.array(x)


    return x
###############################################


#ssc(signal, samplerate=16000, winlen=0.025, winstep=0.01, nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.97, winfunc=<function <lambda>>)

    

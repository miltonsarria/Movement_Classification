#################
#a esta funcion llega un segmento y se extran caracteristicas cada 100 ms, se agrupan en una matriz y se retorna 
def extr_caracteristicas(s,len_window,sr=1000,n_fft=512):
    #banco de filtros 
    fil_bank=librosa.filters.mel(sr=sr, n_fft = n_fft-1, n_mels = 20);    #nuevo de Milton
    ini = 0
    fin = len_window
    x = []
    while fin < s.size:
        y = s[ini:fin]
        n = len(y)-1
        
        #######################################FRECUENCIA###################
        #por ejemplo calcular la FFT
        #Y = fft(y)
        # Calcular magnitud, normalizar por el num de muestras
        #absY = abs(Y)/(y.size)                                      
        #reorganizar el espectro para graficar
        #numero de muestras hasta la mitad del espectro
        #hN=int(math.floor((Y.size+1)/2))
        #absY=np.hstack((absY[hN:],absY[:hN]))
        #calcular la magnitud en dB
        # Si hay ceros, reemplazar por un valor muy pequeno, antes de aplicar el log
        #absY[absY < np.finfo(float).eps] = np.finfo(float).eps    
        #Ydb = 20 * np.log10(absY) 
        
        #Solamente para graficar como prueba
        #plt.plot(Ydb)
        #plt.show()
        
        #NUEVA CARACTERISTICA EN FRECUENCIA DE MILTON
        #por ejemplo calcular la FFT
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
        log = np.exp((np.sum(np.log10(abs(y))))/n)
        mav = np.mean(abs(y))
         
        #feat = [promedio, varianza, valor_Rms, count_ZC, valor_Willison, count_SC, suma_WL, integral, np.mean(Ydb), np.std(Ydb), Ydb.max(), Ydb.min(),aac, dasdv, log, mav, ssi, wl, rms]
        
        #feat = [promedio, varianza, valor_Rms, count_ZC, valor_Willison, count_SC, suma_WL, integral, aac, dasdv, log, mav, ssi, wl, rms]
        
        #feat = [np.mean(Ydb), np.std(Ydb), Ydb.max(), Ydb.min()]   
        #feat = [aac, dasdv, log, mav, ssi, wl, rms]
        #feat = [promedio, varianza, valor_Rms, count_ZC, valor_Willison, count_SC, suma_WL, integral]
        #feat = [mav, wl, aac]
        
        #feat = ceps
        feat = np.hstack([ceps])
           
        #agregar a la matriz de caracteristicas de ese segmento
        x.append(feat)
        
        #avanzar los indices
        ini=fin
        fin = fin + len_window
    x = np.array(x)


    return x
###############################################


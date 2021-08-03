import librosa, librosa.display #carregar assim por conta do problema do display
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import locale #ponto->virgula

#load files
passaro, sr = librosa.load('passarinho.ogg')
bicoChato, sr1 = librosa.load('bico-chato-amarelo.mp3')
sanhacu, sr2 = librosa.load('sanhacu-da-amazonia.mp3') #sr=22050


#time vectors
def timeVector(sinal, sr):
# timePassaro = np.arange(0, len(passaro)/sr, 1/sr)
# timeBicoChato = np.arange(0, len(bicoChato)/sr1, 1/sr1)
# timeSanhacu = np.arange(0, len(sanhacu)/sr2, 1/sr2)
    timeVector = np.arange(0, len(sinal)/sr, 1/sr)
    return timeVector

def stft_dB(sinal,sr,n_fft=2048,win_length=2048,hop_length=512):
    D = np.abs(librosa.stft(sinal,  #stft 
                            n_fft,
                            win_length,))
                           # hop_length,))
                            #window=signal.windows.gaussian(win_length,95)))
                                          
    DB = librosa.amplitude_to_db(D, ref=np.max) #dB normalizado   
    m = DB.shape[1]
    xout = librosa.frames_to_time(np.arange(m+1), #tempo natural
                                  sr,
                                  hop_length)
    
    ax = librosa.display.specshow(DB, 
                                  x_coords=xout, 
                                  sr=sr,
                                  cmap='gray_r',
                                  hop_length=hop_length, 
                                  x_axis='time', 
                                  y_axis='log')
        
    return ax


def ponto_por_virgula(ax,colorbar=False):   
    ticks = []
    xticks = ax.get_xticks(minor=False)
    
    if colorbar==True: 
        [ticks.append(str(round(k,1)).replace('.',',')) for k in xticks] #mudando para virgula
        return ax.set_xticklabels(ticks)
    else:
        [ticks.append(str(k).replace('.',',')) for k in xticks] #mudando para virgula
        return ax.set_xticklabels(ticks) 


def plot(sinal,sr,legenda='Pássaro 1'):

    plt.rcdefaults()
    locale.setlocale(locale.LC_NUMERIC, "de_DE") #needs import locale
    plt.rcParams['axes.formatter.use_locale'] = True
    colors=list(mcolors.TABLEAU_COLORS.keys()) 

    #criando a figura
    #fig, (ax1,ax2) = plt.subplots(2,1)
    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(2, 1, figure=fig)

    #grafico 1 - Tempo [s]
    ax1 = fig.add_subplot(gs[0,0])

    ax1.plot(timeVector(sinal,sr),sinal)
    ax1.legend([legenda], loc = 'upper right')
    ax1.grid(color='gray',linestyle='--', linewidth=0.3)
    ponto_por_virgula(ax1)
    plt.ylabel('Amplitude')
    plt.xlabel('Tempo [s]')

    #grafico 2 - spectrograma
    ax2 = fig.add_subplot(gs[1,0])

    ax2 = stft_dB(sinal,sr) #,n_fft,win_length,hop_length)
    print(type(ax2))
    plt.colorbar(ax=ax2,
                #orientation='horizontal',
                #cax = cbaxes,
                format='%+2.0f dB',
                pad=0.01,
                anchor=(0,10),
                fraction=0.025)

    ponto_por_virgula(ax2,colorbar=True) #ponto por vírgula
    plt.ylabel('Frequência [Hz]') 
    plt.xlabel('Tempo [s]')
    plt.ylim([0,9000])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show() #end



plot(passaro, sr, 'Pássaro desconhecido')
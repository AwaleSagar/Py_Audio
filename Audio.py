#import all libraries 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

#import to read/write wav audio files 
from scipy.io.wavfile import read
from scipy.io.wavfile import write

#declare global variables 
song_path = './data/Faded.wav'

def audio_2_array(path):
    a = read(path)
    return np.array(a[1],dtype=float)

def array_2_audio(audio_array,samplerate= 44100):
    scaled = np.int16(audio_array/np.max(np.abs(audio_array)) * 32767)
    write('./data/test.wav', samplerate, scaled)
    return 0

song = audio_2_array(song_path)

#Reverse song -

rev_song = np.flip(song)

array_2_audio(rev_song,22100)
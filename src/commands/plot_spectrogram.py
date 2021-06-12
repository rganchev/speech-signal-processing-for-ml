from ..constants import OUTPUT_DIR
from .stft_example import read_wav_file
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import math

def plot_waveform(ax, wav):
  ax.plot(wav)
  ax.set(title='Waveform', xlabel='samples', ylabel='amplitude')

def plot_spectrogram(ax, wav, fs):
  win_size = math.floor(0.032 * fs) # use a 32ms window size
  ax.specgram(wav, Fs=fs, NFFT=win_size, noverlap=win_size / 2)
  ax.set(title='Spectrogram', xlabel='time', ylabel='frequency')

def main():
  fs, wav = read_wav_file()
  wav = wav[:65536] # Analyze only the first few seconds of the wav file

  _, (ax1, ax2) = plt.subplots(nrows=2)
  plot_waveform(ax1, wav)
  plot_spectrogram(ax2, wav, fs)

  output_file = OUTPUT_DIR.joinpath('spectrogram.png')
  plt.tight_layout()
  plt.savefig(output_file)
  print('Spectrogram saved in %s' % output_file)


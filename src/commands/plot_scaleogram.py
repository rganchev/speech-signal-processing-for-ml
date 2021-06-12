from ..constants import OUTPUT_DIR
from .stft_example import read_wav_file
from .plot_spectrogram import plot_waveform
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pywt

def plot_scaleogram(ax, wav):
  levels = 6 # use six levels of decomposition
  wavelet = 'sym8' # use a Symlet wavelet with 8 vanishing moments
  # Use wavedec to apply several levels of decomposition at once.
  coeffs = pywt.wavedec(wav, wavelet, level=levels, mode='periodization')
  # wavedec returns the coefficients in reverse order.
  # Flip them, so we get a better ordering in the heat map.
  coeffs = np.flip(coeffs, axis=0)
  # The first level coefficients vector has the largest size.
  size = coeffs[0].shape[0]
  # Repeat the coefficients in each level so that we get a rectangular matrix.
  arr = np.array([np.repeat(c, size / c.shape[0]) for c in coeffs])
  # Plot the heatmap using plt.imshow
  ax.imshow(np.abs(arr), aspect='auto')
  ax.set(title='Scaleogram', xlabel='translation', ylabel='scale',
         yticks=range(0, levels + 1, 2), yticklabels=range(1, levels + 2, 2))

def main():
  fs, wav = read_wav_file()
  wav = wav[:65536]

  _, (ax1, ax2) = plt.subplots(nrows=2)
  plot_waveform(ax1, wav)
  plot_scaleogram(ax2, wav)

  output_file = OUTPUT_DIR.joinpath('scaleogram.png')
  plt.tight_layout()
  plt.savefig(output_file)
  print('Scaleogram saved in %s' % output_file)

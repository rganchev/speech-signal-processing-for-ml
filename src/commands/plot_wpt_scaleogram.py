from ..constants import OUTPUT_DIR
from .stft_example import read_wav_file
from .plot_spectrogram import plot_waveform
import matplotlib as mpl
mpl.use('agg')

import matplotlib.pyplot as plt
import numpy as np
import pywt

def plot_wpt_scaleogram(ax, wav):
  maxlevel = 6
  wp = pywt.WaveletPacket(data=wav, wavelet='sym8', maxlevel=maxlevel)
  # Stack the coefficients of all nodes and arrange them in a rectangular matrix.
  # Use order='freq' to obtain the WPT coefficients ordered by frequency band.
  coeffs = np.stack([node.data for node in wp.get_level(maxlevel, order='freq')])
  ax.imshow(np.abs(coeffs), aspect='auto', origin='lower')
  ax.set(title='Scaleogram', xlabel='translation', ylabel='scale')

def main():
  fs, wav = read_wav_file()
  wav = wav[:65536]

  _, (ax1, ax2) = plt.subplots(nrows=2)
  plot_waveform(ax1, wav)
  plot_wpt_scaleogram(ax2, wav)

  output_file = OUTPUT_DIR.joinpath('wpt_scaleogram.png')
  plt.tight_layout()
  plt.savefig(output_file)
  print('WPT Scaleogram saved in %s' % output_file)

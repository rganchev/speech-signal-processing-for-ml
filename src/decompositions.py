import numpy as np
from scipy.signal import stft, istft
from pywt import WaveletPacket, wavedec, waverec, ravel_coeffs, unravel_coeffs
from librosa.core import load

RATE = 16000

def read_wav_file(path):
  return load(path, sr=RATE)[0]

def wp_paths_at_level(level):
  paths = ['']
  for l in range(level):
    paths = [path for parent in paths for path in [parent + 'a', parent + 'd']]
  return paths

def gen_wpt(signal, wavelet='db6', maxlevel=6):
  wp = WaveletPacket(data=signal, wavelet=wavelet, maxlevel=maxlevel)
  return np.stack([node.data for node in wp.get_level(maxlevel, order='natural')])

def gen_iwpt(coefs, wavelet='db6', maxlevel=6):
  wp = WaveletPacket(data=None, wavelet=wavelet, maxlevel=maxlevel)
  leaf_paths = wp_paths_at_level(maxlevel)
  for path, data in zip(leaf_paths, coefs):
    wp[path] = data

  return wp.reconstruct(update=False)

def gen_wt(signal, wavelet='db6', level=6):
  coeffs = wavedec(signal, wavelet, level=level)
  # Returns (coeffs, coeff_slices, coeff_shapes)
  return ravel_coeffs(coeffs)

def gen_iwt(coeffs_array, coeff_slices, coeff_shapes, wavelet='db6'):
  coeffs = unravel_coeffs(coeffs_array, coeff_slices, coeff_shapes, output_format='wavedec')
  return waverec(coeffs, wavelet)

# Ideal Binary Mask
def ibm(signal, noise):
  return np.where(np.abs(signal) - np.abs(noise) > 0, 1, 0)

# Ideal Ratio Mask
def irm(signal, noise):
  signal_mag_sq, noise_mag_sq = [np.square(np.abs(s)) for s in [signal, noise]]
  return np.sqrt(np.divide(signal_mag_sq, signal_mag_sq + noise_mag_sq))

# Optimal Ratio Mask: https://arxiv.org/pdf/1709.00917.pdf
def orm(signal, noise):
  signal_mag_sq, noise_mag_sq = [np.square(np.abs(s)) for s in [signal, noise]]
  cor = np.real(np.multiply(signal, np.conj(noise)))
  return np.divide(signal_mag_sq + cor, signal_mag_sq + noise_mag_sq + 2*cor)

def mix(files, ratio = 0.5):
  signals = [read_wav_file(file) for file in files]
  minlen = min([s.shape[0] for s in signals])
  signals = [s[:minlen] for s in signals]
  signal, *noises = signals
  noise = np.sum(noises, axis=0) / len(noises)
  signal *= ratio
  noise *= 1 - ratio
  return signal, noise, signal + noise

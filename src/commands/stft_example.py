from ..constants import DATA_DIR
import random
import numpy
import scipy.io.wavfile
import scipy.signal
import math
from glob import glob

def get_file_path():
  return random.choice(glob(str(DATA_DIR.joinpath('Speaker*', '*.wav'))))

def read_wav_file():
  file_path = get_file_path() # obtain a path to a sample WAV file
  return scipy.io.wavfile.read(file_path)

def main():
  numpy.set_printoptions(threshold=10, suppress=True)

  fs, wav = read_wav_file()
  print('Sampling frequency: %i Hz' % fs)
  print('Number of samples: %i' % wav.shape[0])

  win_size = math.floor(0.032 * fs) # use a 32ms window size
  f, t, coeffs = scipy.signal.stft(wav, fs=fs, window='hann',
                                  nperseg=win_size, noverlap=win_size / 2)
  print('Analyzed frequencies (Hz): %s' % f)
  print('Time steps (seconds): %s' % t)
  print('Output dimensions: %s' % (coeffs.shape,))

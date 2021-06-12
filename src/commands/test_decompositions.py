import numpy as np
import random
import pywt
import math
import json
import speechmetrics
from time import time
from pprint import pprint
from glob import glob
from os import path
from scipy.signal import stft, istft
from statistics import mean
from ..constants import DATA_DIR, OUTPUT_DIR
from ..decompositions import *

metric_names = ['pesq', 'stoi', 'sisdr']
metrics = speechmetrics.load(metric_names, window=None)

def get_random_files(n):
  return random.sample(glob(str(DATA_DIR.joinpath('Speaker*', '*.wav'))), n)

def calc_metrics(original_signal, reconstructed_signal):
  length = min(original_signal.shape[0], reconstructed_signal.shape[0])
  return metrics(original_signal[:length], reconstructed_signal[:length], rate=RATE)

def gen_samples(n):
  samples = [get_random_files(2) for _ in range(n)]
  print('Samples (signal, noise):')
  pprint([[path.basename(file) for file in sample] for sample in samples])
  return [mix(pair) for pair in samples]

def gen_mask(signal, noise):
  return ibm(signal, noise)

def test(name, samples, transform, **kwargs):
  print('Testing %s...' % name)
  start = time()
  reconstructed_signals = [transform(sample, **kwargs) for sample in samples]
  avg_time = (time() - start) / len(samples)
  scores = [calc_metrics(s[0], r) for s, r in zip(samples, reconstructed_signals)]
  avg_scores = { m: mean([s[m] for s in scores]) for m in metric_names }
  print('Avg. metrics: %s, Avg. time: %f' % (avg_scores, avg_time))
  return {
    'method': name,
    'avg_score': avg_scores,
    'avg_time': avg_time,
    **kwargs
  }

def test_stft(signals, **kwargs):
  signal, noise, mixture = signals
  decomp = stft(mixture, **kwargs)[2]
  mask = gen_mask(stft(signal, **kwargs)[2], stft(noise, **kwargs)[2])
  masked_stft = np.multiply(decomp, mask)
  return istft(masked_stft, **kwargs)[1]

def test_wt(signals, wavelet, level):
  signal, noise, mixture = signals
  wt, slices, shapes = gen_wt(mixture, wavelet, level)
  mask = gen_mask(gen_wt(signal, wavelet, level)[0], gen_wt(noise, wavelet, level)[0])
  masked_wt = np.multiply(wt, mask)
  return gen_iwt(masked_wt, slices, shapes, wavelet)

def test_wpt(signals, wavelet, level):
  signal, noise, mixture = signals
  wpt = gen_wpt(mixture, wavelet, level)
  mask = gen_mask(gen_wpt(signal, wavelet, level), gen_wpt(noise, wavelet, level))
  masked_wpt = np.multiply(wpt, mask)
  return gen_iwpt(masked_wpt, wavelet, level)

def gen_stft_params(fs):
  windows = ['hann', 'boxcar']
  window_sizes = [5, 10, 16, 25, 32, 50, 100, 120]
  overlap_pcts = [25, 50, 75]
  for window in windows:
    for size in window_sizes:
      for overlap_pct in overlap_pcts:
        nperseg = round(fs * size / 1000.0)
        yield {
          'fs': fs,
          'window': window,
          'nperseg': nperseg,
          'noverlap': round(overlap_pct * nperseg / 100)
        }

def gen_wavelet_params(sample_length):
  wavelets = pywt.wavelist(kind='discrete')
  decomp_levels = range(1, math.floor(math.log2(sample_length)) + 1)
  for wavelet in wavelets:
    for level in decomp_levels:
      yield {
        'wavelet': wavelet,
        'level': level
      }

def main():
  samples = gen_samples(10)
  sample_len = len(samples[0][0])
  results = []
  for stft_params in gen_stft_params(RATE):
    name = 'STFT, %dms %s window, %d%% overlap' % (
      round(stft_params['nperseg'] * 1000 / RATE),
      stft_params['window'],
      round(stft_params['noverlap'] * 100 / stft_params['nperseg'])
    )
    results.append(test(name, samples, test_stft, **stft_params))

  for wavelet_params in gen_wavelet_params(sample_len):
    params = (wavelet_params['wavelet'], wavelet_params['level'])
    results.append(test('Wavelet %s, %d levels' % params, samples, test_wt, **wavelet_params))
    results.append(test('Wavelet Packet %s, %d levels' % params, samples, test_wpt, **wavelet_params))
  
  sorted_results = sorted(results, key = lambda x: x['avg_score']['sisdr'])

  with open(OUTPUT_DIR.joinpath('test_decompositions.json'), 'w') as file:
    json.dump(sorted_results, file, indent=2)

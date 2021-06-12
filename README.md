# Speech Signal Processing for Machine Learning

This project contains examples and tests of Short-Time Fourier transform (STFT), Wavelet Transform (WT), and Wavelet Packet Transform (WPT) - signal transformations which can be applied to the input signal of voice processing ML models.

## Input Data

The tests use the [Speaker Recognition Audio Dataset](https://www.kaggle.com/vjcalling/speaker-recognition-audio-dataset/version/1). Before running the examples, this dataset must be downloaded and extracted into the `data` directory.

## Dependencies

Before installing the dependincies, consider creating a [virtual Python environment](https://docs.python.org/3/library/venv.html) with Python 3.7. Dependencies can be installed by running:

```sh
pip install -r requirements.txt
```

## Running the Examples

Examples are implemented as separate "commands". Each example can be executed by running

```sh
python main.py <command>
```

where `<command>` is one of:

* `stft_example`: Runs STFT on a random audio file and prints various statistics and dimensions of the transformation.
* `plot_spectrogram`: Picks a random audio file and plots its waveform and spectrogram.
* `plot_scaleogram`: Picks a random audio file and plots its waveform and WT scaleogram.
* `plot_wpt_scaleogram`: Picks a random audio file and plots its waveform and WPT scaleogram.
* `test_decompositions`: Performs an experiment which compares STFT, WT, and WPT in the context of Ideal Binary Mask (IBM) reconstruction. The following procedure is applied:
  * Generate 10 random voice mixtures;
  * Generate various parameter configurations for each transformation;
  * For each configuration and each audio mixture:
    * Calculate the IBM of a target signal from the mixture, using the configured transformation;
    * Apply the IBM to the mixture and invert the result;
    * Calcuate metrics (PESQ, STOI and SI-SDR) to evaluate the reconstructed signal.
  * Store the results to `out/results.json`.
  * NOTE: The running time for this command can be several hours. To run a smaller experiment, consider modifying `gen_stft_params` and `gen_wavelet_params` in `commands/test_decompositions.py` in order to reduce the amount of generated configurations.

### Randomization

Randomization is "fixed" by an invocation of `random.seed()` in `main.py`. To generate different random results, either remove this invocation or change the value of the seed.

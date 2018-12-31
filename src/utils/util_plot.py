from common import SAMPLING_FREQ
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.signal import stft
from scipy.signal import find_peaks


def plot_peaks_with_signal(signals, fig_name='peak_sig.png'):
  peaks, _ = find_peaks(signals, height=np.max(signals) / 5, distance=10)
  plt.figure()
  plt.plot(signals)
  plt.plot(peaks, signals[peaks], "x")
  plt.plot(np.zeros_like(signals), "--", color="gray")
  plt.savefig(fig_name)


def plot_stft_with_signal(signals, fig_name='stft.png', fs=SAMPLING_FREQ):
  """Plot Short-Time Fourier Transform.

  :param signals: 1D array
  :param fig_name: Figure name
  :param fs: Sampling frequency.
  :return: None
  """
  f, t, Zxx = stft(signals, fs, nperseg=2 ** 14)
  f[f > 0] = np.log(f[f > 0])

  # plot STFT
  plt.figure()
  plt.subplot(211)
  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0)
  plt.title('STFT Magnitude')
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')

  # plot raw signals
  plt.subplot(212)
  t_vec = np.arange(len(signals)) / fs
  plt.plot(t_vec, signals)
  plt.ylabel('Magnitude')
  plt.xlabel('Time [sec]')

  plt.savefig(fig_name)


def plot_fft_with_signal(signals, fig_name='fft.png', fs=SAMPLING_FREQ):
  """Plot signal spectrum.

  :param signals:
  :param fig_name:
  :param fs:
  :return:
  """
  N = len(signals)
  yf = fft(signals)
  xf = np.linspace(0.0, fs / 2, N // 2)

  # plot fft
  plt.figure()
  plt.subplot(211)
  plt.semilogx(xf, 2.0 / N * np.abs(yf[0:N // 2]))
  plt.grid()

  # plot raw signals
  plt.subplot(212)
  t_vec = np.arange(len(signals)) / fs
  plt.plot(t_vec, signals)
  plt.ylabel('Magnitude')
  plt.xlabel('Time [sec]')

  plt.savefig(fig_name)


def plot_filtered_with_signal(filtered, signals, fig_name='filtered_signal.png', fs=SAMPLING_FREQ):
  """Plot signal spectrum.

  :param signals:
  :param fig_name:
  :param fs:
  :return:
  """

  # plot fft
  plt.figure()
  plt.subplot(211)
  t_vec = np.arange(len(filtered)) / fs
  plt.plot(t_vec, filtered)
  plt.ylabel('Filtered')
  plt.xlabel('Time [sec]')

  # plot raw signals
  plt.subplot(212)
  t_vec = np.arange(len(signals)) / fs
  plt.plot(t_vec, signals)
  plt.ylabel('Orignal')
  plt.xlabel('Time [sec]')

  plt.savefig(fig_name)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def loaddata(dataname,labelname):
	data = np.loadtxt(open("datas.csv","rb"), delimiter=",", skiprows=0)
	label = np.loadtxt(open("labels.csv","rb"), delimiter=",", skiprows=0)

# def getfft(rowdata):
# 	fftdata = map(lambda x: np.fft.fft(x),list(rowdata))
# 	return fftdata

def getfrequency(rowdata,fs):
	# 每次输入一个样本的数据进来，共有32路信号

	#all frequency signal
	"""
	detal:1-4Hz,theta:4~8Hz,alpha:8~13Hz,beta:18~30Hz
	"""
	listdata = list(rowdata)
	delta = np.arrary(list(map(lambda x: butterbandpassfilter(x, 1, 4, fs),listdata)))
	theta = np.arrary(list(map(lambda x: butterbandpassfilter(x, 4, 8, fs),listdata)))
	alpha = np.arrary(list(map(lambda x: butterbandpassfilter(x, 8, 13, fs),listdata)))
	beta = np.arrary(list(map(lambda x: butterbandpassfilter(x, 18, 30, fs),listdata)))

	print(delta.shape)


def getfeature(delta,theta,alpha,beta):
	return delta

def butterbandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butterbandpassfilter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def testfrequency():
	samplerate = 512

	t = np.arange(0, 1.0, 1.0/samplerate)
	x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
	x1 = butterbandpassfilter(x, 100, 200, samplerate)
	x2 = butterbandpassfilter(x, 200, 250, samplerate)

	fftx = abs(np.fft.rfft(x))
	freqs = np.linspace(0, samplerate/2,len(fftx))

	fftx1 = abs(np.fft.rfft(x1))
	fftx2 = abs(np.fft.rfft(x2))

	plt.figure(facecolor='w',figsize=(9,8))

	plt.subplot(211)
	plt.plot(freqs,fftx,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(212)
	plt.plot(t,x,'b-')
	plt.title('Row Data')
	plt.show()

	plt.figure(facecolor='w',figsize=(9,8))
	plt.subplot(211)
	plt.plot(freqs,fftx1,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(212)
	plt.plot(t,x1,'b-')
	plt.title('Row Data')
	plt.grid()
	plt.show()

	plt.figure(facecolor='w',figsize=(9,8))
	plt.subplot(211)
	plt.plot(freqs,fftx2,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(212)
	plt.plot(t,x2,'b-')
	plt.title('Row Data')
	plt.grid()

	plt.show()

	
	
if __name__ == '__main__':
	dataname = 'onedatas.csv'
	labelname = 'onelabels.csv'
	fs = 128
	# testfrequency()
	data,label = loaddata(dataname, labelname)
	getfrequency(data, fs)


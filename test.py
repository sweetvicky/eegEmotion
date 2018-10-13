import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def loaddata(dataname,labelname):
	data = np.loadtxt(open(dataname,"rb"), delimiter=",", skiprows=0)
	label = np.loadtxt(open(labelname,"rb"), delimiter=",", skiprows=0)
	return data,label

def getfrequency(rowdata,fs):
	# 每次输入一个样本的数据进来，共有32路信号

	#all frequency signal
	"""
	detal:1-4Hz,theta:4~8Hz,alpha:8~13Hz,beta:18~30Hz
	"""
	listdata = list(rowdata)
	delta = np.array(list(map(lambda x: butterbandpassfilter(x, 1, 4, fs),listdata)))
	theta = np.array(list(map(lambda x: butterbandpassfilter(x, 4, 8, fs),listdata)))
	alpha = np.array(list(map(lambda x: butterbandpassfilter(x, 8, 13, fs),listdata)))
	beta = np.array(list(map(lambda x: butterbandpassfilter(x, 18, 30, fs),listdata)))

	return delta,theta,alpha,beta

def butterbandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butterbandpassfilter(data, lowcut, highcut, fs, order=5):
    b, a = butterbandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def testfrequency(data,delta,theta,alpha,beta,fs):

	t = np.arange(0, 60, 60/len(data))


	fftdata = abs(np.fft.rfft(data))
	freqs = np.linspace(0, fs/2,len(fftdata))

	fftdelta = abs(np.fft.rfft(delta))
	ffttheta = abs(np.fft.rfft(theta))
	fftalpha = abs(np.fft.rfft(alpha))
	fftbeta = abs(np.fft.rfft(beta))

	plt.figure(facecolor='w',figsize=(9,8))

	plt.subplot(521)
	plt.plot(freqs,fftdata,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(522)
	plt.plot(t,data,'b-')
	plt.title('Row Data')
	

	plt.subplot(523)
	plt.plot(freqs,fftdelta,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(524)
	plt.plot(t,delta,'b-')
	plt.title('Row Data')
	plt.grid()
	

	plt.subplot(525)
	plt.plot(freqs,ffttheta,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(526)
	plt.plot(t,theta,'b-')
	plt.title('Row Data')
	plt.grid()

	plt.subplot(527)
	plt.plot(freqs,fftalpha,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(528)
	plt.plot(t,alpha,'b-')
	plt.title('Row Data')
	plt.grid()

	plt.subplot(529)
	plt.plot(freqs,fftbeta,'b-')
	plt.title('Row Data FFT')
	plt.grid()
	plt.subplot(5,2,10)
	plt.plot(t,beta,'b-')
	plt.title('Row Data')
	plt.grid()

	plt.show()


if __name__ == '__main__':
	dataname = 'onedatas.csv'
	labelname = 'onelabels.csv'
	fs = 128
	
	data,label = loaddata(dataname, labelname)
	# testdata = data[0,:]
	delta,theta,alpha,beta = getfrequency(data, fs)
	testfrequency(data[0,:], delta[0,:], theta[0,:], alpha[0,:], beta[0,:], fs)


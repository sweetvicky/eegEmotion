import numpy as np
import matplotlib.pyplot as plt

def loaddata(dataname,labelname):
	data = np.loadtxt(open("datas.csv","rb"), delimiter=",", skiprows=0)
	label = np.loadtxt(open("labels.csv","rb"), delimiter=",", skiprows=0)

def getfft(rowdata):
	fftdata = map(lambda x: np.fft.fft(x),list(rowdata))
	return fftdata

def getfrequency(rowdata,fs):
	# 每次输入一个样本的数据进来，共有32路信号
	datalen = rowdata.shape[1]
	fftdata = getfft(rowdata)

	freq = map(lambda x: abs(x/datalen),fftdata)
	freq = map(lambda x: x[: datalen/2+1]*2,freq)


	#all frequency signal
	"""
	theta:4~7Hz,alpha:8~13Hz,beta:18~30Hz
	"""
	delta = np.arrary(list(map(lambda x: x[datalen*1/fs-1: datalen*4/fs],freq)))
	theta = np.arrary(list(map(lambda x: x[datalen*4/fs-1: datalen*8/fs],freq)))
	alpha = np.arrary(list(map(lambda x: x[datalen*8/fs-1: datalen*13/fs],freq)))
	beta = np.arrary(list(map(lambda x: x[datalen*13/fs-1: datalen*13/fs],freq)))

	print(delta.shape)


def getfeature(delta,theta,alpha,beta):
	return delta

def testfrequency():
	samplerate = 256
	t = np.arange(0, 1.0, 1.0/samplerate)
	x = np.sin(2*np.pi*156.25*t)  + 2*np.sin(2*np.pi*234.375*t)
	fftx = np.fft.rfft(x)
	freqs = np.linspace(0, samplerate/2,len(fftx))
	plt.figure(facecolor='w',figsize=(9,8))
	plt.plot(freqs,fftx,'b-')
	plt.grid()
	plt.show()

	
	
if __name__ == '__main__':
	dataname = 'datas.csv'
	labelname = 'labels.csv'
	fs = 128
	testfrequency()
	# data,label = loaddata(dataname, labelname)
	# getfrequency(data, fs)


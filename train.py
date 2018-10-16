import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from sklearn import svm
from sklearn.metrics import accuracy_score

import pyeeg

fs = 128
tonums = 40 
frepnums = 4


def loaddata(dataname,labelname):
	data = np.loadtxt(open(dataname,"rb"), delimiter=",", skiprows=0)
	label = np.loadtxt(open(labelname,"rb"), delimiter=",", skiprows=0)
	return data,label

def getfrequency(rowdata):
	# 每次输入一个样本的数据进来，共有32路信号

	#all frequency signal
	"""
	detal:1-4Hz,theta:4~8Hz,alpha:8~13Hz,beta:18~30Hz
	"""
	listdata = list(rowdata)
	delta = np.reshape(np.array(list(map(lambda x: butterbandpassfilter(x, 1, 4, fs),listdata))),(tonums,-1,data.shape[1]))
	theta = np.reshape(np.array(list(map(lambda x: butterbandpassfilter(x, 4, 8, fs),listdata))),(tonums,-1,data.shape[1]))
	alpha = np.reshape(np.array(list(map(lambda x: butterbandpassfilter(x, 8, 13, fs),listdata))),(tonums,-1,data.shape[1]))
	beta = np.reshape(np.array(list(map(lambda x: butterbandpassfilter(x, 18, 30, fs),listdata))),(tonums,-1,data.shape[1]))

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



def getonefeatures(data):

	listdata = list(data)
	# Std
	fstd = np.array(list(map(lambda x: np.std(x),listdata)))

	# Approximate-Entropy(ApEN)
	m = 3
	r = 0.2*fstd
	fae = np.array(list(map(lambda x,y: pyeeg.ap_entropy(x, m, y),listdata,list(r))))
	
	# Power
	# fpower,fre = pyeeg.bin_power(data, [1,30], fs)
	# print(fpower)
	# print("特征--{std:%.4f,AE:%.4f,Power:%.4f}"%(fstd,fae,fpower))

	# First-order Diff ???
	# firstoderlist = pyeeg.first_order_diff(data)

	# Hjorth
	fhjormod_com = np.array(list(map(lambda x: pyeeg.hjorth(x),listdata)))
	fhjor_act = np.array(list(map(lambda x: np.var(x),listdata)))

	# Spectrum Entropy
	# fse = pyeeg.spectral_entropy(data, [0,fs/2], fs)
	# print(fse)

	# Power Spectral Density
	fpsd = np.array(list(map(lambda x: np.max(plt.psd(x,fs)[0]),listdata)))

	# Features Stack
	featurestmp = np.stack((fstd,fae,fhjormod_com[:,0],fhjormod_com[:,1],fhjor_act,fpsd),axis=1)
	temprow,tmpcol = featurestmp.shape
	features = np.reshape(featurestmp, (temprow*tmpcol,))
	return features

def getfeatures(delta,theta,alpha,beta):

    # features's column is setted by yourself
    '''
    frepnums:numember of frequency ; 
    channelnums:numember of channel ;
    featnums:numember of features ;
    '''
    channelnums = 2
    featnums = 6
    features = np.zeros((0,frepnums*channelnums*featnums))

    for r in range(delta.shape[0]):
        # for r in range(1):
        tmpdelta = getonefeatures(delta[r,:5,:])
        tmptheta = getonefeatures(theta[r,:5,:])
        tmpalpha = getonefeatures(alpha[r,:5,:])
        tmpbeta = getonefeatures(beta[r,:5,:])
        tmpfeatures = np.reshape(np.stack((tmpdelta,tmptheta,tmpalpha,tmpbeta),axis=0),(features.shape[1],))
        # tmpfeatures = np.stack((tmpdelta,tmptheta,tmpalpha,tmpbeta),axis=0)
        features = np.row_stack((features,tmpfeatures))

    return features

def testpsd(data):
	fpsd = np.max(plt.psd(data[0,0,:],fs)[0])

def svmclassier(datas,labels):
    # Split traindata and testdata
    x_train, x_test, y_train, y_test = train_test_split(datas, labels, train_size=0.8, random_state=1)

    # Classier
    clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr')
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    clf.fit(x_train, y_train.ravel())

    # Accuracy
    clf.score(x_train,y_train)
    train_acc = accuracy_score(y_train, clf.predict(x_train))

    clf.score(x_test, y_test)
    test_acc = accuracy_score(y_test, clf.predict(x_test))

    print("train accurate：%.2f",train_acc)
    print("test accurate:%.2f",test_acc)
	
	
if __name__ == '__main__':
	dataname = '..//data/onedatas.csv'
	labelname = '..//data/onelabels.csv'
	fs = 128
	# testfrequency()	
	data,label = loaddata(dataname, labelname)
	indexdata = []
	for r in range(40):
		indexdata = indexdata + list(range(r*32,r*32+2))
	# print(indexdata)
	delta,theta,alpha,beta = getfrequency(data[indexdata])
	# print(delta.shape)
	features = getfeatures(delta,theta,alpha,beta)
	# print(features.shape)
	svmclassier(features, label)




import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def savedatasets(dirname,dataname,labelname,eegchanel,eegnum,trigernum):
	files = os.listdir(dirname)
	subdrows = eegchanel * trigernum
	drows = len(files) * subdrows
	dcloums = eegnum
	labelen = len(files) * trigernum
	datas = np.zeros((drows,dcloums),dtype=np.float)
	labels = np.zeros((labelen,4))
	dnum = 0
	lnum = 0
	for filename in files:
		filenametmp = dirname + '/' + filename
		datatmp = pickle.load(open(filenametmp, 'rb'),encoding='iso-8859-1')
		datas[dnum:dnum+subdrows,:] = datatmp['data'][:,:eegchanel,-eegnum:].reshape((-1,eegnum))
		labels[lnum:lnum+trigernum,:] = datatmp['labels'] 
		dnum += subdrows
		lnum += trigernum

	return datas,labels 
	

def plotVAemotion(labels):
	neglabels = labels[labels[:,0]<5,:]
	poslabels = labels[labels[:,0]>5,:]
	callabels = labels[labels[:,0]==5,:]

	# 画二维图像
	# plt.figure(facecolor='w',figsize=(9,8))
	# plt.plot(neglabels[:,0],neglabels[:,1],'mv',label='消极')
	# plt.plot(poslabels[:,0],poslabels[:,1],'ro',label='积极')
	# plt.plot(callabels[:,0],callabels[:,1],'b*',label='平静')
	# plt.grid()
	# plt.title('Valence-Arousal维度标签')
	# plt.show()

	# 画三维图像
	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(neglabels[:,0], neglabels[:,1], neglabels[:,3],
           marker='v', color='b', label='消极')
	ax.scatter(poslabels[:,0], poslabels[:,1], poslabels[:,3],
           marker='o', color='m', label='积极')
	ax.scatter(callabels[:,0], callabels[:,1], callabels[:,3],
           marker='^', color='r', label='平静')
	ax.scatter(np.median(neglabels[:,0]), np.median(neglabels[:,1]), np.median(neglabels[:,3]),
           marker='v', color='k', label='消极分类中心')
	ax.scatter(np.median(poslabels[:,0]), np.median(poslabels[:,1]), np.median(poslabels[:,3]),
           marker='o', color='k', label='消极分类中心')
	plt.title('VAL情感维度标签')
	ax.set_xlabel('Valence')
	ax.set_ylabel('Arousal')
	ax.set_zlabel('Liking')
	plt.show()

def labelsmapping(labels):
	print(len(list(labels[:,0])))
	newlabels = np.array(list(map(lambda x: int(x >= 4.5)+int(x > 5.5)-1, list(labels[:,0]))))
	return newlabels

if __name__ == '__main__':
	eegchanel = 32
	eegnum = 7680
	trigernum = 40
	rootdir = '..//data'
	onerootdir = '..//onedata'
	datas,labelss = savedatasets(onerootdir,'dataname','labelname',eegchanel,eegnum,trigernum)
	# plotVAemotion(labelss)
	labels = labelsmapping(labelss)
	# print(datas.shape)
	# print(labels.shape)
	np.savetxt("onedatas.csv", datas, delimiter=',')
	np.savetxt("onelabels.csv", labels, delimiter=',')

	# testdata = np.array([5.4,6,8,2])
	# newlabels = np.array(list(map(lambda x: int(x >= 4.5)+int(x > 5.5)-1, list(testdata))))
	# print(newlabels)

# testdata = np.loadtxt(open("datas.csv","rb"), delimiter=",", skiprows=0)
# testlabel = np.loadtxt(open("labels.csv","rb"), delimiter=",", skiprows=0)
# print(testdata.shape)
# print(testlabel)

# print('Valence的min：%.2f,max:%.2f,mean:%.2f。'%(min(labels[:,0]),max(labels[:,0]),np.mean(labels[:,0])))
# print('Arousal的min：%.2f,max:%.2f'%(min(labels[:,1]),max(labels[:,1])))


# files=os.listdir(rootdir) #列出当前目录下所有的文件
# print(type(files))
# for filename in files:
# 	rootdir1=rootdir+'/'+filename

# x = pickle.load(open(rootdir1, 'rb'),encoding='iso-8859-1')
# y = x['data']
# print(y.shape)
# print(y[:,:eegchanel,-eegnums:].reshape((-1,eegnums)).shape)
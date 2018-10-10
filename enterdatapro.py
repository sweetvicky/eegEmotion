import pybdf
import numpy as np
# def readfile():
# 	pass
# def main(person):
# 	for i in range(1,3):
# 		path = 'Part'+person+'_IAPS_SES'+'_EEG_fNIRS'
# 	readfile()
	

# if __name__ == '__main__':
	# #选择实验者序号
	# person = '1';
	# main(person)
# 读取EEG数据
rec = pybdf.bdfRecording('sample.bdf')
data = rec.getData()
sampRate = rec.sampRate
durtime = rec.duration
chanlabels = rec.chanLabels
labels = rec.getData(channels=[-1])
print('EEG通道个数为：%d' % len(sampRate))
print('EEG信号采样频率是%d' % sampRate[0])
print('经过时间是：%.2f秒，%.2f分钟' % (durtime,durtime/60.0))
print('EEG信号采样频率是%d' % sampRate[0])
print('EEG采样通道:',chanlabels)
print('其中脑电信号通道为%d个通道，眼电信号为%d个通道'%((len(chanlabels)-9),8))
print('EEG数据结构：')
for key in labels.keys():
	print(key)
print('EEG-Data：',labels['data'].shape)
print('EEG-TrigChan：',labels['trigChan'])
print('EEG-SysCodeChan：',labels['sysCodeChan'])
print('EEG-SysCodeChan：',labels['sysCodeChan'])
print('EEG-EventTable：')
for key in labels['eventTable'].keys():
	print(key,labels['eventTable'][key].shape)
# print(labels['eventTable']['dur'])
# print('EEG-Data：',labels)
print('EEG-Data：',labels['data'])
print(data['data'].shape)
# 获取EEG数据刺激点
datamrk = np.loadtxt('../newdata/Part1_IAPS_SES2_EEG.txt')
dur1 = datamrk[:,1] - datamrk[:,0]
print(dur1)
dur2 = datamrk[1:,0] - datamrk[:-1,0]
print(dur2)



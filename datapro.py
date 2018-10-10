import pickle
import os

rootdir = '..//data'
files=os.listdir(rootdir) #列出当前目录下所有的文件
for filename in files:
	rootdir1=rootdir+'/'+filename

x = pickle.load(open(rootdir1, 'rb'),encoding='iso-8859-1')
print(x['data'].shape)
import caffe
import numpy as np
from PIL import Image
import os

# result folder
folder_name = 'result/'
if os.path.isdir(folder_name):
  pass
else:
  os.makedirs(folder_name)

# load test image list
filename = 'data/list.txt'
with open(filename, 'r') as f:
  path_src = [line.rstrip() for line in f.readlines()]

# set up caffe
caffe.set_device(0)
caffe.set_mode_gpu()

# load net
net = caffe.Net('model/deploy_512.prototxt', 'model/harmonize_iter_200000.caffemodel', caffe.TEST)

size = np.array([512,512])
for idx, path_ in enumerate(path_src):
  	# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
	im_ori = Image.open('data/image/' + path_)
	im = im_ori.resize(size, Image.BICUBIC)
	im = np.array(im, dtype=np.float32)
	if im.shape[2] == 4:
    		im = im[:,:,0:3]

	im = im[:,:,::-1]
	im -= np.array((104.00699, 116.66877, 122.67892))
	im = im.transpose((2,0,1))

	mask = Image.open('data/mask/' + path_)
	mask = mask.resize(size, Image.BICUBIC)
	mask = np.array(mask, dtype=np.float32)
	if len(mask.shape) == 3:
    		mask = mask[:,:,0]

	mask -= 128.0
	mask = mask[np.newaxis, ...]

	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *im.shape)
	net.blobs['data'].data[...] = im

	net.blobs['mask'].reshape(1, *mask.shape)
	net.blobs['mask'].data[...] = mask

	# run net for prediction
	net.forward()
	out = net.blobs['output-h'].data[0]
	out = out.transpose((1,2,0))
	out += np.array((104.00699, 116.66877, 122.67892))
	out = out[:,:,::-1]
  
  	neg_idx = out < 0.0
  	out[neg_idx] = 0.0
  	pos_idx = out > 255.0
  	out[pos_idx] = 255.0

  	# save result
  	result = out.astype(np.uint8)
  	result = Image.fromarray(result)
	
  	im = im_ori.resize(size, Image.BICUBIC);
  	im = np.array(im, dtype=np.uint8)
  	if im.shape[2] == 4:
    		im = im[:,:,0:3]

  	end = path_.find('.')
  	result_all = np.concatenate((im, result), axis = 1)
  	result_all = Image.fromarray(result_all)
  	result_all.save(folder_name + path_[0:end] + '.png')

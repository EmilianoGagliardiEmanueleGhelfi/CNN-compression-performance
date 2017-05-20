import os

keywords = ['refs:', 'misses:', 'miss rate:','accuracy','loss']
blacklist = ['Batch']
parsing_dict = {'refs':['I','D','LL'],'misses':['I1','LLi','D1','LLd','LL'],'rate':['I1','LLi','D1','LLd','LL']}
# 
# This class contains all information relative to the net performance
#
class NetPerformance:
	net = None
	caffemodel = None
	caffemodel_size = None
	accuracy = None
	loss = None
	test_time = None
	cachegrind_output_file = None
	compression_mode = None
	error_margin = None

# output info is the output of caffe test running into cachegrind, need to parse it
	def __init__(self,net,caffemodel,cachegrind_output_file,output_info,compression_mode,test_time):
		self.net = net
		self.caffemodel = caffemodel
		self.cachegrind_output_file = cachegrind_output_file
		self.caffemodel_size = os.path.getsize(caffemodel)
		self.compression_mode = compression_mode
		self.test_time = test_time
		self.__parse__(output_info)
		print str(self)

# sorry for this... This is due to cachegrind output formatting...
	def __parse__(self,output_info):
		lines = filter(lambda x: any([y in x for y in keywords]) and all([not y in x for y in blacklist]),output_info.split('\n'))
		for line in lines:
			for key in parsing_dict.keys():
				if key in line.replace(':','').split(' '):
					for val in parsing_dict[key]:
						if val in line.split(' '):
							if not val in ['I','I1','LLi']:
								# last two cases, get last element before close bracket
								string_list = line.split('(')[0].split(' ')
								string_list_filtered = [x for x in string_list if not x == '']
								attrValue = string_list_filtered[len(string_list_filtered)-1]
							else:
								# the first case in cachegrind file
								attrValue = line.split(' ')[len(line.split(' '))-1]
							# replace % with nothing and , with . because of float
							attrValue = float(attrValue.replace('%','').replace(',',''))
							setattr(self,val+"_"+key,attrValue)
			# end of for on dictionary
		last_acc = [line for line in lines if 'accuracy' in line][len([line for line in lines if 'accuracy' in line])-1]
		self.accuracy = float(last_acc.split(' ')[len(last_acc.split(' '))-1])
		last_loss = [line for line in lines if 'loss' in line][len([line for line in lines if 'loss' in line])-1]
		self.loss = float(last_loss.split('(')[0].split(' ')[len(last_loss.split('(')[0].split(' '))-2])

	def __str__(self):
		representation = ""
		for key in self.__dict__.keys():
			representation += key +": "+str(self.__dict__[key])+"\n"
		return representation
			

keywords = ['refs:', 'misses:', 'miss rate:','accuracy','loss']
blacklist = ['Batch']
#
# the input should be a list of strings to filter. 
#
def parser_filter(strings_list):
	return filter(lambda x: any([y in x for y in keywords] and all([not y in x for y in blacklist])),strings_list)




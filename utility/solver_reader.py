from caffe.proto import caffe_pb2
from google.protobuf.text_format import Merge
from google.protobuf.text_format import MessageToString

class SolverReader:
	
	def __init__(self, solver_path):
		self.solver_path = solver_path
		self.solver = caffe_pb2.SolverParameter()
		f = open(solver_path, 'r')
		Merge(f.read(), self.solver)


	def weightsFilename(self):
		"""
		returns the name of the file that caffe creates after the training
		using the information in the solver prototxt file
		"""
		snapshot_prefix = self.solver.snapshot_prefix
		max_iter = self.solver.max_iter
		return snapshot_prefix + '_iter_' + str(max_iter) + '.caffemodel'

	def compressionOutputFilename(self, compression_mode):
		"""
		convention to define names of the compressed network
		"""
		net = self.solver.net
		return net.split('.')[0] + '_' + compression_mode + '.prototxt'

	def fineTuneSolverFile(self, compression_mode):
		"""
		create a new file solver for the compressed network with the 
		adopted convention and return the filename
		"""
		new_solver = caffe_pb2.SolverParameter()
		f = open(self.solver_path, 'r')
		Merge(f.read(), new_solver) # new_solver is a clone of self.solver
		# change the solver file
		new_solver.net = compressionOutputFilename(compression_mode)
		new_solver.snapshot_prefix += '_' + compression_mode
		filename = self.solver_path.split('.')[0] + '_' + compression_mode + '.prototxt'
		new_solver_file = open(filename, 'w')
		new_solver_file.write(MessageToString(new_solver))
		new_solver_file.close()
		return filename
import torch

from MerCBO.graphGP.likelihoods.gaussian import GaussianLikelihood
from MerCBO.graphGP.means.constant import ConstantMean
from MerCBO.graphGP.models.gp import GP


class GPRegression(GP):

	def __init__(self, kernel, mean=ConstantMean()):
		super(GPRegression, self).__init__()
		self.kernel = kernel
		self.mean = mean
		self.likelihood = GaussianLikelihood()

	def init_param(self, output_data):
		output_mean = torch.mean(output_data).item()
		output_log_var = (0.5 * torch.var(output_data)).log().item()
		self.kernel.log_amp.fill_(output_log_var)
		self.mean.const_mean.fill_(output_mean)
		self.likelihood.log_noise_var.fill_(output_mean / 1000.0)

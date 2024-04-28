import torch
import torch.nn as nn

class KGAvgReadout(nn.Module):
	def __init__(self):
		super(KGAvgReadout, self).__init__()
	def forward(self, seq, msk):
		if msk is None:
			return torch.mean(seq, 1)
		else:
			msk = torch.unsqueeze(msk, -1)
			return torch.sum(seq * msk, 1) / torch.sum(msk)

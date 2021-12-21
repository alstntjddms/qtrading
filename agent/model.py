import torch.nn as nn

class DQN(nn.Module):
	def __init__(self, state_size, action_size=3):
		super(DQN, self).__init__()
		self.main = nn.Sequential(
			nn.Linear(state_size, 256),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Linear(256, 64),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Linear(64, 32),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Linear(32, 16),
			nn.LeakyReLU(0.01, inplace=True),

			nn.Linear(16, action_size)

		)

	def forward(self, input):
		return self.main(input)
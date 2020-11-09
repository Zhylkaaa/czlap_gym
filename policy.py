import torch
import torch.nn.functional as F

class Policy(torch.nn.Module):
    def __init__(self, input_size, hidden_size, out_size, constraints):
        super(Policy, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, out_size)
        assert out_size == len(constraints)
        self.constraints = torch.tensor(constraints, requires_grad=False)

    def forward(self, x):
        relu1 = self.linear1(x).clamp(min=0)
        model_out = self.linear2(relu1)

        rescaled_output = F.tanh(model_out) * self.constraints
        return rescaled_output

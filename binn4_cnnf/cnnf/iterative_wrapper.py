import torch 
# this is to call the model correctly -acs
class IterativeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.run_average(x)
from binn3_eat.edge_detector import *
import torch

class AddEdgeMap:
    '''This is a custom class to facilitate use of the Borji model with the robustness distribution script'''
    def __call__(self, img):
        edge_map = detect_edge_new_cifar(img)

        if not isinstance(edge_map, torch.Tensor):
            edge_map = torch.tensor(edge_map, dtype=torch.float32)

        return torch.cat((img, edge_map[None]), dim=0)
    

class RedetectWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = x.squeeze(0)
        edge_map = torch.as_tensor(
            detect_edge_new_cifar(x[:-1].detach()),
            dtype=torch.float32
        ).to(x.device)
        x = torch.cat([x[:-1], edge_map[None]], dim=0)
        return self.model(x.unsqueeze(0))


from edge_detector import *
import torch

class AddEdgeMap:
    '''This is a custom class to facilitate use of the Borji model with the robustness distribution script'''
    def __call__(self, img):
        edge_map = detect_edge_gtsrb(img)

        if not isinstance(edge_map, torch.Tensor):
            edge_map = torch.tensor(edge_map, dtype=torch.float32)

        return torch.cat((img, edge_map[None]), dim=0)
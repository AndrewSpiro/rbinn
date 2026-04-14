import torch

def models_are_equal(model1, model2):
    # Get the state dicts
    sd1 = torch.load(model1)
    sd2 = torch.load(model2)

    # check if same number of layers/buffers
    if len(sd1) != len(sd2):
        return False, "Different number of layers."

    # check each key and tensor value
    for key in sd1:
        if key not in sd2:
            return False, f"Key {key} missing in model 2."
        
        # torch.equal checks bit-for-bit equality
        if not torch.equal(sd1[key], sd2[key]):
            return False, f"Difference found in layer: {key}"

    return True, "Models are identical."

model_a = 'models/huang_CNNF_2_cifar.pt'
model_b = 'models/CNNF_2_cifar.pt'
result, message = models_are_equal(model_a, model_b)
print(message)
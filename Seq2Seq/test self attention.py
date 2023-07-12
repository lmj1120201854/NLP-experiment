from model import Encoder
import torch

import warnings
warnings.filterwarnings("ignore")

y_tensor = torch.ones((256, 512))
h_tensor = torch.ones((1, 256, 256))
test_tensor = torch.ones((256, 20, 256)).transpose(0, 1)

print(test_tensor.size())
# x = encoder(test_tensor)

encoder2 = DecoderWithMask(256, 256, 256, 512, 256)
x2 = encoder2(y_tensor, h_tensor, test_tensor, 3)




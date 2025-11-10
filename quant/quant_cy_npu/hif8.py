import torch
import torch_npu
from quant_cy_npu import QType, quant_dequant_float
import numpy as np

np.random.seed(42)

N = 1024
M = 1024

x = (0.2*np.random.randn(M,N) + np.random.uniform(-0.03,0.04,(M,N))).astype(np.float32)
# x = np.ones([16, 16]).astype(np.float32) * 0.052599
x_torch = torch.from_numpy(x)

qtype = QType('hif8')

# y1 = To_F8.To_HiF8(x)
y1 = quant_dequant_float(x_torch, qtype, force_py=True).cpu().numpy()
y2 = quant_dequant_float(x_torch.npu(), qtype, force_py=False).cpu().numpy()
# print(y1)
# print(y2)

diff = np.abs(y1 - y2)
print('DIFF MAX: ', diff.max())
# arg = diff.flatten().argmax()
# print(x.flatten()[arg], y1.flatten()[arg], y2.flatten()[arg])
# print(x[:,0])
# print(y1[:,0])
# print(y2[:,0])
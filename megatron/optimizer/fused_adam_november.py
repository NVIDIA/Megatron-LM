#
# Author:		Kwangryeol Park
# Email:		pkr7098@gmail.com
# File name:	fused_adam_november.py
# Repo:			https://github.com/KwangryeolPark
# Created on:	Fri Sep 15 2023
# Modified on:	Fri Sep 15 2023 9:59:38 AM
# Description:	
#
# Copyright (c) 2023 Kwangryeol Park All Rights Reserved.
#

import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FusedNovember(torch.optim.Optimizer):
    
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999),
                 eps=1e-8, weight_decay=0., decay_rate=-0.5, vector_reshape=True,
                 lambda_factor=0.999):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid momentum: {0}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta: {0}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))
        if not decay_rate <= 0:
            raise ValueError("Invalid decay rate: {0}".format(decay_rate))
        if not 0.0 <= lambda_factor < 1.0:
            raise ValueError("Invalid lambda factor: {0}".format(lambda_factor))
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, decay_rate=decay_rate,
                        vector_reshape=vector_reshape, lambda_factor=lambda_factor)
        super(FusedNovember, self).__init__(params, defaults)
        
        print(1111111111)
        if multi_tensor_applier.available:
            import amp_C
            self._dummy_overflow_buf = torch.cuda.IntTensor([0])
            self.multi_tensor_adam = amp_C.multi_tensor_adam
            print(1)
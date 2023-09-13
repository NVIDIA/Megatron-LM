# Adam과 AMSGrad 논문에 따라 \beta_1과 \beta_2를 scheduling합니다.
# \beta_1t는 \beta_1 * \lambda ^(t-1)이고, \beta_2t는 (1 - 1/t)입니다.
# AMSGrad에서 제공된 \beta_2t의 scheduling 방식은 Adafactor와 유사합니다.
# \beta_1t의 decay 방식은 \beta_1 / t를 사용해도 된다고 함. -> 이건 다음 버전에서 해보자.
# 해당 알고리즘은 AdamNC를 사용합니다. 
# AdamNC는 AMSGrad에 나온 알고리즘입니다. -> Bias correction이 없음.

import torch
import math
from torch.optim import Optimizer

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-07,
        weight_decay=0.0,
        **kwargs
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {0}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid momentum: {0}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta: {0}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps: {0}".format(eps))
        
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "decay_rate": kwargs["decay_rate"],
            "vector_reshape": kwargs["vector_reshape"],
            "lambda_factor": kwargs["lambda_factor"]
        }
        super(Adam, self).__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            # beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            vector_reshape = group['vector_reshape']
            lambda_factor = group['lambda_factor']
            decay_rate = group['decay_rate']
            
            for index, params in enumerate(group['params']):
                grad = params.grad
                if params is None or grad is None:
                    continue
                
                if weight_decay != 0.0:
                    grad = grad.add(params, alpha=weight_decay)
                    
                state = self.state[params]
                original_shape = grad.shape
                rank = len(grad.squeeze().shape)
                
                #   Option에 따라 reshape을 할 지 말 지를 결정합니다.
                factorization = not (rank == 1 and (not vector_reshape))
                if not factorization:  #   Original form
                    effective_shape = original_shape
                else:   #   Shape deformation
                    effective_shape = self._get_effective_shape(params.numel())
                    if effective_shape[0] == params.numel():
                       effective_shape = original_shape
                       factorization = False
                    else:
                        grad = grad.contiguous().view(effective_shape)
                
                #   초기 설정
                if len(state) == 0:
                    state['step'] = 1.
                    self._init_auxiliary_values(state, effective_shape, params.device, factorization)

                beta_1 = group['betas'][0] * lambda_factor ** (state['step'] - 1)
                beta_2 = 1 - state['step'] ** (decay_rate)
                                
                if not factorization:
                    state['momentum_m'].mul_(beta_1).add_(grad, alpha=(1. - beta_1))
                    state['momentum_v'].mul_(beta_2).add_(grad**2, alpha=(1. - beta_2))
                    update_m = state['momentum_m']
                    update_v = state['momentum_v']
                else:
                    update_m = self._unnmf(row_col=state['momentum_m'])
                    torch.where(state['sign'], update_m, -update_m, out=update_m)
                    update_m.mul_(beta_1).add_(grad, alpha=(1. - beta_1))
                    self._nnmf(update_m, out=state['momentum_m'])
                    state['sign'] = update_m > 0
                    
                    update_v = self._unnmf(row_col=state['momentum_v'])
                    update_v.mul_(beta_2).add_(grad**2, alpha=(1. - beta_2))
                    self._nnmf(update_v, out=state['momentum_v'])
                    
                    update_m = update_m.contiguous().view(original_shape)
                    update_v = update_v.contiguous().view(original_shape)
                
                update_v = 1 / (torch.sqrt(update_v) + eps)
                update = torch.mul(update_m, update_v)
                if update.isnan().any():
                    print("\n\n\nNaN\n\n\n")
                    exit(0)
                if update.isinf().any():
                    print("\n\n\Inf\n\n\n")
                    exit()
                
                params.sub_(update * group['lr'])
                state['step'] += 1.
        return loss
    
    @torch.no_grad()
    def _nnmf(
        self,
        matrix:torch.Tensor,
        out:tuple=None
    )->tuple:
        matrix = abs(matrix)
        if out == None:
            out[0] = torch.empty(matrix.shape[0], device=matrix.device)
            out[1] = torch.empty(matrix.shape[1], device=matrix.device)
        torch.sum(matrix, dim=1, out=out[0])  #   row sum
        torch.sum(matrix, dim=0, out=out[1])  #   col sum
        
        scale = out[0].sum()
        if scale != 0:
            torch.div(out[1], scale, out=out[1])
        return (out[0], out[1])

    @torch.no_grad()
    def _unnmf(
        self,
        row_col:tuple,
        out:torch.Tensor=None
    ):
        if out == None:
            out = torch.empty(row_col[0].shape[0], row_col[1].shape[0], device=row_col[0].device)
        torch.outer(row_col[0], row_col[1], out=out)
        return out
    
    def _get_effective_shape(
        self,
        numel:int
    )->tuple:
        assert type(numel) == int
        factor = int(numel ** 0.5) ** 2
        
        if numel == factor: #   square
            factor = int(numel ** 0.5)
            return (factor, factor)
        else:
            for i in reversed(range(1, int(numel ** 0.5) + 1)):
                if numel % i == 0:
                    return (numel // i, i)
        return (numel, 1)

    def _init_auxiliary_values(
        self,
        state:dict,
        shape:torch.Size,
        device:str,
        factorization:bool
    ):
        if not factorization:
            state['momentum_m']:torch.Tensor = torch.zeros(shape).to(device)
            state['momentum_v']:torch.Tensor = torch.zeros(shape).to(device)
        else:
            state['momentum_m']:tuple = (
                torch.zeros(shape[0]).to(device),
                torch.zeros(shape[1]).to(device)
            )
            state['momentum_v']:tuple = (
                torch.zeros(shape[0]).to(device),
                torch.zeros(shape[1]).to(device)
            )
            state['sign'] = torch.zeros(shape, dtype=bool, device=device)
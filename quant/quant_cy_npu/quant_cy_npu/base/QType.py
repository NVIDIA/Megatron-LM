import re
from copy import deepcopy


class QType():
    # declare datatype and default values
    desc: str
    exp_bits: int = -1
    man_bits: int = -1
    k_bits: int = -1
    k_outer_bits: int = 0
    blk_size: int = 1
    blk_outer_size: int = 1
    exp_max: int = -1
    exp_min: int = -1
    k_max: int = -1
    fp_val_max: float = -1
    q_dim: int = -1
    man_shift_bit: int = -1
    exp_offset: int = 0
    do_carry: bool = True

    def __init__(self, desc: str):
        # some special ones
        self.desc = desc
        if desc in ['fp16', 'fp32', 'bf16', 'int8sym', 'hif8']:
            pass
        elif desc.lower()=='nvf4':
            self.blk_size = 16
        else:
            # convert special types to universal type representation
            if desc.lower()=='mxfp4':
                desc = 'e2m1k8b32c'
            elif desc.lower()=='mxfp6e3m2':
                desc = 'e2m3k8b32c'
            elif desc.lower()=='mxfp8e4m3':
                desc = 'e4m3k8b32c'
            elif desc.lower()=='mxfp8e5m2':
                desc = 'e5m2k8b32c'
            elif desc.lower()[:4]=='hifx':
                res = re.match(r'^hifx([2345])$', desc.lower())
                if res is None:
                    raise ValueError('HiFx only supports hifx*')
                else:
                    n_bit_tmp = int(res.group(1)) - 1
                    desc = 'e0m%dK1k4B1b%dCoff38'%(n_bit_tmp, 64)

            # start to parse
            res = re.match(r'^e([0-9]+)m([0-9]+)k([0-9]+)b([0-9]+)([Cc]?)$', desc)
            res2 = re.match(r'^e([0-9]+)m([0-9]+)K([0-9]+)k([0-9]+)B([0-9]+)b([0-9]+)([Cc]?)(off[0-9]+)?$', desc)
            if res is not None:
                self.exp_bits = int(res.group(1))
                self.man_bits = int(res.group(2))
                self.k_bits = int(res.group(3))
                self.blk_size = int(res.group(4))
                offset_number = None
                if res.group(5) is None:
                    self.do_carry = False
                else:
                    self.do_carry = str(res.group(5)) == 'C'
            elif res2 is not None:
                self.exp_bits = int(res2.group(1))
                self.man_bits = int(res2.group(2))
                self.k_outer_bits = int(res2.group(3))
                self.k_bits = int(res2.group(4))
                self.blk_outer_size = int(res2.group(5))
                self.blk_size = int(res2.group(6))
                if res2.group(7) is None:
                    self.do_carry = False
                else:
                    self.do_carry = str(res2.group(7)) == 'C'
                if res2.group(8) is not None:
                    offset_number = int(res2.group(8)[3:])
                else:
                    offset_number = None
            else:
                raise ValueError('Quant type string must be [e*m*k*b* or e*m*K*k*B*b*], or special float types [fp16, fp32, bf16, int8sym]')

            assert self.exp_bits!=1, 'exp_bits==1 is not supported. E1Mx is equivalent to E0M(x+1)'
            assert self.man_bits>=1, 'man_bits should >=1'

            # compute exp max and min
            if self.exp_bits==0:
                self.exp_max = self.man_bits - 1
                self.exp_min = 0
            else:
                self.exp_max = 2 ** (self.exp_bits - 1)
                # special case for e5m2
                if self.exp_bits==5 and self.man_bits==2:
                    self.exp_max -= 1
                self.exp_min = - 2 ** (self.exp_bits - 1) + 2

            # compute k range
            self.k_max = 2 ** (self.k_bits + self.k_outer_bits - 1) - 1

            # compute exp offset
            if offset_number is None:
                self.exp_offset = self.exp_max
            else:
                self.exp_offset = offset_number - self.k_max - 1 + self.exp_max

            # compute shift bits
            self.man_shift_bit = self.man_bits

            # compute fp value max
            if self.exp_bits==4 and self.man_bits==3:
                self.fp_val_max = 448  # special case for e4m3
            else:
                self.fp_val_max = 2**self.exp_max * float(2**(self.man_bits+1) - 1) / 2**(self.man_bits)
                self.fp_val_max = min(self.fp_val_max, 1e38)   # approx float max

    def dim_(self, dim: int):
        # inplace function
        self.q_dim = dim
        return self

    def dim(self, dim: int):
        out = deepcopy(self)
        out.q_dim = dim
        return out

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return f'QType: {self.desc}   Dim: {self.q_dim}  ExpOffset: {self.exp_offset}'


if __name__=='__main__':
    from copy import deepcopy
    t = QType('e2m1k8b8')
    t2 = deepcopy(t)
    print()

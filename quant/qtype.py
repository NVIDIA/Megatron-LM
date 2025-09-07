import re
from copy import deepcopy

class QType:
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
        self.desc = desc

        if desc in ['fp16', 'fp32', 'bf16', 'int8sym', 'hif8']:
            pass  # 保留默认配置
        elif 'nf4' in desc.lower():
            res = re.match(r'^nf4B([0-9]+)b([0-9]+)$', desc)
            if res is None:
                raise ValueError("Quant type string must be [nf4B*b*]")
            self.blk_outer_size = int(res.group(1))
            self.blk_size = int(res.group(2))
        elif desc.lower()[:3]=='tmx':
            res = re.match(r"^tmx([0-9]+)$", desc)
            self.blk_size = 16
            self.man_bits = int(res.group(1)) - 2
        elif desc.lower() == 'nvf4':
            self.blk_size = 256  # padding to 256 to align kernel batch size
        else:
            if desc.lower() == 'mxfp4':
                desc = 'e2m1k8b32c'
            elif desc.lower() == 'mxfp6e3m2':
                desc = 'e3m2k8b32c'
            elif desc.lower() == 'mxfp8e4m3':
                desc = 'e4m3k8b32c'
            elif desc.lower() == 'mxfp8e5m2':
                desc = 'e5m2k8b32c'
            elif desc.lower()[:4] == 'hifx' and desc.lower()[-3:] == 'v12':
                res = re.match(r"^hifx([2345])_v12$", desc.lower())
                if res is None:
                    raise ValueError("HiFx only supports hifx[2-5]_v12")
                n_bit_tmp = int(res.group(1)) - 1
                desc = f"e0m{n_bit_tmp}k1k4B1b{n_bit_tmp}Coff38"

        res = re.match(r"^e([0-9]+)m([0-9]+)k([0-9]+)b([0-9]+)([Cc]?)$", desc)
        res2 = re.match(r"^e([0-9]+)m([0-9]+)K([0-9]+)k([0-9]+)B([0-9]+)b([0-9]+)([Cc]?)(off[0-9]+)?$", desc)

        if res is not None:
            self.exp_bits = int(res.group(1))
            self.man_bits = int(res.group(2))
            self.k_bits = int(res.group(3))
            self.blk_size = int(res.group(4))
            offset_number = None
            if res.group(5) is None:
                self.do_carry = False
            else:
                self.do_carry = str(res.group(5)).upper() == 'C'
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
                self.do_carry = str(res2.group(7)).upper() == 'C'
            if res2.group(8) is not None:
                offset_number = int(res2.group(8)[3:])
            else:
                offset_number = None
        else:
            raise ValueError(
                "Quant type string must be like 'e<m>k<b>' or 'e<m>kk<B>b<b>', or special float types [fp16, fp32, bf16, int8sym]"
            )

        assert self.exp_bits !=1, "exp_bits==1 is not supported. E0M(x) is equivalent to E1M(x-1)"
        assert self.man_bits >= 1, "man_bits should >= 1"

        if self.exp_bits == 0:
            self.exp_max = self.man_bits - 1
            self.exp_min = 0
        else:
            self.exp_max = 2 ** (self.exp_bits - 1) 
            if self.exp_bits == 5 and self.man_bits == 2:
                self.exp_max-=1
            self.exp_min = -2 ** (self.exp_bits - 1) + 2


        self.k_max = 2 ** (self.k_bits + self.k_outer_bits - 1) - 1

        if offset_number is None:
            self.exp_offset = self.exp_max
        else:
            self.exp_offset = offset_number - self.k_max - 1 + self.exp_max

        # 计算 shift bits
        self.man_shift_bit = self.man_bits

        # 计算 fp_val_max
        if self.exp_bits == 4 and self.man_bits == 3:
            self.fp_val_max = 448  # 特殊处理 e4m3
        else:
            self.fp_val_max = 2 ** self.exp_max * float(2 ** (self.man_bits + 1) - 1) / (2 ** self.man_bits)
            self.fp_val_max = min(self.fp_val_max, 1e38)  # 近似 float max

    def dim_(self, dim: int):
        """in-place function"""
        self.q_dim = dim
        return self

    def dim(self, dim: int):
        """non-in-place function"""
        out = deepcopy(self)
        out.q_dim = dim
        return out

    def copy(self):
        return deepcopy(self)

    def __repr__(self) -> str:
        return f'QType: {self.desc} Dim: {self.q_dim} ExpOffset: {self.exp_offset}'

# 示例用法
if __name__ == "__main__":
    t = QType("e2m1k8b8")
    t2 = deepcopy(t)
    print(t)

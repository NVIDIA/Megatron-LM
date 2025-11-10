### MXFP 仿真量化实现综述

- **入口函数 `_quantize_mx`**  
  - 负责将输入张量 `A` 按指定量化格式执行共享指数的块式量化。流程包括：参数整理、可选的块重排、指数计算、mantissa 量化以及还原原始形状。  
  - 若 `elem_format` 为空直接返回原张量；`axes` 支持负索引并在 block 模式下自动扩维；`scale_bits` 决定外部量化尺度寄存器宽度。  
  - 通过 `_get_format_params` 拿到格式相关的 `ebits/mbits/emax/max_norm`，之后走共享指数和逐元素量化路径。  
```365:437:quant/mxfp_ori.py
def _quantize_mx(
    A,
    scale_bits,
    elem_format,
    shared_exp_method="max",
    axes=None,
    block_size=0,
    round="nearest",
    flush_fp32_subnorms=False,
    scaling_control="max",
):
    if elem_format == None:
        return A
    ...
    shared_exp_axes = [x + 1 for x in axes] if block_size > 0 else axes
    shared_exp = _shared_exponents(...)
    ...
    A = _quantize_elemwise_core(...)
    ...
    if block_size:
        A = _undo_reshape_to_blocks(...)
    return A
```

- **数据格式解析 `_get_format_params`**  
  - 使用枚举 `ElemFormat` 定义受支持的 MXFP/整型/浮点格式，输出量化所需的指数位、尾数位、最大/最小正规数等参数。  
  - 支持缓存，避免多次解析相同格式。  
```49:114:quant/mxfp_ori.py
def _get_format_params(fmt):
    if type(fmt) is str:
        fmt = ElemFormat.from_str(fmt)
    if fmt in _FORMAT_CACHE:
        return _FORMAT_CACHE[fmt]
    ...
    _FORMAT_CACHE[fmt] = (ebits, mbits, emax, max_norm, min_norm)
    return ebits, mbits, emax, max_norm, min_norm
```

### 块式量化管线

- **块维度重排 `_reshape_to_blocks`**  
  - 针对给定轴做 tiling：为每个 axis 插入一个块内维度，确保后续共享指数在同一块内计算。  
  - 需要时在块维度上做 padding，使长度为 `block_size` 的整数倍。  
  - 返回新张量、更新后的轴索引、原始形状和填充后的形状。  
```290:349:quant/mxfp_ori.py
axes = [(x + len(A.shape) if x < 0 else x) for x in axes]
...
for i in range(len(axes)):
    axes[i] += i
    A = torch.unsqueeze(A, dim=axes[i] + 1)
...
pad = list(reversed(pad))
A = torch.nn.functional.pad(A, pad, mode="constant")
...
A = A.view(reshape)
return A, axes, orig_shape, padded_shape
```

- **共享指数 `_shared_exponents`**  
  - 默认 `method="max"`，对指定 `axes` 取绝对值最大值求 `log2` 并下取整。  
  - 支持策略 `scaling_control`，可用 `max_minus_1` 避免溢出。  
  - `ebits>0` 时限制指数范围并对溢出/下溢做 NaN 或钳制。  
```236:287:quant/mxfp_ori.py
if axes is None:
    max_val = torch.max(torch.abs(A))
else:
    shared_exp = A
    for axis in axes:
        shared_exp, _ = torch.max(torch.abs(shared_exp), dim=axis, keepdim=True)
...
shared_exp = torch.floor(torch.log2(...))
if ebits > 0:
    emax = 2**(ebits-1) - 1
    shared_exp[shared_exp > emax] = float("NaN")
    shared_exp[shared_exp < -emax] = -emax
```

- **指数与尺度寄存器**  
  - `_quantize_mx` 在拿到共享指数后减去目标格式的最大指数 `emax`，保证转换到局部 scaling 范围。  
  - 再根据 `scale_bits` 将指数裁剪到可表示范围，超过正上限设为 NaN，低于负上限饱和。  

- **元素级量化 `_quantize_elemwise_core`**  
  - 统一入口，同时覆盖纯整型(`exp_bits=0`)与浮点(`exp_bits>0`)路径；前者直接对整数尾数做裁剪，后者则先提取一个“私有”指数用于尺度对齐（`quant/mxfp_ori.py` 第198-205行）。  
  - 浮点路径：`torch.floor(torch.log2(torch.abs(A)))` 在第199行计算 `private_exp`，再通过 `_safe_lshift` 把尾数左移 `bits-2` 位（第209行），让有效精度进入整数区间；如果指数低于格式允许的最小值 `min_exp` 会被钳制（第203-205行），保障 denorm 处理一致。  
  - 舍入阶段由 `_round_mantissa` 完成（第211行），支持 `nearest/even/floor/dither`；`dither` 会注入 U[0,1) 噪声用于抖动量化误差。  
  - `_safe_rshift` 把尾数移回原尺度（第214行），如果 `saturate_normals=True` 或是定点量化，随后用 `torch.clamp` 将绝对值限制到 `max_norm` 之内（第217-219行）。  
  - 稀疏 COO 张量会先解稀疏、量化 values，再在尾部重建稀疏张量（第183-231行）；当前实现里重建时引用的 `output` 未定义，若启用稀疏路径需补全返回值修复该 bug。  
```165:233:quant/mxfp_ori.py
if exp_bits != 0:
    private_exp = torch.floor(torch.log2(torch.abs(A) + ...))
    min_exp = -(2**(exp_bits-1)) + 2
    private_exp = private_exp.clip(min=min_exp)
...
out = _safe_lshift(out, bits - 2, private_exp)
out = _round_mantissa(out, bits, round, clamp=False)
out = _safe_rshift(out, bits - 2, private_exp)
if saturate_normals or exp_bits == 0:
    out = torch.clamp(out, min=-max_norm, max=max_norm)
```

- **块后处理 `_undo_reshape_to_blocks`**  
  - 把张量从块形式还原：先 view 回 `padded_shape`，切掉 padding，再移除额外维度。  
```352:362:quant/mxfp_ori.py
A = A.view(padded_shape)
if not list(padded_shape) == list(orig_shape):
    slices = [slice(0, x) for x in orig_shape]
    A = A[slices]
for axis in reversed(axes):
    A = torch.squeeze(A, dim=axis + 1)
return A
```

### 关键设计要点

- **指数共享轴的调整**：块模式下 `_reshape_to_blocks` 会在每个目标轴后插入块内维度，因此共享指数时必须对轴索引 `+1`，使指数在同一块内广播。否则指数会跨块计算导致错误量化。  
- **NaN 标记溢出**：当共享指数超出 `scale_bits` 允许范围时置 NaN，可在后续分析时定位溢出块。  
- **自适应舍入策略**：`round` 参数允许不同硬件或算法需求；`dither` 支持噪声抖动以减轻量化误差。  
- **稀疏张量处理**：在 `_quantize_elemwise_core` 中对 COO 布局提供特殊路径（但输出处存在 `output` 未定义的潜在 bug，若启用需补写）。

### 典型使用 (`__main__` 示例)

- 构造 `1024×1024` 张量，使用 `scale_bits=8`、`elem_format='fp8_e4m3'`、共享轴 `-1`、`block_size=32`，执行量化并打印均方误差，可作为快速单元测试。  
```440:449:quant/mxfp_ori.py
if __name__ == "__main__":
    A = torch.randn(1024, 1024).cuda()
    A_q = _quantize_mx(... axes=-1, block_size=32, ...)
    print(A_q)
    loss = torch.mean((A - A_q) ** 2)
    print(loss)
```

### 建议

- 若继续扩展到 OCP 实际硬件，应补充 `round` 策略与硬件舍入模式的映射、处理稀疏张量的 bug，并提供单元测试覆盖 `block_size`/`axes` 的组合情况。
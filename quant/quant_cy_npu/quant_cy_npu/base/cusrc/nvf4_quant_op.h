#include "tensorutils.h"


struct NVF4ShapeInfo{
    int M, N, N_ceil;
    int mn0, mn1;
};


template <typename T>
class NVF4VecKernel{
public:
    aifunc NVF4VecKernel(){}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR out_, int M_, int N_){
        shape.M = M_;
        shape.N = N_;
        tiling();

        // assign global tensors
        xmtx = Tensor<T, PGM>(xmtx_);
        out = Tensor<T, PGM>(out_);

        // assign in/out buffer
        int offset = 0;
        xbuf = DBuff<T, PUB>(0, BATCH, offset);
        outbuf = DBuff<T, PUB>(0, BATCH, offset);

        //
        dupxbuf = Tensor<float, PUB>(0, BATCH*4, offset);
        absbuf = Tensor<float, PUB>(0, BATCH*4, offset);
        sfbuf = Tensor<float, PUB>(0, BATCH/16, offset);
        sfexpbuf = Tensor<float, PUB>(0, BATCH/16, offset);
        brcbbuf = Tensor<float, PUB>(0, BATCH/2, offset);
        sfbrcbbuf = Tensor<float, PUB>(0, BATCH*4, offset);
        innerexpbuf = Tensor<float, PUB>(0, BATCH*4, offset);

        fp32inbuf = Tensor<float, PUB>(0, BATCH, offset);
        fp32outbuf = Tensor<float, PUB>(0, BATCH, offset);

        expmask = Tensor<uint32_t, PUB>(0, 64, offset);
        vector_dup(expmask.ptr(), 0x7F800000, 1, 1, 1, 8, 8);
        vector_dup(dupxbuf.ptr(), 0.0f, BATCH*4/64, 1, 1, 8, 8);
        // printf("UB USED: %d KB\n", offset/1024);
    }

    aifunc void tiling(){
        shape.N_ceil = (shape.N + BATCH - 1) / BATCH * BATCH;
        int total_split = shape.M * shape.N_ceil / BATCH;
        int split_per_core = (total_split + GetBlockNum() - 1) / GetBlockNum();
        shape.mn0 = split_per_core * GetBlockIdx();
        shape.mn1 = split_per_core + shape.mn0;
        if (shape.mn1 > total_split){
            shape.mn1 = total_split;
        }
    }

    aifunc void Compute(){
        input_empty.setall();
        output_empty.setall();
        for (int mn=shape.mn0; mn<shape.mn1; ++mn){
            Compute(mn);
        }
        input_empty.release();
        output_empty.release();
    }

    aifunc void Compute(int mn){
        Tensor<float, PUB> inp_tsr;
        Tensor<float, PUB> out_tsr;

        int m = mn / (shape.N_ceil / BATCH);
        int n = mn % (shape.N_ceil / BATCH);
        int n_tail = (n+1) * BATCH - shape.N;
        input_empty.wait();
        copy_gm_to_ubuf(xbuf.get(mn).vptr(), xmtx[m*shape.N + n*BATCH].vptr(), 0, 1, BATCH*sizeof(T)/32, 0, 0);
        input_ready.set();

        output_empty.wait();
        input_ready.wait();
        if constexpr(std::is_same<T, float>::value){
            inp_tsr = xbuf.get(mn);
            out_tsr = outbuf.get(mn);
        }else{
            inp_tsr = fp32inbuf;
            out_tsr = fp32outbuf;
            vconv_bf162f32(inp_tsr.ptr(), xbuf.get(mn).ptr(), BATCH/64, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }

        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        if (n_tail>0){
            // if tail, mask out the unused numbers
            for (int tail=n_tail; tail>0; tail-=64){
                if (tail<64){
                    // uint64_t mask = 0x8000000000000000;
                    uint64_t mask = 1;
                    for (int i=1;i<tail;++i){
                        mask |= (mask << 1);
                    }
                    set_vector_mask(0, mask);
                }
                pipe_barrier(PIPE_V);
                vector_dup(inp_tsr[BATCH-64].ptr(), 0.0f, 1, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
            }
        }
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);

        // copy to make it twice larger
        copy_ubuf_to_ubuf(dupxbuf.vptr(), inp_tsr.vptr(), 0, BATCH/16, 16/8, 0, 48/8);
        pipe_barrier(PIPE_V);
        // get abs value for scale factor
        vabs(absbuf.ptr(), dupxbuf.ptr(), BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // get group max
        vcmax(sfbuf.ptr(), absbuf.ptr(), BATCH*4/64, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        // div by 6.0 for scale factor
        vmuls(sfbuf.ptr(), sfbuf.ptr(), 1.0f/6.0f, BATCH/16/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmins(sfbuf.ptr(), sfbuf.ptr(), 448.0f, BATCH/16/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // get exp for scale factor
        vand((__ubuf__ uint16_t*)sfexpbuf.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), (__ubuf__ uint16_t*)sfbuf.ptr(), BATCH/16/64, 1, 1, 1, 8, 0, 8);
        pipe_barrier(PIPE_V);
        vmaxs(sfexpbuf.ptr(), sfexpbuf.ptr(), 0.015625f, BATCH/16/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(sfexpbuf.ptr(), sfexpbuf.ptr(), 0.125f, BATCH/16/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // get mantissa and quantized number
        vdiv(sfbuf.ptr(), sfbuf.ptr(), sfexpbuf.ptr(), BATCH/16/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32r(sfbuf.ptr(), sfbuf.ptr(), BATCH/16/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(sfbuf.ptr(), sfbuf.ptr(), sfexpbuf.ptr(), BATCH/16/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // broadcast sf to original length
        vbrcb((__ubuf__ uint32_t*)brcbbuf.ptr(), (__ubuf__ uint32_t*)sfbuf.ptr(), 1, 8, BATCH/2/64);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*)sfbrcbbuf.ptr(), (__ubuf__ uint32_t*)brcbbuf.ptr(), 1, 8, BATCH*4/64);
        pipe_barrier(PIPE_V);
        vadds(sfbrcbbuf.ptr(), sfbrcbbuf.ptr(), (float)eps, BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // div scale to get inner group value
        vdiv(dupxbuf.ptr(), dupxbuf.ptr(), sfbrcbbuf.ptr(), BATCH*4/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // get exp value
        vand((__ubuf__ uint16_t*)innerexpbuf.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), (__ubuf__ uint16_t*)dupxbuf.ptr(), BATCH*4/64, 1, 1, 1, 8, 0, 8);
        pipe_barrier(PIPE_V);
        // clip to minimum 1.0
        vmaxs(innerexpbuf.ptr(), innerexpbuf.ptr(), 1.0f, BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(innerexpbuf.ptr(), innerexpbuf.ptr(), 0.5f, BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // get mantissa and quantized number
        vdiv(dupxbuf.ptr(), dupxbuf.ptr(), innerexpbuf.ptr(), BATCH*4/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32r(dupxbuf.ptr(), dupxbuf.ptr(), BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(dupxbuf.ptr(), dupxbuf.ptr(), innerexpbuf.ptr(), BATCH*4/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // clip inner value within -6~6
        vmaxs(dupxbuf.ptr(), dupxbuf.ptr(), -6.0f, BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmins(dupxbuf.ptr(), dupxbuf.ptr(), 6.0f, BATCH*4/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // final value = inner value * scale factor
        vmul(sfbrcbbuf.ptr(), dupxbuf.ptr(), sfbrcbbuf.ptr(), BATCH*4/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        copy_ubuf_to_ubuf(out_tsr.vptr(), sfbrcbbuf.vptr(), 0, BATCH/16, 16/8, 48/8, 0);
        pipe_barrier(PIPE_V);

        if constexpr(std::is_same<T, float>::value){
        }else{
            vconv_f322bf16a(outbuf.get(mn).ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
        }


        output_ready.set();
        input_empty.set();

        output_ready.wait();
        // mte3
        if (n_tail>0){
            copy_ubuf_to_gm(out[m*shape.N + n*BATCH].vptr(), outbuf.get(mn).vptr(), 0, 1, sizeof(T)*(BATCH-n_tail), 0, 0, BM_ENABLE);
        }else{
            copy_ubuf_to_gm(out[m*shape.N + n*BATCH].vptr(), outbuf.get(mn).vptr(), 0, 1, BATCH*sizeof(T)/32, 0, 0);
        }
        output_empty.set();
    }


private:
    TPipe pipe;
    NVF4ShapeInfo shape;
    Tensor<T, PGM> xmtx, out;
    DBuff<T, PUB> xbuf, outbuf;
    Tensor<float, PUB> dupxbuf, absbuf, sfbuf, sfexpbuf, brcbbuf, sfbrcbbuf, innerexpbuf;
    Tensor<float, PUB> fp32inbuf, fp32outbuf;
    Tensor<uint32_t, PUB> expmask;
    DEvent<PIPE_MTE2, PIPE_V> input_ready{3,4};
    DEvent<PIPE_V, PIPE_MTE2> input_empty{3,4};
    DEvent<PIPE_V, PIPE_MTE3> output_ready{3,4};
    DEvent<PIPE_MTE3, PIPE_V> output_empty{3,4};

    static constexpr int BATCH = 1024;
    static constexpr float eps = 5.421011e-20;
};


extern "C" __global__ __aicore__ void nvf4_kernel(GM_ADDR xmtx, GM_ADDR out, int M, int N){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        NVF4VecKernel<float> vec;
        vec.Init(xmtx, out, M, N);
        vec.Compute();
#endif
    }
}

extern "C" __global__ __aicore__ void nvf4_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int M, int N){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        NVF4VecKernel<bfloat16_t> vec;
        vec.Init(xmtx, out, M, N);
        vec.Compute();
#endif
    }
}

void run_nvf4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N) {
    nvf4_kernel<<<40, nullptr, stream>>>(xmtx, out, M, N);
}

void run_nvf4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N) {
    nvf4_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, M, N);
}
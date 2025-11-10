#include "tensorutils.h"


struct MXFPShapeInfo{
    int M, N, N_ceil;
    int mn0, mn1;
};


template <typename T>
class MXFPVecKernel{
public:
    aifunc MXFPVecKernel(){}
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
        dupxbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        absbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        expbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        expmaxbuf = Tensor<float, PUB>(0, 64, offset);
        grpexpbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        expmaxbrcbbuf = Tensor<float, PUB>(0, 64*64, offset);
        privexpbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        upperbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        lowerbuf = Tensor<float, PUB>(0, BATCH*2, offset);
        mant = Tensor<float, PUB>(0, BATCH*2, offset);
        fp32inbuf = Tensor<float, PUB>(0, BATCH, offset);
        fp32outbuf = Tensor<float, PUB>(0, BATCH, offset);

        expmask = Tensor<uint32_t, PUB>(0, 64, offset);
        vector_dup(expmask.ptr(), 0x7F800000, 1, 1, 1, 8, 8);
        vector_dup(dupxbuf.ptr(), 0.0f, BATCH*2/64, 1, 1, 8, 8);
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

    aifunc void Process(){
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
        copy_ubuf_to_ubuf(dupxbuf.vptr(), inp_tsr.vptr(), 0, BATCH/32, 32/8, 0, 32/8);
        // get element-wise exp
        pipe_barrier(PIPE_V);
        vabs(absbuf.ptr(), dupxbuf.ptr(), BATCH*2/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vand((__ubuf__ uint16_t*)expbuf.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), (__ubuf__ uint16_t*)absbuf.ptr(), BATCH*2/64, 1, 1, 1, 8, 0, 8);
        pipe_barrier(PIPE_V);

        // get max exp for each block
        vcmax(expmaxbuf.ptr(), expbuf.ptr(), BATCH*2/64, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*)expmaxbrcbbuf.ptr(), (__ubuf__ uint32_t*)expmaxbuf.ptr(), 1, 8, BATCH*2/64/8);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*)grpexpbuf.ptr(), (__ubuf__ uint32_t*)expmaxbrcbbuf.ptr(), 1, 8, BATCH*2/64);
        pipe_barrier(PIPE_V);

        // clip to get private exp
        vmuls(upperbuf.ptr(), grpexpbuf.ptr(), 1.5f, BATCH*2/64, 1, 1, 8, 8);
        vmuls(lowerbuf.ptr(), grpexpbuf.ptr(), -1.5f, BATCH*2/64, 1, 1, 8, 8);
        vmuls(privexpbuf.ptr(), grpexpbuf.ptr(), 0.25f, BATCH*2/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmax(privexpbuf.ptr(), privexpbuf.ptr(), expbuf.ptr(), BATCH*2/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(privexpbuf.ptr(), privexpbuf.ptr(), 0.5f, BATCH*2/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vadds(privexpbuf.ptr(), privexpbuf.ptr(), (float)eps, BATCH*2/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // // round mantissa
        vdiv(mant.ptr(), dupxbuf.ptr(), privexpbuf.ptr(), BATCH*2/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32a(mant.ptr(), mant.ptr(), BATCH*2/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // get final result (mantissa(signed) * exp)
        vmul(mant.ptr(), privexpbuf.ptr(), mant.ptr(), BATCH*2/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmin(mant.ptr(), mant.ptr(), upperbuf.ptr(), BATCH*2/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmax(mant.ptr(), mant.ptr(), lowerbuf.ptr(), BATCH*2/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);


        copy_ubuf_to_ubuf(out_tsr.vptr(), mant.vptr(), 0, BATCH/32, 32/8, 32/8, 0);
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
    MXFPShapeInfo shape;
    Tensor<T, PGM> xmtx, out;
    DBuff<T, PUB> xbuf, outbuf;
    Tensor<float, PUB> dupxbuf, absbuf, expbuf, grpexpbuf, privexpbuf, expmaxbuf, expmaxbrcbbuf, mant, upperbuf, lowerbuf;
    Tensor<float, PUB> fp32inbuf, fp32outbuf;
    Tensor<uint32_t, PUB> expmask;
    DEvent<PIPE_MTE2, PIPE_V> input_ready{3,4};
    DEvent<PIPE_V, PIPE_MTE2> input_empty{3,4};
    DEvent<PIPE_V, PIPE_MTE3> output_ready{3,4};
    DEvent<PIPE_MTE3, PIPE_V> output_empty{3,4};

    static constexpr int BATCH = 512;
    static constexpr float eps = 5.421011e-20;
};




extern "C" __global__ __aicore__ void mxfp4_kernel(GM_ADDR xmtx, GM_ADDR out, int M, int N){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        MXFPVecKernel<float> vec;
        vec.Init(xmtx, out, M, N);
        vec.Process();
#endif
    }
}

extern "C" __global__ __aicore__ void mxfp4_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int M, int N){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        MXFPVecKernel<bfloat16_t> vec;
        vec.Init(xmtx, out, M, N);
        vec.Process();
#endif
    }
}



void run_mxfp4_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N) {
    mxfp4_kernel<<<40, nullptr, stream>>>(xmtx, out, M, N);
}

void run_mxfp4_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N) {
    mxfp4_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, M, N);
}
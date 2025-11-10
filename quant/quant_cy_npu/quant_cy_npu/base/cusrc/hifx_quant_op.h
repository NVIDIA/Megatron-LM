#include "tensorutils.h"


struct Hifxv14ShapeInfo{
    int M, N, N_ceil;
    int mn0, mn1;
    int MB;
};


template <typename T, int E6MX>
class Hifv14Kernel{
public:
    aifunc Hifv14Kernel(){}
    aifunc void Init(GM_ADDR xmtx_, GM_ADDR out_, int M_, int N_, int MB_){
        shape.M = M_;
        shape.N = N_;
        shape.MB = MB_;
        tiling();

        // assign global tensors
        xmtx = Tensor<T, PGM>(xmtx_);
        out = Tensor<T, PGM>(out_);

        // assign in/out buffer
        int offset = 0;
        xbuf = DBuff<T, PUB>(0, BATCH, offset);
        outbuf = DBuff<T, PUB>(0, BATCH, offset);

        //
        absbuf = Tensor<float, PUB>(0, BATCH, offset);
        exp0buf = Tensor<float, PUB>(0, BATCH, offset);
        exp0recbuf = Tensor<float, PUB> (0, BATCH, offset);
        brcbbuf = Tensor<float, PUB>(0, BATCH, offset);
        sfexpbuf = Tensor<float, PUB>(0, BATCH, offset);

        // for exp1
        exp1maxbuf = Tensor<float, PUB>(0, BATCH, offset);
        exp1buf = Tensor<float, PUB>(0, BATCH, offset);
        exp1maskbuf = Tensor<float, PUB>(0, BATCH, offset);  // no need batch but it's ok to allocate more

        // for exp2
        exp2parta = Tensor<float, PUB>(0, BATCH, offset);
        exp2partb = Tensor<float, PUB>(0, BATCH, offset);
        exp2maxbufa = Tensor<float, PUB>(0, BATCH, offset);
        exp2maxbufb = Tensor<float, PUB>(0, BATCH, offset);
        exp2buf = Tensor<float, PUB>(0, BATCH, offset);
        exp2maskbuf = Tensor<float, PUB>(0, BATCH, offset);  // no need batch but it's ok to allocate more

        // for mantissa
        mantbuf = Tensor<float, PUB>(0, BATCH, offset);

        fp32inbuf = Tensor<float, PUB>(0, BATCH, offset);
        fp32outbuf = Tensor<float, PUB>(0, BATCH, offset);

        expmask = Tensor<uint32_t, PUB>(0, 64, offset);
        onesbuf = Tensor<float, PUB>(0, BATCH, offset);
        twosbuf = Tensor<float, PUB>(0, 64, offset);
        bf16_seven_buf_bf16 = Tensor<bfloat16_t, PUB>(0, BATCH, offset);
        bf16_seven_buf = Tensor<float, PUB>(0, BATCH, offset);
        bf16_general_buf = Tensor<bfloat16_t, PUB>(0, BATCH, offset);

        vector_dup(expmask.ptr(), 0x7F800000, 1, 1, 1, 8, 8);
        vector_dup(onesbuf.ptr(), 1.0f, BATCH/64, 1, 1, 8, 8);
        vector_dup(twosbuf.ptr(), 2.0f, 1, 1, 1, 8, 8);
        vector_dup(bf16_seven_buf_bf16.ptr(), (bfloat16_t)(1.0f / 7.0f), BATCH/128, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(bf16_seven_buf.ptr(), bf16_seven_buf_bf16.ptr(), BATCH/64, 1, 1, 8, 4);
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
        // get element-wise exp
        pipe_barrier(PIPE_V);
        vabs(absbuf.ptr(), inp_tsr.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // get max value and find scale factor
        vcmax(exp0buf.ptr(), absbuf.ptr(), BATCH/64, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*) brcbbuf.ptr(), (__ubuf__ uint32_t*) exp0buf.ptr(), 1, 8, (BATCH+511)/512);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*) exp0buf.ptr(), (__ubuf__ uint32_t*) brcbbuf.ptr(), 1, 8, BATCH/64);
        pipe_barrier(PIPE_V);
        // vmuls(exp0buf.ptr(), exp0buf.ptr(), 1.0f/7.0f, BATCH/64, 1, 1, 8, 8);
        vmul(exp0buf.ptr(), exp0buf.ptr(), bf16_seven_buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(bf16_general_buf.ptr(), exp0buf.ptr(), BATCH/64, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(exp0buf.ptr(), bf16_general_buf.ptr(), BATCH/64, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmins(exp0buf.ptr(), exp0buf.ptr(), 49152.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        uint32_t twopowneg48_int = 0x27800000;
        float twopowneg48 = *(float*) &twopowneg48_int;
        vmaxs(exp0buf.ptr(), exp0buf.ptr(), twopowneg48, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        // convert scale factor to e6m2
        vand((__ubuf__ uint16_t*)sfexpbuf.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), (__ubuf__ uint16_t*)exp0buf.ptr(), BATCH/64, 1, 1, 1, 8, 0, 8);
        pipe_barrier(PIPE_V);
        if constexpr(E6MX==2){
            vmuls(sfexpbuf.ptr(), sfexpbuf.ptr(), 0.25f, BATCH/64, 1, 1, 8, 8);
        }else if constexpr(E6MX==1){
            vmuls(sfexpbuf.ptr(), sfexpbuf.ptr(), 0.5f, BATCH/64, 1, 1, 8, 8);
        }else{

        }
        pipe_barrier(PIPE_V);
        vdiv(exp0buf.ptr(), exp0buf.ptr(), sfexpbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32r(exp0buf.ptr(), exp0buf.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(exp0buf.ptr(), exp0buf.ptr(), sfexpbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // get exp1
        vcgmax(exp1maxbuf.ptr(), absbuf.ptr(), BATCH/64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*) exp1buf.ptr(), (__ubuf__ uint32_t*) exp1maxbuf.ptr(), 1, 8, BATCH/64);
        pipe_barrier(PIPE_V);
        // vdiv(exp1buf.ptr(), exp1buf.ptr(), exp0buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        // pipe_barrier(PIPE_V);
        vdiv(exp0recbuf.ptr(), onesbuf.ptr(), exp0buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322bf16r(bf16_general_buf.ptr(), exp0recbuf.ptr(), BATCH/64, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        vconv_bf162f32(exp0recbuf.ptr(), bf16_general_buf.ptr(), BATCH/64, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmul(exp1buf.ptr(), exp1buf.ptr(), exp0recbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vcmpvs_ge((__ubuf__ uint8_t*)exp1maskbuf.ptr(), exp1buf.ptr(), 4.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(onesbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(exp1buf.ptr(), twosbuf.ptr(), exp1maskbuf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);

        // // exp1 = exp1 * exp0
        // vmul(exp1buf.ptr(), exp1buf.ptr(), exp0buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        // pipe_barrier(PIPE_V);

        // get exp2
        // split part a and part b
        copy_ubuf_to_ubuf(exp2parta.vptr(), absbuf.vptr(), 0, 1, BATCH/8, 0, 0);
        copy_ubuf_to_ubuf(exp2partb.vptr(), absbuf.vptr(), 0, 1, BATCH/8, 0, 0);
        pipe_barrier(PIPE_V);
        // set half to zero
        set_vector_mask(0, 0x0f0f0f0f0f0f0f0f);
        pipe_barrier(PIPE_V);
        vector_dup(exp2parta.ptr(), -999999.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0xf0f0f0f0f0f0f0f0);
        pipe_barrier(PIPE_V);
        vector_dup(exp2partb.ptr(), -999999.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        // get max
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        vcgmax(exp2maxbufa.ptr(), exp2parta.ptr(), BATCH/64, 1, 1, 8);
        vcgmax(exp2maxbufb.ptr(), exp2partb.ptr(), BATCH/64, 1, 1, 8);
        pipe_barrier(PIPE_V);
        vbrcb((__ubuf__ uint32_t*) exp2parta.ptr(), (__ubuf__ uint32_t*) exp2maxbufa.ptr(), 1, 8, BATCH/64);
        vbrcb((__ubuf__ uint32_t*) exp2partb.ptr(), (__ubuf__ uint32_t*) exp2maxbufb.ptr(), 1, 8, BATCH/64);
        pipe_barrier(PIPE_V);
        // combine two parts
        set_vector_mask(0, 0x0f0f0f0f0f0f0f0f);
        pipe_barrier(PIPE_V);
        vector_dup(exp2parta.ptr(), 0.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, 0xf0f0f0f0f0f0f0f0);
        pipe_barrier(PIPE_V);
        vector_dup(exp2partb.ptr(), 0.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);
        vadd(exp2buf.ptr(), exp2parta.ptr(), exp2partb.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        // >=2.0
        vdiv(exp2buf.ptr(), exp2buf.ptr(), exp1buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(exp2buf.ptr(), exp2buf.ptr(), exp0recbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vcmpvs_ge((__ubuf__ uint8_t*)exp2maskbuf.ptr(), exp2buf.ptr(), 2.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(onesbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(exp2buf.ptr(), twosbuf.ptr(), exp2maskbuf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);

        // // make exp2buf equals to shared_exp
        // vmul(exp2buf.ptr(), exp2buf.ptr(), exp1buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        // pipe_barrier(PIPE_V);

        // get mant
        vdiv(mantbuf.ptr(), inp_tsr.ptr(), exp2buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(mantbuf.ptr(), mantbuf.ptr(), exp1buf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(mantbuf.ptr(), mantbuf.ptr(), exp0recbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vmuls(mantbuf.ptr(), mantbuf.ptr(), (float)(1<<(shape.MB-1)), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32a(mantbuf.ptr(), mantbuf.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(mantbuf.ptr(), mantbuf.ptr(), 1.0f/(float)(1<<(shape.MB-1)), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        float maxmant = 2.0f - 1.0f / (float)(1<<(shape.MB-1));
        vmins(mantbuf.ptr(), mantbuf.ptr(), maxmant, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmaxs(mantbuf.ptr(), mantbuf.ptr(), -1.0f * maxmant, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);

        vmul(out_tsr.ptr(), exp2buf.ptr(), mantbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(out_tsr.ptr(), exp1buf.ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(out_tsr.ptr(), exp0buf.ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // copy_ubuf_to_ubuf(out_tsr.vptr(), exp2buf.vptr(), 0, 1, BATCH/8, 0, 0);
        // pipe_barrier(PIPE_V);





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
    Hifxv14ShapeInfo shape;
    Tensor<T, PGM> xmtx, out;
    DBuff<T, PUB> xbuf, outbuf;
    Tensor<float, PUB> absbuf, exp0buf, brcbbuf, sfexpbuf, exp0recbuf;   // for exp0
    Tensor<float, PUB> exp1maxbuf, exp1buf, exp1maskbuf;    // for exp1
    Tensor<float, PUB> exp2parta, exp2partb, exp2maxbufa, exp2maxbufb, exp2buf, exp2maskbuf;  // for exp2
    Tensor<float, PUB> mantbuf;
    Tensor<float, PUB> fp32inbuf, fp32outbuf;
    Tensor<uint32_t, PUB> expmask;
    Tensor<float, PUB> onesbuf, twosbuf, bf16_seven_buf;
    Tensor<bfloat16_t, PUB> bf16_seven_buf_bf16, bf16_general_buf;
    DEvent<PIPE_MTE2, PIPE_V> input_ready{3,4};
    DEvent<PIPE_V, PIPE_MTE2> input_empty{3,4};
    DEvent<PIPE_V, PIPE_MTE3> output_ready{3,4};
    DEvent<PIPE_MTE3, PIPE_V> output_empty{3,4};
    static constexpr int BATCH = 512;
    static constexpr float eps = 5.421011e-20;
};





extern "C" __global__ __aicore__ void hifx_kernel(GM_ADDR xmtx, GM_ADDR out, int M, int N, int mant_bit){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        Hifv14Kernel<float, 2> vec;
        vec.Init(xmtx, out, M, N, mant_bit);
        vec.Process();
#endif
    }
}

extern "C" __global__ __aicore__ void hifx_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int M, int N, int mant_bit){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        Hifv14Kernel<bfloat16_t, 2> vec;
        vec.Init(xmtx, out, M, N, mant_bit);
        vec.Process();
#endif
    }
}


void run_hifx_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit) {
    hifx_kernel<<<40, nullptr, stream>>>(xmtx, out, M, N, mant_bit);
}

void run_hifx_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int M, int N, int mant_bit) {
    hifx_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, M, N, mant_bit);
}
#include "tensorutils.h"




template <typename T>
class Hif8Vec{
public:
    aifunc Hif8Vec(){}
    aifunc void Init(GM_ADDR src, GM_ADDR dst, int len){
        int n_percore = CeilDiv(CeilDiv(len, get_block_num()), BATCH) * BATCH;
        n1 = n_percore*get_block_idx();
        n2 = n1 + n_percore;
        if (n2 > len){ n2 = len; }

        x = Tensor<T, PGM>(src);
        out = Tensor<T, PGM>(dst);

        int offset = 0;
        xbuf = DBuff<T, PUB>(0, BATCH, offset);
        outbuf = DBuff<T, PUB>(0, BATCH, offset);
        expbuf = Tensor<float, PUB>(0, BATCH, offset);
        expabsbuf = Tensor<float, PUB>(0, BATCH, offset);
        expmaxbuf = Tensor<float, PUB>(0, 8, offset);
        expmaxbrcbbuf = Tensor<float, PUB>(0, 64, offset);
        grpexpbuf = Tensor<float, PUB>(0, BATCH, offset);
        scalebuf = Tensor<float, PUB>(0, BATCH, offset);

        le15buf = Tensor<float, PUB>(0, 64, offset);
        le7buf = Tensor<float, PUB>(0, 64, offset);
        le3buf = Tensor<float, PUB>(0, 64, offset);
        eq0buf = Tensor<float, PUB>(0, 64, offset);
        posinfbuf = Tensor<float, PUB>(0, 64, offset);
        neginfbuf = Tensor<float, PUB>(0, 64, offset);
        geminbuf = Tensor<float, PUB>(0, 64, offset);
        // leMaxbuf = Tensor<float, PUB>(0, 64, offset);
        ltzerobuf = Tensor<float, PUB>(0, 64, offset);
        //geMaxbuf = Tensor<float, PUB>(0, 64, offset);
        twosbuf = Tensor<float, PUB>(0, 64, offset);
        onesbuf = Tensor<float, PUB>(0, 64, offset);
        zerosbuf = Tensor<float, PUB>(0, 64, offset);
        epsbuf = Tensor<float, PUB>(0, 64, offset);
        // maxbuf = Tensor<float, PUB>(0, 64, offset);
        //n_maxbuf = Tensor<float, PUB>(0, 64, offset);
        // minbuf = Tensor<float, PUB>(0, 64, offset);
        // n_minbuf = Tensor<float, PUB>(0, 64, offset);
        n_one = Tensor<float, PUB>(0, 64, offset);
        
        expbias1 = Tensor<float, PUB>(0, BATCH, offset);
        expbias2 = Tensor<float, PUB>(0, BATCH, offset);
        expbias3 = Tensor<float, PUB>(0, BATCH, offset);
        sign = Tensor<float, PUB>(0, BATCH, offset);

        xf32buf = Tensor<float, PUB>(0, BATCH, offset);
        inp_abs_tsr = Tensor<float, PUB>(0, BATCH, offset);
        out_abs_tsr = Tensor<float, PUB>(0, BATCH, offset);
        outf32buf = Tensor<float, PUB>(0, BATCH, offset);
        scaled_inp_tsr = Tensor<float, PUB>(0, BATCH, offset);
        // uint32_t max = 0x47000000; //Hif8绝对值的最大值是2^15
        // float *f_max = (float *)&max;
        // *f_max = *f_max * 0.95f;
        // uint32_t exp_min = 0x34800000;//指数最小值-22
        // float *f_exp_min = (float *)&exp_min;
        float eps = 5.421011e-20;

        expmask = Tensor<uint32_t, PUB>(0, 64, offset);
        posinfval = Tensor<uint32_t, PUB>(0, 64, offset);
        // neginfval = Tensor<uint32_t, PUB>(0, 64, offset);
        vector_dup(expmask.ptr(), 0x7F800000, 1, 1, 1, 8, 8);
        vector_dup(posinfval.ptr(), 0x7F800000, 1, 1, 1, 8, 8);//float的正无穷大
        // vector_dup(neginfval.ptr(), 0xFF800000, 1, 1, 1, 8, 8);//float的负无穷大
        vector_dup(twosbuf.ptr(), 0.5f, 1, 1, 1, 8, 8);
        vector_dup(onesbuf.ptr(), 1.0f, 1, 1, 1, 8, 8);
        vector_dup(zerosbuf.ptr(), 0.0f, 1, 1, 1, 8, 8);
        vector_dup(epsbuf.ptr(), eps, 1, 1, 1, 8, 8);
        // vector_dup(maxbuf.ptr(), *f_max, 1, 1, 1, 8, 8);
        //vector_dup(n_maxbuf.ptr(), -*f_max, 1, 1, 1, 8, 8);
        // vector_dup(minbuf.ptr(), *f_exp_min, 1, 1, 1, 8, 8);
        vector_dup(n_one.ptr(), -1.0f, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_ALL);
        // set_cmpmask(onesbuf.vptr());
    }

    aifunc void Compute(){
        in_empty.setall();
        out_empty.setall();

        int cnt = 0;
        for (int n=n1; n<n2; n+=BATCH){
            Compute(n, cnt);
        }

        in_empty.release();
        out_empty.release();
    }

    aifunc void Compute(int &n, int &cnt){
        Tensor<float, PUB> inp_tsr;
        Tensor<float, PUB> out_tsr;
        
        in_empty.wait();
        copy_gm_to_ubuf(xbuf.get(cnt).vptr(), x[n].vptr(), 0, 1, BATCH*sizeof(T)/32, 0, 0);//一次读512个数进来
        in_ready.set();
        
        in_ready.wait();
        out_empty.wait();

        // do type conversion if necessary
        if constexpr(std::is_same<T, float>::value){
            inp_tsr = xbuf.get(cnt);
            out_tsr = outbuf.get(cnt);
        }else{
            inp_tsr = xf32buf;
            out_tsr = outf32buf;
            //out_tsr = outbuf.get(cnt);
            vconv_bf162f32(inp_tsr.ptr(), xbuf.get(cnt).ptr(), BATCH/64, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
        }

       
        uint32_t exp_min = 0x34800000;//指数最小值-22
        float *f_exp_min = (float *)&exp_min;
        uint32_t min = 0x34000000; //绝对值小于2^-23的值归为0
        float *f_min = (float *)&min;

//=================================quant process========================================
        // ---- get exp ---- 
        vand((__ubuf__ uint16_t*)expbuf.ptr(), (__ubuf__ uint16_t*)inp_tsr.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), BATCH/64, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        //64个数一个block，取最大值
        vcmax(expmaxbuf.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 8, ONLY_VALUE);
        pipe_barrier(PIPE_V);
        // 从8 broadcast 到 64
        vbrcb((__ubuf__ uint32_t*)expmaxbrcbbuf.ptr(), (__ubuf__ uint32_t*)expmaxbuf.ptr(), 1, 8, BATCH/64/8);
        pipe_barrier(PIPE_V);
        // 再 broadcast 到 BATCH
        vbrcb((__ubuf__ uint32_t*)grpexpbuf.ptr(), (__ubuf__ uint32_t*)expmaxbrcbbuf.ptr(), 1, 8, BATCH/64);
        pipe_barrier(PIPE_V);
        //除0保护
        vcmpvs_ne((__ubuf__ uint8_t*)eq0buf.ptr(), grpexpbuf.ptr(), 0.0f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_cmpmask(epsbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(grpexpbuf.ptr(), grpexpbuf.ptr(), eq0buf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);
        //量化后的最大值取8, 1/8 = 0.125
        vmuls(scalebuf.ptr(), grpexpbuf.ptr(), 0.125f, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vdiv(scaled_inp_tsr.ptr(), inp_tsr.ptr(), scalebuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        vabs(inp_abs_tsr.ptr(),scaled_inp_tsr.ptr(),BATCH/64, 1, 1, 8, 8);//取绝对值
        pipe_barrier(PIPE_V);

//===================指数处理=====================
        // ---- get exp ---- 
        vand((__ubuf__ uint16_t*)expbuf.ptr(), (__ubuf__ uint16_t*)scaled_inp_tsr.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), BATCH/64, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);

        // ---- get abs exp ----
        // vrec(expabsbuf.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 8, 8); // fuck. this cannot produce correct recip
        vnot((__ubuf__ uint16_t*) expabsbuf.ptr(), (__ubuf__ uint16_t*) expbuf.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vand((__ubuf__ uint16_t*)expabsbuf.ptr(), (__ubuf__ uint16_t*)expabsbuf.ptr(), (__ubuf__ uint16_t*)expmask.ptr(), BATCH/64, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        vadds((__ubuf__ int32_t*)expabsbuf.ptr(), (__ubuf__ int32_t*)expabsbuf.ptr(), -8388608, BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmax(expabsbuf.ptr(), expbuf.ptr(), expabsbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);


        vmaxs(expbuf.ptr(), expbuf.ptr(), *f_exp_min, BATCH/64, 1, 1, 8, 8);//指数的下限夹取,将指数=-23的变为-22
        pipe_barrier(PIPE_V);
        // ---- adjust the exp according to absexp ----
        // compare with different thresholds 
        vcmpvs_le((__ubuf__ uint8_t*)le15buf.ptr(), expabsbuf.ptr(), 32768.0f, BATCH/64, 1, 1, 8, 8);
        vcmpvs_le((__ubuf__ uint8_t*)le7buf.ptr(), expabsbuf.ptr(), 128.0f, BATCH/64, 1, 1, 8, 8);
        vcmpvs_le((__ubuf__ uint8_t*)le3buf.ptr(), expabsbuf.ptr(), 8.0f, BATCH/64, 1, 1, 8, 8);
        vcmpvs_ge((__ubuf__ uint8_t*)eq0buf.ptr(), inp_abs_tsr.ptr(), *f_min, BATCH/64, 1, 1, 8, 8);//对小于2^-23的数归0
        pipe_barrier(PIPE_V);
        
        set_cmpmask(onesbuf.vptr());
        pipe_barrier(PIPE_V);

        vsel(expbias1.ptr(), twosbuf.ptr(), le15buf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        vsel(expbias2.ptr(), twosbuf.ptr(), le7buf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        vsel(expbias3.ptr(), twosbuf.ptr(), le3buf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);

        // manipulate exp 
        vmul(expbias1.ptr(), expbias2.ptr(), expbias1.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        vmul(expbuf.ptr(), expbias3.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(expbuf.ptr(), expbias1.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

        // x div exp and round and mul exp 
        vdiv(out_tsr.ptr(), scaled_inp_tsr.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f32a(out_tsr.ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(out_tsr.ptr(), out_tsr.ptr(), expbuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

         // 0值处理
        set_cmpmask(zerosbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(out_tsr.ptr(), out_tsr.ptr(), eq0buf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);

//==========================dequant process===============================
        vmul(out_tsr.ptr(), out_tsr.ptr(), scalebuf.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);

//===================对结果上下溢出处理================================
        vabs(out_abs_tsr.ptr(),out_tsr.ptr(),BATCH/64, 1, 1, 8, 8);//取绝对值

        vcmpvs_lt((__ubuf__ uint8_t*)ltzerobuf.ptr(), out_tsr.ptr(), 0.0f, BATCH/64, 1, 1, 8, 8);//把负数过滤出来
        vcmpvs_lt((__ubuf__ uint8_t*)posinfbuf.ptr(), out_abs_tsr.ptr(), 40960.0f, BATCH/64, 1, 1, 8, 8);//超过+/-2^15 * 1.25的值就超出Hif8能表示的范围，变为正负无穷的值
        // vcmpvs_ge((__ubuf__ uint8_t*)geminbuf.ptr(), out_abs_tsr.ptr(), *f_exp_min, BATCH/64, 1, 1, 8, 8);//对于小于2^-22的数全归为2^-22
        // vcmpvs_ge((__ubuf__ uint8_t*)eq0buf2.ptr(), out_abs_tsr.ptr(), *f_min, BATCH/64, 1, 1, 8, 8); //对小于2^-23的数归0
        pipe_barrier(PIPE_V);

         //做一个全是-1和1的张量sign，原来是负数的位置为-1
        set_cmpmask(onesbuf.vptr());
        pipe_barrier(PIPE_V);
        vsel(sign.ptr(), n_one.ptr(), ltzerobuf.vptr(), BATCH/64, 1, 1, 1, 8, 0, 1, 1);
        pipe_barrier(PIPE_V);
        
        // fill posinf 处理正无穷的值
        set_cmpmask(posinfval.vptr());
        pipe_barrier(PIPE_V);
        vsel(out_abs_tsr.ptr(), out_abs_tsr.ptr(), posinfbuf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        pipe_barrier(PIPE_V);

        //   //处理小于2^-22的值
        // set_cmpmask(minbuf.vptr());
        // pipe_barrier(PIPE_V);
        // vsel(out_abs_tsr.ptr(), out_abs_tsr.ptr(), geminbuf.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        // pipe_barrier(PIPE_V);

        vmul(out_tsr.ptr(), out_abs_tsr.ptr(), sign.ptr(), BATCH/64, 1, 1, 1, 8, 8, 8);//将符号乘回去
        // pipe_barrier(PIPE_V);

        //  // zero handle
        // set_cmpmask(zerosbuf.vptr());
        // pipe_barrier(PIPE_V);
        // vsel(out_tsr.ptr(), out_tsr.ptr(), eq0buf2.vptr(), BATCH/64, 1, 1, 1, 8, 8, 1, 1);
        // pipe_barrier(PIPE_V);



        // do type conversion if necessary 
        if constexpr(std::is_same<T, float>::value){
        }else{
            vconv_f322bf16a(outbuf.get(cnt).ptr(), out_tsr.ptr(), BATCH/64, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
        }

      

        out_ready.set();
        in_empty.set();

        out_ready.wait();
        copy_ubuf_to_gm(out[n].vptr(), outbuf.get(cnt).vptr(), 0, 1, BATCH*sizeof(T)/32, 0, 0);
        out_empty.set();


        cnt++;
    }

private:
    TPipe pipe;
    int n1, n2;
    //Tensor<float, PGM> x, out;
    Tensor<T, PGM> x, out;
    DBuff<T, PUB> xbuf, outbuf; 
    Tensor<float, PUB> xf32buf, outf32buf;
    Tensor<float, PUB> expbuf, expabsbuf, le15buf, le7buf, le3buf, eq0buf, twosbuf, onesbuf, zerosbuf, expbias1, expbias2, expbias3;
    Tensor<float, PUB> posinfbuf, neginfbuf, out_abs_tsr, n_one, ltzerobuf, geminbuf, sign, inp_abs_tsr;
    Tensor<float, PUB> expmaxbuf, expmaxbrcbbuf, grpexpbuf, epsbuf, scaled_inp_tsr, scalebuf;
    Tensor<uint32_t, PUB> expmask, posinfval;

    DEvent<PIPE_MTE2, PIPE_V> in_ready{3,4};
    DEvent<PIPE_V, PIPE_MTE2> in_empty{3,4};
    DEvent<PIPE_V, PIPE_MTE3> out_ready{3,4};
    DEvent<PIPE_MTE3, PIPE_V> out_empty{3,4};

    static constexpr int BATCH = 512;
};


extern "C" __global__ __aicore__ void hif8_kernel(GM_ADDR xmtx, GM_ADDR out, int len){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        Hif8Vec<float> vec;
        vec.Init(xmtx, out, len);
        vec.Compute();
#endif 
    }
}

extern "C" __global__ __aicore__ void hif8_kernel_bf16(GM_ADDR xmtx, GM_ADDR out, int len){
    if ASCEND_IS_AIV{
#ifdef __DAV_C220_VEC__
        Hif8Vec<bfloat16_t> vec;
        vec.Init(xmtx, out, len);
        vec.Compute();
#endif 
    }
}

void run_hif8_kernel(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len) {
    hif8_kernel<<<40, nullptr, stream>>>(xmtx, out, len);
}

void run_hif8_kernel_bf16(uint32_t blockDim, void* stream, uint8_t* xmtx, uint8_t* out, int len) {
    hif8_kernel_bf16<<<40, nullptr, stream>>>(xmtx, out, len);
}
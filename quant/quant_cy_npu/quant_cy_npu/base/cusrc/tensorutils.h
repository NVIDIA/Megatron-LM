#pragma once
#include "kernel_operator.h"



#ifdef JUSTFORHINTER
// just for type hinting.
#undef __aicore__
#define __aicore__
#include "stub_fun.h"
#endif


using namespace AscendC;

#define gmfloat __gm__ float*
#define gmhalf __gm__ half*
#define gmu8 __gm__ uint8_t*
#define gmbf __gm__ bfloat16_t*
#define aifunc __aicore__ inline


typedef enum{
    PGM=0,
    PL1=1,
    PL0A=2,
    PL0B=3,
    PL0C=4,
    PUB=5,
} pos_t;


__aicore__ constexpr uint32_t GetBufSize(pos_t pos){
    if (pos==PL1){
        return 512 * 1024;
    }
    if (pos==PL0A){
        return 64 * 1024;
    }
    if (pos==PL0B){
        return 64 * 1024;
    }
    if (pos==PL0C){
        return 128 * 1024;
    }
    if (pos==PL1){
        return 192 * 1024;
    }
    return 0;
}

__aicore__ constexpr uint32_t Align32B(uint32_t x){
    return (x + 31) / 32 * 32;
}

__aicore__ constexpr uint32_t Align64B(uint32_t x){
    return (x + 63) / 64 * 64;
}

__aicore__ constexpr uint32_t Align128B(uint32_t x){
    return (x + 127) / 128 * 128;
}

__aicore__ constexpr uint32_t Align256B(uint32_t x){
    return (x + 255) / 256 * 256;
}

__aicore__ constexpr uint32_t Align512B(uint32_t x){
    return (x + 511) / 512 * 512;
}

aifunc int CeilDiv(int a, int b){
    return (a + b - 1) / b;
}

template <typename T, typename T1, typename T2>
aifunc T1 shiftAddr(T1 base, uint64_t size, T2 &offset){
    auto res = base + offset;
    offset += size*sizeof(T);
    return res;
}


/* ------------- Tensor ------------- */

template <typename T, pos_t pos>
class Tensor{};


template <typename T>
class Tensor<T, PGM>{
public:
    aifunc Tensor(){}
    aifunc Tensor(uint64_t offset){
        m_ptr = (__gm__ T*) offset;
    }
    aifunc Tensor(__gm__ uint8_t* ptr){
        m_ptr = (__gm__ T*) ptr;
    }
    aifunc Tensor(__gm__ uint8_t* ptr, int size, int &offset){
        m_ptr = (__gm__ T*) (ptr+offset);
        offset += size * sizeof(T);
    }
    aifunc __gm__ T* ptr(){
        return m_ptr;
    }
    aifunc __gm__ void* vptr(){
        return (__gm__ void*) m_ptr;
    }
    aifunc Tensor<T, PGM> operator[](int off){
        return Tensor<T, PGM>((__gm__ uint8_t*)(m_ptr + off));
    }
    template<typename U>
    aifunc operator Tensor<U, PGM>(){
        return Tensor<U, PGM>((__gm__ uint8_t*) m_ptr);
    }
private:
    __gm__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL1>{
public:
    aifunc Tensor(){}
    aifunc Tensor(uint64_t offset){
        m_ptr = (__cbuf__ T*) offset;
    }
    aifunc Tensor(__cbuf__ uint8_t* ptr){
        m_ptr = (__cbuf__ T*) ptr;
    }
    aifunc Tensor(__cbuf__ uint8_t* ptr, int size, int &offset){
        m_ptr = (__cbuf__ T*) (ptr+offset);
        offset += size * sizeof(T);
    }
    aifunc __cbuf__ T* ptr(){
        return m_ptr;
    }
    aifunc __cbuf__ void* vptr(){
        return (__cbuf__ void*) m_ptr;
    }
    aifunc Tensor<T, PL1> operator[](int off){
        return Tensor<T, PL1>((__cbuf__ uint8_t*)(m_ptr + off));
    }
    aifunc Tensor<T, PL1> rowcol(int r, int c, int C){
        return Tensor<T, PL1>(((__cbuf__ uint8_t*)m_ptr) + (r*C + c) * 16*32);
    }
    template<typename U>
    aifunc operator Tensor<U, PL1>(){
        return Tensor<U, PL1>((__cbuf__ uint8_t*) m_ptr);
    }
private:
    __cbuf__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL0A>{
public:
    aifunc Tensor(){}
    aifunc Tensor(uint64_t offset){
        m_ptr = (__ca__ T*) offset;
    }
    aifunc Tensor(__ca__ uint8_t* ptr){
        m_ptr = (__ca__ T*) ptr;
    }
    aifunc __ca__ T* ptr(){
        return m_ptr;
    }
    aifunc __ca__ void* vptr(){
        return (__ca__ void*) m_ptr;
    }
    aifunc Tensor<T, PL0A> operator[](int off){
        return Tensor<T, PL0A>((__ca__ uint8_t*)(m_ptr + off));
    }
    aifunc Tensor<T, PL0A> rowcol(int r, int c, int C){
        return Tensor<T, PL0A>((__ca__ uint8_t*)m_ptr + (r*C + c) * 16*32);
    }
    template<typename U>
    aifunc operator Tensor<U, PL0A>(){
        return Tensor<U, PL0A>((__ca__ uint8_t*) m_ptr);
    }
private:
    __ca__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL0B>{
public:
    aifunc Tensor(){}
    aifunc Tensor(uint64_t offset){
        m_ptr = (__cb__ T*) offset;
    }
    aifunc Tensor(__cb__ uint8_t* ptr){
        m_ptr = (__cb__ T*) ptr;
    }
    aifunc __cb__ T* ptr(){
        return m_ptr;
    }
    aifunc __cb__ void* vptr(){
        return (__cb__ void*) m_ptr;
    }
    aifunc Tensor<T, PL0B> operator[](int off){
        return Tensor<T, PL0B>((__cb__ uint8_t*)(m_ptr + off));
    }
    aifunc Tensor<T, PL0B> rowcol(int r, int c, int C){
        return Tensor<T, PL0B>((__cb__ uint8_t*)m_ptr + (r*C + c) * 16*32);
    }
    template<typename U>
    aifunc operator Tensor<U, PL0B>(){
        return Tensor<U, PL0B>((__cb__ uint8_t*) m_ptr);
    }
private:
    __cb__ T* m_ptr;
};


template <typename T>
class Tensor<T, PL0C>{
public:
    aifunc Tensor(){}
    aifunc Tensor(uint64_t offset){
        m_ptr = (__cc__ T*) offset;
    }
    aifunc Tensor(__cc__ uint8_t* ptr){
        m_ptr = (__cc__ T*) ptr;
    }
    aifunc __cc__ T* ptr(){
        return m_ptr;
    }
    aifunc __cc__ void* vptr(){
        return (__cc__ void*) m_ptr;
    }
    aifunc Tensor<T, PL0C> operator[](int off){
        return Tensor<T, PL0C>((__cc__ uint8_t*)(m_ptr + off));
    }
    template<typename U>
    aifunc operator Tensor<U, PL0C>(){
        return Tensor<U, PL0C>((__cc__ uint8_t*) m_ptr);
    }
private:
    __cc__ T* m_ptr;
};


template <typename T>
class Tensor<T, PUB>{
public:
    aifunc Tensor(){}
    aifunc Tensor(uint64_t offset){
        m_ptr = (__ubuf__ T*) offset;
    }
    aifunc Tensor(__ubuf__ uint8_t* ptr){
        m_ptr = (__ubuf__ T*) ptr;
    }
    aifunc Tensor(__ubuf__ uint8_t* ptr, int size, int &offset){
        m_ptr = (__ubuf__ T*) (ptr+offset);
        offset += size * sizeof(T);
    }
    aifunc __ubuf__ T* ptr(){
        return m_ptr;
    }
    aifunc __ubuf__ void* vptr(){
        return (__ubuf__ void*) m_ptr;
    }
    aifunc Tensor<T, PUB> operator[](int off){
        return Tensor<T, PUB>((__ubuf__ uint8_t*)(m_ptr + off));
    }
    template<typename U>
    aifunc operator Tensor<U, PUB>(){
        return Tensor<U, PUB>((__ubuf__ uint8_t*) m_ptr);
    }
private:
    __ubuf__ T* m_ptr;
};

/* ------------- Tensor ------------- */


/* ------------- Double Buffer ------------- */

template <typename T, pos_t pos>
class DBuff{
};


template <typename T>
class DBuff<T, PGM>{
public:
    aifunc DBuff(){}
    aifunc DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PGM>(base + offset);
        tsr2 = Tensor<T, PGM>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(__gm__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PGM>(ptr + offset);
        tsr2 = Tensor<T, PGM>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc Tensor<T, PGM> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PGM> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL1>{
public:
    aifunc DBuff(){}
    aifunc DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL1>(base + offset);
        tsr2 = Tensor<T, PL1>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(__cbuf__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL1>(ptr + offset);
        tsr2 = Tensor<T, PL1>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(Tensor<T, PL1> t1, Tensor<T, PL1> t2){
        tsr1 = t1;
        tsr2 = t2;
    }
    aifunc Tensor<T, PL1> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL1> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL0A>{
public:
    aifunc DBuff(){}
    aifunc DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL0A>(base + offset);
        tsr2 = Tensor<T, PL0A>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(__ca__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL0A>(ptr + offset);
        tsr2 = Tensor<T, PL0A>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc Tensor<T, PL0A> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL0A> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL0B>{
public:
    aifunc DBuff(){}
    aifunc DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL0B>(base + offset);
        tsr2 = Tensor<T, PL0B>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(__cb__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL0B>(ptr + offset);
        tsr2 = Tensor<T, PL0B>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc Tensor<T, PL0B> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL0B> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PL0C>{
public:
    aifunc DBuff(){}
    aifunc DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PL0C>(base + offset);
        tsr2 = Tensor<T, PL0C>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(__cc__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PL0C>(ptr + offset);
        tsr2 = Tensor<T, PL0C>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc Tensor<T, PL0C> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PL0C> tsr1, tsr2;
};


template <typename T>
class DBuff<T, PUB>{
public:
    aifunc DBuff(){}
    aifunc DBuff(int base, int size, int &offset){
        tsr1 = Tensor<T, PUB>(base + offset);
        tsr2 = Tensor<T, PUB>(base + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc DBuff(__ubuf__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PUB>(ptr + offset);
        tsr2 = Tensor<T, PUB>(ptr + offset + size*sizeof(T));
        offset += 2 * size * sizeof(T);
    }
    aifunc Tensor<T, PUB> get(int i){
        if (i%2==0){
            return tsr1;
        }else{
            return tsr2;
        }
    }
private:
    Tensor<T, PUB> tsr1, tsr2;
};

/* ------------- Double Buffer ------------- */




/* ------------- Triple Buffer ------------- */

template <typename T, pos_t pos>
class TBuff{
};


template <typename T>
class TBuff<T, PUB>{
public:
    aifunc TBuff(){}
    aifunc TBuff(__ubuf__ uint8_t* ptr, int size, int &offset){
        tsr1 = Tensor<T, PUB>(ptr + offset);
        tsr2 = Tensor<T, PUB>(ptr + offset + size*sizeof(T));
        tsr3 = Tensor<T, PUB>(ptr + offset + 2*size*sizeof(T));
        offset += 3 * size * sizeof(T);
    }
    aifunc Tensor<T, PUB> get(int i){
        if (i%3==0){
            return tsr1;
        }
        else if (i%3==1){
            return tsr2;
        }
        else{
            return tsr3;
        }
    }
private:
    Tensor<T, PUB> tsr1, tsr2, tsr3;
};

/* ------------- Triple Buffer ------------- */


/* ------------- Events ------------- */

template <pipe_t p1, pipe_t p2>
class SEvent{
public:
    aifunc SEvent(){}
    aifunc SEvent(int e_id1, int e_id2){
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
    }
    aifunc SEvent(event_t e_id1, event_t e_id2){
        id1 = e_id1;
        id2 = e_id2;
    }
    aifunc void wait(){
        wait_flag(p1, p2, id1);
    }
    aifunc void set(){
        set_flag(p1, p2, id1);
    }
    aifunc void setall(){
        set();
    }
    aifunc void release(){
        wait();
    }

private:
    event_t id1=(event_t)0, id2=(event_t)1;
};



template <pipe_t p1, pipe_t p2>
class DEvent{
public:
    aifunc DEvent(){}
    aifunc DEvent(int e_id1, int e_id2){
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
    }
    aifunc DEvent(event_t e_id1, event_t e_id2){
        id1 = e_id1;
        id2 = e_id2;
    }
    aifunc void wait(){
        if (wait_cnt%2==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        wait_cnt ++;
    }
    aifunc void set(){
        if (set_cnt%2==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        set_cnt ++;
    }
    aifunc void setall(){
        set();
        set();
    }
    aifunc void release(){
        for (int i=wait_cnt; i<set_cnt; ++i){
            wait();
        }
    }

private:
    event_t id1=(event_t)0, id2=(event_t)1;
    int wait_cnt = 0;
    int set_cnt = 0;
};


template <pipe_t p1, pipe_t p2>
class DEventP{
public:
    aifunc DEventP(){}
    aifunc DEventP(const int e_id1, const int e_id2): id1((event_t)e_id1), id2((event_t)e_id2){
        // id1 = (event_t)e_id1;
        // id2 = (event_t)e_id2;
    }
    aifunc DEventP(const event_t e_id1, const event_t e_id2): id1(e_id1), id2(e_id2){
        // id1 = e_id1;
        // id2 = e_id2;
    }
    aifunc void wait(int &wait_cnt){
        if (wait_cnt==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        // wait_cnt ++;
    }
    aifunc void set(int &set_cnt){
        if (set_cnt==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        // set_cnt ++;
    }
    aifunc void wait(int &&wait_cnt){
        if (wait_cnt==0){
            wait_flag(p1, p2, id1);
        }else{
            wait_flag(p1, p2, id2);
        }
        // wait_cnt ++;
    }
    aifunc void set(int &&set_cnt){
        if (set_cnt==0){
            set_flag(p1, p2, id1);
        }else{
            set_flag(p1, p2, id2);
        }
        // set_cnt ++;
    }
    aifunc void setall(){
        set_flag(p1, p2, id1);
        set_flag(p1, p2, id2);
    }
    aifunc void release(){
        // for (int i=wait_cnt; i<set_cnt; ++i){
            // wait();
        // }
        wait_flag(p1, p2, id1);
        wait_flag(p1, p2, id2);
    }

private:
    const event_t id1, id2;
};


template <pipe_t p1, pipe_t p2>
class TEvent{
public:
    aifunc TEvent(){}
    aifunc TEvent(int e_id1, int e_id2, int e_id3){
        id1 = (event_t)e_id1;
        id2 = (event_t)e_id2;
        id3 = (event_t)e_id3;
    }
    aifunc TEvent(event_t e_id1, event_t e_id2, event_t e_id3){
        id1 = e_id1;
        id2 = e_id2;
        id3 = e_id3;
    }
    aifunc void wait(){
        if (wait_cnt%3==0){
            wait_flag(p1, p2, id1);
        }
        if (wait_cnt%3==1){
            wait_flag(p1, p2, id2);
        }
        if (wait_cnt%3==2){
            wait_flag(p1, p2, id3);
        }
        wait_cnt ++;
    }
    aifunc void set(){
        if (set_cnt%3==0){
            set_flag(p1, p2, id1);
        }
        if (set_cnt%3==1){
            set_flag(p1, p2, id2);
        }
        if (set_cnt%3==2){
            set_flag(p1, p2, id3);
        }
        set_cnt ++;
    }
    aifunc void setall(){
        set();
        set();
        set();
    }
    aifunc void release(){
        for (int i=wait_cnt; i<set_cnt; ++i){
            wait();
        }
    }

private:
    event_t id1=(event_t)0, id2=(event_t)1, id3=(event_t)2;
    int wait_cnt = 0;
    int set_cnt = 0;
};

/* ------------- Events ------------- */
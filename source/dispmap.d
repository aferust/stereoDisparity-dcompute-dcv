@compute(CompileFor.deviceOnly)
module dispmap;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.cuda.sync;

import dcompute.std.memory;
// https://github.com/NVIDIA/cuda-samples/blob/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/5_Domain_Specific/stereoDisparity/stereoDisparity.cu

// to debug or validate ptx
// > ptxas file

uint __usad4(uint A, uint B, uint C = 0) {
  uint result = __irEx!(`
            `, `
        %val = call i32 asm sideeffect "vabsdiff4.u32.u32.u32.add $0, $1, $2, $3;", "=r,r,r,r"(i32 %0, i32 %1, i32 %2), !srcloc !1
        
        ret i32 %val
            `, `!1 = !{i32 1}`, uint, uint, uint, uint)(A, B, C);

  return result;
}

// CUDA tex2D return type
struct int4
{
    uint x, y, z, w;
}

pragma(LDC_intrinsic, "llvm.nvvm.tex.unified.2d.v4u32.s32") //uint
int4 _tex2D(ulong, int /*x*/, int /*y*/) @trusted nothrow @nogc;

uint tex2D(ulong tex, int x, int y){
    return _tex2D(tex, x, y).x;
}

T abs(T)(T val) @trusted nothrow @nogc {
    return (val >= 0) ? val : -val;
}


@kernel void stereoDisparityKernel(ulong img0, ulong img1, GlobalPointer!(int) odata,
                      size_t w, size_t h, int minDisparity, int maxDisparity)
{
    enum RAD = 8;
    enum STEPS = 3;
    enum blockSize_x = 32;
    enum blockSize_y = 8;
    
    const tidx = cast(int)GlobalIndex.x;
    const tidy = cast(int)GlobalIndex.y;
    const sidx = SharedIndex.x + RAD;
    const sidy = SharedIndex.y + RAD;

    uint imLeft;
    uint imRight;
    uint cost;
    uint bestCost = 9999999;
    uint bestDisparity = 0;
    
    enum rows = blockSize_y + 2 * RAD;
    enum cols = blockSize_x + 2 * RAD;
    
    // __shared__ uint diff[rows*cols];
    auto diff = sharedStaticReserve!(uint[rows*cols], "diff0");

    // store needed values for left image into registers (constant indexed local
    // vars)
    uint[3] _imLeftA, _imLeftB;
    uint* imLeftA = _imLeftA.ptr; // skip d's bounds check which requires extra linkage.
    uint* imLeftB = _imLeftB.ptr;

    foreach (i; 0..STEPS) {
        int offset = -RAD + i * RAD;
        imLeftA[i] = tex2D(img0, tidx - RAD, tidy + offset);
        imLeftB[i] = tex2D(img0, tidx - RAD + blockSize_x, tidy + offset);
    }

    // for a fixed camera system this could be hardcoded and loop unrolled
    for (int d = minDisparity; d <= maxDisparity; d++) {
    // LEFT
    //#pragma unroll
    
        foreach (immutable i; 0..STEPS) {
            int offset = -RAD + i * RAD;
            imLeft = imLeftA[i];
            imRight = tex2D(img1, tidx - RAD + d, tidy + offset);
            diff[(sidy + offset)*cols + (sidx - RAD)] = __usad4(imLeft, imRight);
        }
    
    // RIGHT
    //#pragma unroll

        foreach (immutable i; 0..STEPS) {
            int offset = -RAD + i * RAD;

            if (SharedIndex.x < 2 * RAD) {
                // imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
                imLeft = imLeftB[i];
                imRight = tex2D(img1, tidx - RAD + blockSize_x + d,
                                            tidy + offset);

                diff[(sidy + offset )*cols + (sidx - RAD + blockSize_x)] = __usad4(imLeft, imRight);
            }
        }
        
        barrier0();
        
    // sum cost horizontally
    //#pragma unroll

        foreach (immutable j; 0..STEPS) {
            int offset = -RAD + j * RAD;
            cost = 0;
        //#pragma unroll

            for (int i = -RAD; i <= RAD; i++) {
                cost += diff[(sidy + offset )*cols+ (sidx + i)];
            }

            barrier0();
            diff[(sidy + offset)*cols + sidx] = cost;
            barrier0();
        }

        // sum cost vertically
        cost = 0;
    //#pragma unroll

        for (int i = -RAD; i <= RAD; i++) {
            cost += diff[(sidy + i)*cols + sidx];
        }

        // see if it is better or not
        if (cost < bestCost) {
            bestCost = cost;
            bestDisparity = d + RAD;
        }

        barrier0();
    }

    if (tidy < h && tidx < w) {
        odata[tidy * w + tidx] = bestDisparity;
        // odata[tidy * w + tidx] = tex2D(img0, tidx, tidy); // debug tex2D
    }
}


void printInt(uint val){
    __irEx!(`
        @str = private addrspace(4) constant [4 x i8] c"%d\0A\00"
        declare i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)*) nounwind readnone
        declare i32 @vprintf(i8* nocapture, i8*) nounwind
        declare i32 addrspace(5)* @llvm.nvvm.ptr.gen.to.local.p5i32.p0i32(i32*) nounwind readnone    
            `, `
        %tmp = alloca [12 x i32], align 8
        %tmp2 = getelementptr inbounds [12 x i32], [12 x i32]* %tmp, i64 0, i64 0
        %gen2local = call i32 addrspace(5)* @llvm.nvvm.ptr.gen.to.local.p5i32.p0i32(i32* %tmp2)
        
        %getElem12 = getelementptr i32, i32 addrspace(5)* %gen2local, i64 0
        store i32 %0, i32 addrspace(5)* %getElem12, align 8

        %fmt = call i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(4)* @str, i64 0, i64 0))

        %val = bitcast [12 x i32]* %tmp to i8*

        %call = call i32 @vprintf(i8* %fmt, i8* %val)
        ret void
            `, ``, void, uint)(val);
}
void printFloat(float val){
    __irEx!(`
        @str = private addrspace(4) constant [4 x i8] c"%f\0A\00"
        declare i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)*) nounwind readnone
        declare i32 @vprintf(i8* nocapture, i8*) nounwind
        declare float addrspace(5)* @llvm.nvvm.ptr.gen.to.local.p5float.p0float(float*) nounwind readnone    
            `, `
        %tmp = alloca [12 x float], align 8
        %tmp2 = getelementptr inbounds [12 x float], [12 x float]* %tmp, i64 0, i64 0
        %gen2local = call float addrspace(5)* @llvm.nvvm.ptr.gen.to.local.p5float.p0float(float* %tmp2)
        
        %getElem12 = getelementptr float, float addrspace(5)* %gen2local, i64 0
        store float %0, float addrspace(5)* %getElem12, align 8

        %fmt = call i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8(i8 addrspace(4)* getelementptr inbounds ([4 x i8], [4 x i8] addrspace(4)* @str, i64 0, i64 0))

        %val = bitcast [12 x float]* %tmp to i8*

        %call = call i32 @vprintf(i8* %fmt, i8* %val)
        ret void
            `, ``, void, float)(val);
}
/+
extern (C) { // nvptx supports these
    int vprintf(const char*, const char*);
    void* malloc(size_t);
    void free(void*);
}
+/
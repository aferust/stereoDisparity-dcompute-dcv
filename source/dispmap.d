@compute(CompileFor.deviceOnly)
module dispmap;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.cuda.sync;
import dcompute.std.cuda.index;

import dcompute.std.memory;
// https://github.com/NVIDIA/cuda-samples/blob/2e41896e1b2c7e2699b7b7f6689c107900c233bb/Samples/5_Domain_Specific/stereoDisparity/stereoDisparity.cu

// to debug ptx
// validate ptx "c:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\bin\ptxas.exe" file

// CUDA tex2D return type
struct int4
{
    //align (2): // 16 bit alignmemnt
    uint x, y, z, w;
}

// %int4 @llvm.nvvm.tex.unified.2d.v4s32.s32(i64 %tex, i32 %x, i32 %y) // int
pragma(LDC_intrinsic, "llvm.nvvm.tex.unified.2d.v4u32.s32") //uint
int4 tex2D(ulong, int /*x*/, int /*y*/) @trusted nothrow @nogc;

uint tex2DWChannel(ulong adr, int x, int y, size_t c){
    uint val = __irEx!(`
        declare {i32, i32, i32, i32} @llvm.nvvm.tex.unified.2d.v4u32.s32(i64, i32, i32)
            `, `
        %val = tail call { i32, i32, i32, i32 } @llvm.nvvm.tex.unified.2d.v4u32.s32(i64 %0, i32 %1, i32 %2)
        %ret = extractvalue { i32, i32, i32, i32 } %val, %3
        ret i32 %ret
            `, ``, uint, ulong, int, int, size_t)(adr, x, y);
    
    return val;
}

uint toInt(int4 i4val){
    uint ret;
    ubyte[4] ub = [cast(ubyte)i4val.x, cast(ubyte)i4val.y, cast(ubyte)i4val.z, cast(ubyte)i4val.w];

    ret = *cast(uint*)ub.ptr;
    return ret;
}

pragma(LDC_intrinsic, "llvm.nvvm.fabs.d")
double fabs(double) @trusted nothrow @nogc;

T abs(T)(T val) @trusted nothrow @nogc {
    return cast(T)fabs(cast(double)val);
}

pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P) @trusted nothrow @nogc;


// https://forum.dlang.org/post/kqzbrynsaslyleacspqt@forum.dlang.org
pragma(LDC_inline_ir)
    R __irEx(string prefix, string code, string suffix, R, P...)(P) @trusted nothrow @nogc;


@kernel void stereoDisparityKernel(ulong img0, ulong img1, GlobalPointer!(int) odata,
                      size_t w, size_t h, int minDisparity, int maxDisparity)
{
    // membar_cta();
    
    enum RAD = 8;
    enum STEPS = 3;
    enum blockSize_x = 32;
    enum blockSize_y = 8;
    /*
    const tidx = cast(uint)GlobalIndex.x; //ntid_x() * ctaid_x() + tid_x();
    const tidy = cast(uint)GlobalIndex.y; //ntid_y() * ctaid_y() + tid_y();
    const sidx = cast(uint)SharedIndex.x + RAD;
    const sidy = cast(uint)SharedIndex.y + RAD;
    */
    const tidx = cast(int)(SharedDimension.x * GroupIndex.x + SharedIndex.x);
    const tidy = cast(int)(SharedDimension.y * GroupIndex.y + SharedIndex.y);
    const sidx = cast(int)(SharedIndex.x + RAD);
    const sidy = cast(int)(SharedIndex.y + RAD);

    int4 imLeft;
    int4 imRight;
    uint cost;
    uint bestCost = 9999999;
    uint bestDisparity = 0;
    
    enum rows = blockSize_y + 2 * RAD;
    enum cols = blockSize_x + 2 * RAD;
    
    SharedPointer!uint diff = sharedStaticReserve!(uint[rows*cols], "diff0");
    // store needed values for left image into registers (constant indexed local
    // vars)
    // uint[3] imLeftA;
    int4* imLeftA = reserveInt4();
    int4* imLeftB = reserveInt4();
    /*
    uint* imLeftA = inlineIR!(`
        %a = alloca [3 x i32], align 4
        %b = alloca i32*, align 8
        %1 = bitcast [3 x i32]* %a to i32*
        %2 = bitcast i32* %1 to i8*
        %3 = bitcast i32* %1 to i8*
        %4 = bitcast [3 x i32]* %a to i32*
        %5 = insertvalue { i64, i32* } { i64 3, i32* undef }, i32* %4, 1 
        %6 = bitcast [3 x i32]* %a to i32*
        store i32* %6, i32** %b, align 8
        %7 = load i32*, i32** %b, align 8
        %8 = load i32*, i32** %b, align 8
        ret i32* %8`, uint*)();
    
    
    uint* imLeftB = inlineIR!(`
        %a = alloca [3 x i32], align 4
        %b = alloca i32*, align 8
        %1 = bitcast [3 x i32]* %a to i32*
        %2 = bitcast i32* %1 to i8*
        %3 = bitcast i32* %1 to i8*
        %4 = bitcast [3 x i32]* %a to i32*
        %5 = insertvalue { i64, i32* } { i64 3, i32* undef }, i32* %4, 1 
        %6 = bitcast [3 x i32]* %a to i32*
        store i32* %6, i32** %b, align 8
        %7 = load i32*, i32** %b, align 8
        %8 = load i32*, i32** %b, align 8
        ret i32* %8`, uint*)();
    */
    foreach (i; 0..STEPS) {
        int offset = -RAD + i * RAD;
        imLeftA[i] = tex2D(img0, tidx - RAD, tidy + offset);
        imLeftB[i] = tex2D(img0, tidx - RAD + blockSize_x, tidy + offset);
    }

    // for a fixed camera system this could be hardcoded and loop unrolled
    for (int d = minDisparity; d <= maxDisparity; d++) {
    // LEFT
    //#pragma unroll
        foreach (i; 0..STEPS) {
            int offset = -RAD + i * RAD;
            imLeft = imLeftA[i];
            imRight = tex2D(img1, tidx - RAD + d, tidy + offset);

            
            uint absdiff;
            {
                absdiff += cast(uint)abs(imLeft.x - imRight.x);
                absdiff += cast(uint)abs(imLeft.y - imRight.y);
                absdiff += cast(uint)abs(imLeft.z - imRight.z);
                //absdiff += cast(uint)abs(imLeft.w - imRight.w);
            }
            
            cost = absdiff;
            
            //cost = __usad4(imLeft, imRight, dummy);
            diff[(sidy + offset)*cols + (sidx - RAD)] = cost;

        }

    // RIGHT
    //#pragma unroll

        foreach (i; 0..STEPS) {
            int offset = -RAD + i * RAD;

            if (SharedIndex.x < 2 * RAD) {
                // imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
                imLeft = imLeftB[i];
                imRight = tex2D(img1, tidx - RAD + blockSize_x + d,
                                            tidy + offset);
                uint absdiff;
                {
                    absdiff += cast(uint)abs(imLeft.x - imRight.x);
                    absdiff += cast(uint)abs(imLeft.y - imRight.y);
                    absdiff += cast(uint)abs(imLeft.z - imRight.z);
                    //absdiff += cast(uint)abs(imLeft.w - imRight.w);
                }
                cost = absdiff;
                
                //cost = __usad4(imLeft, imRight, dummy);
                diff[(sidy + offset )*cols+ (sidx - RAD + blockSize_x)] = cost;
                
            }
        }

        
        barrier0();
        

    // sum cost horizontally
    //#pragma unroll

        foreach (j; 0..STEPS) {
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
            bestDisparity = d + 8;
        }

        barrier0();
    }

    if (tidy < h && tidx < w) {
        odata[tidy * w + tidx] = bestDisparity;
    }

}

int4* reserveInt4(){
    void* address = __irEx!(`
        %Dummy = type { i32, i32, i32, i32 }    
            `, `
        %a = alloca [3 x %Dummy], align 4
        %b = alloca %Dummy*, align 8
        %1 = bitcast [3 x %Dummy]* %a to %Dummy*
        %2 = bitcast %Dummy* %1 to i8*
        %3 = bitcast %Dummy* %1 to i8*
        %4 = bitcast [3 x %Dummy]* %a to %Dummy*
        %5 = insertvalue { i64, %Dummy* } { i64 3, %Dummy* undef }, %Dummy* %4, 1 
        %6 = bitcast [3 x %Dummy]* %a to %Dummy*
        store %Dummy* %6, %Dummy** %b, align 8
        %7 = load %Dummy*, %Dummy** %b, align 8
        %8 = load %Dummy*, %Dummy** %b, align 8
        %vptr = bitcast %Dummy* %8 to i8*
        ret i8* %vptr
            `, ``, void*)();
    return cast(int4*)address;
}
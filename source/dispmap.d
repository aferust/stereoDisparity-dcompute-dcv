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
    align (2): // 16 bit alignmemnt
    uint x, y, z, w;
}

// %int4 @llvm.nvvm.tex.unified.2d.v4s32.s32(i64 %tex, i32 %x, i32 %y) // int
pragma(LDC_intrinsic, "llvm.nvvm.tex.unified.2d.v4u32.s32") //uint
int4 tex2D(ulong, int /*x*/, int /*y*/);

uint tex2DD(ulong adr, int x, int y){
    uint val = __irEx!(`
        declare {i32, i32, i32, i32} @llvm.nvvm.tex.unified.2d.v4u32.s32(i64, i32, i32)
            `, `
        %val = tail call { i32, i32, i32, i32 } @llvm.nvvm.tex.unified.2d.v4u32.s32(i64 %0, i32 %1, i32 %2)
        %ret = extractvalue { i32, i32, i32, i32 } %val, 0
        ret i32 %ret
            `, ``, uint, ulong, int, int)(adr, x, y);
    
    return val;
}

uint toInt(int4 i4val){
    return i4val.x;
}

pragma(LDC_intrinsic, "llvm.nvvm.fabs.d")
double fabs(double);

T abs(T)(T val){
    return cast(T)fabs(cast(double)val);
}

pragma(LDC_inline_ir)
    R inlineIR(string s, R, P...)(P);


// https://forum.dlang.org/post/kqzbrynsaslyleacspqt@forum.dlang.org
pragma(LDC_inline_ir)
    R __irEx(string prefix, string code, string suffix, R, P...)(P) @trusted nothrow @nogc;


@kernel void stereoDisparityKernel(ulong img0, ulong img1, GlobalPointer!(int) odata,
                      size_t w, size_t h, size_t pitch, int minDisparity, int maxDisparity)
{
    // membar_cta();
    
    enum RAD = 8;
    enum STEPS = 3;
    enum blockSize_x = 32;
    enum blockSize_y = 8;

    const tidx = cast(uint)GlobalIndex.x; //ntid_x() * ctaid_x() + tid_x();
    const tidy = cast(uint)GlobalIndex.y; //ntid_y() * ctaid_y() + tid_y();
    const sidx = cast(uint)SharedIndex.x + RAD;
    const sidy = cast(uint)SharedIndex.y + RAD;

    uint imLeft;
    uint imRight;
    uint cost;
    uint bestCost = 9999999;
    uint bestDisparity = 0;
    
    enum rows = blockSize_y + 2 * RAD;
    enum cols = blockSize_x + 2 * RAD;
    
    SharedArr diff = sharedStaticReserve!(uint[rows*cols], "diff0");
    // store needed values for left image into registers (constant indexed local
    // vars)
    // uint[3] imLeftA;
    
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

    for (int i = 0; i < STEPS; i++) {
        int offset = -RAD + i * RAD;
        imLeftA[i] = tex2DD(img0, tidx - RAD, tidy + offset);
        imLeftB[i] = tex2DD(img0, tidx - RAD + blockSize_x, tidy + offset);
    }

    int dummy = 0;
    // for a fixed camera system this could be hardcoded and loop unrolled
    for (int d = minDisparity; d <= maxDisparity; d++) {
    // LEFT
    //#pragma unroll
        for (int i = 0; i < STEPS; i++) {
            int offset = -RAD + i * RAD;
            imLeft = imLeftA[i];
            imRight = tex2DD(img1, tidx - RAD + d, tidy + offset);

            
            uint absdiff;

            ubyte* A = cast(ubyte*)(&imLeft);

            ubyte* B = cast(ubyte*)(&imRight);

            for (int k=0; k<4; k++)
            {
                //auto val = cast(int)(A[k] - B[k]);
                //absdiff +=  (val > 0)? val : - val;
                absdiff += cast(int)abs(A[k] - B[k]);
            }
            
            cost = absdiff;
            
            //cost = __usad4(imLeft, imRight, dummy);
            //diff[(sidy + offset)*cols + (sidx - RAD)] = cost;
            setSharedVal!uint(diff, (sidy + offset)*cols + (sidx - RAD), cost);

        }

    // RIGHT
    //#pragma unroll

        for (int i = 0; i < STEPS; i++) {
            int offset = -RAD + i * RAD;

            if (SharedIndex.x < 2 * RAD) {
                // imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
                imLeft = imLeftB[i];
                imRight = tex2DD(img1, tidx - RAD + blockSize_x + d,
                                            tidy + offset/+, w+/);
                uint absdiff;
                ubyte* A = cast(ubyte*)(&imLeft);

                ubyte* B = cast(ubyte*)(&imRight);

                for (int k=0; k<4; k++)
                {
                    //auto val = cast(int)(A[k] - B[k]);
                    //absdiff +=  (val > 0)? val : - val;
                    absdiff += cast(int)abs(A[k] - B[k]);
                }
                cost = absdiff;
                
                //cost = __usad4(imLeft, imRight, dummy);
                //diff[(sidy + offset )*m+ (sidx - RAD + blockSize_x)] = cost;
                setSharedVal!uint(diff, (sidy + offset )*cols + (sidx - RAD + blockSize_x), cost);
            }
        }

        
        barrier0();
        

    // sum cost horizontally
    //#pragma unroll

        for (int j = 0; j < STEPS; j++) {
            int offset = -RAD + j * RAD;
            cost = 0;
        //#pragma unroll

            for (int i = -RAD; i <= RAD; i++) {
                //cost += diff[(sidy + offset )*m+ (sidx + i)];
                cost += getSharedVal!uint(diff, (sidy + offset )*cols + (sidx + i));
            }

            barrier0();
            //syncAndSetDiff(diff, (sidy + offset)*m + sidx, cost);
            //diff[(sidy + offset)*m + sidx] = cost;
            setSharedVal!uint(diff, (sidy + offset)*cols + sidx, cost);
            barrier0();
        }

        // sum cost vertically
        cost = 0;
    //#pragma unroll

        for (int i = -RAD; i <= RAD; i++) {
            //cost += diff[(sidy + i)*m + sidx];
            cost += getSharedVal!uint(diff, (sidy + i)*cols + sidx);
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

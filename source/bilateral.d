@compute(CompileFor.deviceOnly)
module bilateral;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.cuda.sync;

import dcompute.std.memory;

// CUDA tex2D return type
struct float4
{
    float x, y, z, w;

    float4 opBinary(string op)(float s) if (op == "+"){
        return float4(x+s, y+s, z+s, w+s);
    }
    float4 opBinary(string op)(float s) if (op == "*") {
        return float4(x*s, y*s, z*s, w*s);
    }
    float4 opBinary(string op)(float s) if (op == "/") {
        return float4(x/s, y/s, z/s, w/s);
    }

    float4 opBinary(string op)(float4 other){
        static if (op == "+"){
            return float4(x+other.x, y+other.y, z+other.z, w+other.w);
        } else
            static assert(0, "op is not implemented");
    }
}

pragma(LDC_intrinsic, "llvm.nvvm.tex.unified.2d.v4f32.s32") //float4
float4 tex2D(ulong, int, int) @trusted nothrow @nogc;


T abs(T)(T val) @trusted nothrow @nogc {
    return (val >= 0) ? val : -val;
}

float __expf(float x){ // intrinsic?
    float x1;
    enum precision = 0.01f;
    float sum = 0.0f;
    int n = 0;
    x1 = 1;
    do {
        sum += x1;
        x1  *= (x / ++n);
    } while (x1 > precision);

    return sum;
}

float saturatef(float val){ // intrinsic?
    if(val <= 0) return 0;
    if(val >= 1) return 1;
    return val;
}

float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.0f * d * d));
}

uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = saturatef(abs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = saturatef(abs(rgba.y));
    rgba.z = saturatef(abs(rgba.z));
    rgba.w = saturatef(abs(rgba.w));
    return (cast(uint)(rgba.w * 255.0f) << 24) | (cast(uint)(rgba.z * 255.0f) << 16) | (cast(uint)(rgba.y * 255.0f) << 8) | cast(uint)(rgba.x * 255.0f);
}

float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

/+
enum radius = 3;
enum delta = 4.0f;
...
TexHandle dismap_tex = cudaAllocAndGetTextureObjectFloat4(cast(void*)imres.ptr, width, height); 
ulong rgbaTex = dismap_tex.texid;
scope(exit) cudaFree(dismap_tex.devmemptr);
...
uint[3] gridSize = [cast(uint)((width + 16 - 1) / 16), cast(uint)((height + 16 - 1) / 16), 1];
uint[3] blockSize = [16, 16, 1];
+/

@kernel void bilateral_filter(ulong rgbaTex, GlobalPointer!(int) output, size_t w, size_t h,
                   float delta, int radius)
{
    int x = cast(int)GlobalIndex.x;
    int y = cast(int)GlobalIndex.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float[64] __cGaussian; // TODO: how to allocate ConstantPointer!float
    float* cGaussian = __cGaussian.ptr;
    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float _x = i-radius;
        cGaussian[i] = __expf(-(_x*_x) / (2*delta*delta));
    }

    float sum = 0.0f;
    float factor;
    float4 t = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 center = tex2D(rgbaTex, x, y);

    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            float4 curPix = tex2D(rgbaTex, x + j, y + i);
            
            factor = cGaussian[i + radius] * cGaussian[j + radius] *     //domain factor
                     euclideanLen(curPix, center, delta);             //range factor

            t =  t + curPix * factor;
            sum += factor;
        }
    }
    
    output[y * w + x] = rgbaFloatToInt(t/sum); /*rgbaFloatToInt(tex2D(rgbaTex, x, y));*/
}

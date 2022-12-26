@compute(CompileFor.deviceOnly)
module bilateral;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
import dcompute.std.cuda.sync;

import dcompute.std.memory;
import dcompute.std.cuda.math : ex2_approx_f, saturate_f;
import dcompute.std.cuda.texture : float4, tex2D = tex_unified_2d_v4f32_s32;

// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/bilateralFilter/bilateral_kernel.cu

T abs(T)(T val) @trusted nothrow @nogc {
    return (val >= 0) ? val : -val;
}

float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return ex2_approx_f(-mod / (2.0f * d * d));
}

uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = saturate_f(abs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = saturate_f(abs(rgba.y));
    rgba.z = saturate_f(abs(rgba.z));
    rgba.w = saturate_f(abs(rgba.w));
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

    // host code initialize this __constant__ array
    immutable(float)* cGaussian = constStaticReserve!(float[64], "gauss0");

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

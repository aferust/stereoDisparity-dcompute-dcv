@compute(CompileFor.deviceOnly)
module sobel;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;
/*
pragma(LDC_intrinsic, "llvm.nvvm.sqrt.rn.f")
float sqrt(float);

@kernel void sobel_gpu(GlobalPointer!(ubyte) output, GlobalPointer!(ubyte) input, size_t width, size_t height) {
    auto x = GlobalIndex.x;
    auto y = GlobalIndex.y;
    float dx, dy;
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* input[(y-1)*width + (x-1)]) + (-2*input[y*width+(x-1)]) + (-1*input[(y+1)*width+(x-1)]) +
             (    input[(y-1)*width + (x+1)]) + ( 2*input[y*width+(x+1)]) + (   input[(y+1)*width+(x+1)]);
        dy = (    input[(y-1)*width + (x-1)]) + ( 2*input[(y-1)*width+x]) + (   input[(y-1)*width+(x+1)]) +
             (-1* input[(y+1)*width + (x-1)]) + (-2*input[(y+1)*width+x]) + (-1*input[(y+1)*width+(x+1)]);
        output[y*width + x] = cast(ubyte)sqrt( (dx*dx) + (dy*dy) );
    }
}
*/
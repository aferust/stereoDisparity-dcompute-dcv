module cudahelper;

import derelict.cuda;
import std.typecons;

alias TexHandle = Tuple!(ulong, "texid", void*, "devmemptr", size_t, "pitch");

TexHandle cudaAllocAndGetTextureObject(void* imdata, size_t width, size_t height){
    cudaResourceDesc resDesc;
    
    size_t pitch;
    void* devmemptr;
    cudaMallocPitch(&devmemptr, &pitch, uint.sizeof*width, height);

    cudaMemcpy2D(devmemptr, pitch, imdata, uint.sizeof*width, uint.sizeof*width, height, cudaMemcpyHostToDevice);

    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devmemptr;
    resDesc.res.pitch2D.pitchInBytes =  pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.pitch2D.desc.x = 32; // bits per channel 
    resDesc.res.pitch2D.desc.y = 0;

    cudaTextureDesc texDesc;
    texDesc.readMode = cudaReadModeElementType;
    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, null);

    return TexHandle(tex, devmemptr, pitch);
}
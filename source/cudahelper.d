module cudahelper;

import derelict.cuda;
import std.typecons;

// https://stackoverflow.com/questions/54098747/cuda-how-to-create-2d-texture-object

alias TexHandle = Tuple!(ulong, "texid", void*, "devmemptr", size_t, "pitch");

TexHandle cudaAllocAndGetTextureObject(void* imdata, size_t width, size_t height, int devtexpitchalignment){
    //height = 4*devtexpitchalignment*height;

    cudaResourceDesc resDesc;
    
    size_t pitch;
    void* devmemptr;
    cudaMallocPitch(&devmemptr, &pitch, uint.sizeof*width, height);

    cudaMemcpy2D(devmemptr, pitch, imdata, uint.sizeof*width, uint.sizeof*width, height, cudaMemcpyHostToDevice);
    
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devmemptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc(cast(int)(uint.sizeof * 8), 0, 0, 0, cudaChannelFormatKindUnsigned);

    cudaTextureDesc texDesc;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, null);
    return TexHandle(tex, devmemptr, pitch);
}
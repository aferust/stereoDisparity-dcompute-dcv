module cudatexture;

import core.stdc.stdio : printf;
import core.stdc.stdlib : exit;

import derelict.cuda;
import std.typecons;

// https://stackoverflow.com/questions/54098747/cuda-how-to-create-2d-texture-object

alias TexHandle = Tuple!(ulong, "texid", void*, "devmemptr");

TexHandle cudaAllocAndGetTextureObject(void* imdata, size_t width, size_t height){
    //const numData = width * height;
    //const memSize = 4 * ubyte.sizeof * numData;

    cudaResourceDesc resDesc;
    
    size_t pitch;
    void* devmemptr;
    gpuAssert(cudaMallocPitch(&devmemptr, &pitch, 4 * ubyte.sizeof*width, height));
    //cudaMalloc(cast(void **)&devmemptr, memSize);
    gpuAssert(cudaMemcpy2D(devmemptr, pitch, imdata, 4 * ubyte.sizeof*width, 4 * ubyte.sizeof*width, height, cudaMemcpyHostToDevice));
    //cudaMemcpy(devmemptr, imdata, memSize, cudaMemcpyHostToDevice);
    //pitch.writeln;
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devmemptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc(cast(int)uint.sizeof * 8, 0, 0, 0, cudaChannelFormatKindUnsigned);
    //resDesc.res.pitch2D.pitchInBytes = width*4;

    cudaTextureDesc texDesc;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;

    cudaTextureObject_t tex;
    gpuAssert(cudaCreateTextureObject(&tex, &resDesc, &texDesc, null));
    return TexHandle(tex, devmemptr);
}

TexHandle cudaAllocAndGetTextureObjectFloat4(void* imdata, size_t width, size_t height){
    cudaResourceDesc resDesc;
    
    size_t pitch;
    void* devmemptr;
    gpuAssert(cudaMallocPitch(&devmemptr, &pitch, 4 * ubyte.sizeof*width, height));
    //cudaMalloc(cast(void **)&devmemptr, memSize);
    gpuAssert(cudaMemcpy2D(devmemptr, pitch, imdata, 4 * ubyte.sizeof*width, 4 * ubyte.sizeof*width, height, cudaMemcpyHostToDevice));
    //cudaMemcpy(devmemptr, imdata, memSize, cudaMemcpyHostToDevice);
    //pitch.writeln;
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = devmemptr;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    //resDesc.res.pitch2D.pitchInBytes = width*4;

    cudaTextureDesc texDesc;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t tex;
    gpuAssert(cudaCreateTextureObject(&tex, &resDesc, &texDesc, null));
    return TexHandle(tex, devmemptr);
}

void gpuAssert(cudaError_t code, bool abort=true, string file = __FILE__, int line = __LINE__)
{
   if (code != cudaSuccess) 
   {
      printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file.ptr, line);
      if (abort) exit(code);
   }
}
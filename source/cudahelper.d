module cudahelper;
import std.stdio;
import derelict.cuda;
import std.typecons;

// https://stackoverflow.com/questions/54098747/cuda-how-to-create-2d-texture-object

alias TexHandle = Tuple!(ulong, "texid", void*, "devmemptr", size_t, "pitch");

TexHandle cudaAllocAndGetTextureObject(void* imdata, size_t width, size_t height){
    
    const numData = width * height;
    const memSize = 4 * ubyte.sizeof * numData;

    cudaResourceDesc resDesc;
    
    size_t pitch;
    void* devmemptr;
    cudaMallocPitch(&devmemptr, &pitch, 4 * ubyte.sizeof*width, height);
    //cudaMalloc(cast(void **)&devmemptr, memSize);
    cudaMemcpy2D(devmemptr, pitch, imdata, 4 * ubyte.sizeof*width, 4 * ubyte.sizeof*width, height, cudaMemcpyHostToDevice);
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
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;

    cudaTextureObject_t tex;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, null);
    return TexHandle(tex, devmemptr, pitch);
}
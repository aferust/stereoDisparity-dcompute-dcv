
import std.stdio, std.math : ceil;
import std.experimental.allocator : theAllocator;

import dcv.plot, dcv.imageio, dcv.imgproc, dcv.core;
import mir.ndslice;

import dcompute.driver.cuda : Platform, Context, Pointer, Program, Queue, Buffer, Copy;

import derelict.cuda;

import dispmap;
import cudatexture : cudaAllocAndGetTextureObject, cudaAllocAndGetTextureObjectFloat4, TexHandle;

T iDivUp(T)(T a, T b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void main()
{
    Platform.initialise();
    DerelictCUDARuntime.load();

    auto devs = Platform.getDevices(theAllocator);
    auto dev  = devs[0];

    auto ctx   = Context(dev); scope(exit) ctx.detach();
    Program.globalProgram = Program.fromFile("kernels_cuda300_64.ptx");
    auto q = Queue(false);
    
    auto imrgbleft = imread("im0.png").sliced.make4Channel;
    auto imrgbright = imread("im1.png").sliced.make4Channel;
    
    size_t height = imrgbleft.shape[0];
    size_t width = imrgbleft.shape[1];
    
    auto imres = slice!int(height, width);
    imres[] = 0;
    
    Buffer!(int) b_res;

    b_res =  Buffer!(int)(imres.ptr[0..height*width]); scope(exit) b_res.release();
    
    TexHandle l_handle = cudaAllocAndGetTextureObject(cast(void*)imrgbleft.ptr, width, height); 
    ulong l_img = l_handle.texid;
    scope(exit) cudaFree(l_handle.devmemptr);

    TexHandle r_handle = cudaAllocAndGetTextureObject(cast(void*)imrgbright.ptr, width, height); 
    ulong r_img = r_handle.texid;
    scope(exit) cudaFree(r_handle.devmemptr);

    enum blockSize_x = 32;
    enum blockSize_y = 8;
    
    uint[3] numThreads = [blockSize_x, blockSize_y, 1];
    uint[3] numBlocks       = [
                                cast(uint)iDivUp(width, numThreads[0]), cast(uint)iDivUp(height, numThreads[1]), 1
                            ];

    q.enqueue!(stereoDisparityKernel)
                (numBlocks, numThreads)
                (l_img, r_img, b_res, width, height, -16, 0);
    
    ctx.sync();
    
    b_res.copy!(Copy.deviceToHost);
    imres[] *= 20;
    imshow(imres, "imres");

    import bilateral;
    enum radius = 3;
    enum delta = 4.0f;
    
    TexHandle dismap_tex = cudaAllocAndGetTextureObjectFloat4(cast(void*)imres.ptr, width, height); 
    ulong rgbaTex = dismap_tex.texid;
    scope(exit) cudaFree(dismap_tex.devmemptr);
    
    uint[3] gridSize = [cast(uint)((width + 16 - 1) / 16), cast(uint)((height + 16 - 1) / 16), 1];
    uint[3] blockSize = [16, 16, 1];

    auto filtered = slice!int([height, width], 0);
    auto b_fil =  Buffer!(int)(filtered.ptr[0..height*width]); scope(exit) b_fil.release();

    q.enqueue!(bilateral_filter)
                (gridSize, blockSize)
                (rgbaTex, b_fil, width, height, delta, radius);

    b_fil.copy!(Copy.deviceToHost);

    imshow(filtered, "filtered");

    imwrite(imres.as!ubyte.slice.asImage(ImageFormat.IF_MONO), "dismap.png");
    waitKey();
}



Slice!(ubyte*, 3) make4Channel(Slice!(ubyte*, 3) imrgb) {

    auto size = imrgb.shape[0] * imrgb.shape[1];

    auto outim = slice!ubyte([imrgb.shape[0], imrgb.shape[1], 4], 0);

    outim[0..$, 0..$, 0] = imrgb[0..$, 0..$, 0];
    outim[0..$, 0..$, 1] = imrgb[0..$, 0..$, 1];
    outim[0..$, 0..$, 2] = imrgb[0..$, 0..$, 2];
    
    return outim;
}
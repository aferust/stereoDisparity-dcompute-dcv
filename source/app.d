
import std.stdio, std.math : ceil;
import std.experimental.allocator : theAllocator;

import dcv.plot, dcv.imageio, dcv.imgproc, dcv.core;
import mir.ndslice;

import dcompute.driver.cuda : Platform, Context, Pointer, Program, Queue, Buffer, Copy;

import derelict.cuda;

import sobel;
import dispmap;
import cudahelper : cudaAllocAndGetTextureObject, TexHandle;

T iDivUp(T)(T a, T b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void main()
{
    Platform.initialise();
    DerelictCUDARuntime.load();

    auto devs = Platform.getDevices(theAllocator);
    auto dev   = devs[0];

    int devtexpitchalignment = dev.texturePitchAlignment;

    auto ctx   = Context(dev); scope(exit) ctx.detach();
    Program.globalProgram = Program.fromFile("kernels_cuda300_64.ptx");
    auto q = Queue(false);
    
    auto imrgbleft = imread("left.png").sliced.make4Channel;
    auto imrgbright = imread("right.png").sliced.make4Channel;
    
    size_t height = imrgbleft.shape[0];
    size_t width = imrgbleft.shape[1];
    
    auto imres = slice!int(height, width);
    imres[] = 0;
    
    Buffer!(int) b_res;

    //uint[] leftint = cast(uint[])imrgbleft.ptr[0..height*width*4];
    //uint[] rightint = cast(uint[])imrgbright.ptr[0..height*width*4];

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
    /*
    q.enqueue!(stereoDisparityKernel)
                (numBlocks, numThreads)
                (l_img, r_img, b_res, width, height, -16, 0);
    */
    b_res.copy!(Copy.deviceToHost);

    imres[] *= 20;
    imshow(imres/*.as!ubyte.slice*/, "imres");
    //imwrite(imres.as!ubyte.slice.asImage(ImageFormat.IF_MONO), "out1.png");
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
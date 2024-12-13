# 卷积的优化

卷积的实现一般有三种方式：
1. 直接按照卷积的定义，通过滑动窗口去对图像进行卷积。
2. FFT 卷积计算优化
    根据卷积的性质，利用傅立叶变换可以将卷积转换为频域上的乘法，在频域上的计算对应乘法，再使用傅立叶变换逆变换，就可以得到卷积对应的计算结果。该方法使用高性能的傅立叶变换算法，如 FFT，可以实现卷积计算的优化，算法性能完全取决于傅立叶变换的性能以及相应卷积参数。
3. Im2col+matmul 卷积计算优化
    由于卷积计算中有大量的乘加运算，和矩阵乘具有很多相似的特点，因此该方法使用 Im2col 的操作将卷积运算转化为矩阵运算，最后调用高性能的 Matmul 进行计算。该方法适应性强，支持各种卷积参数的优化，在通道数稍大的卷积中性能基本与 Matmul 持平，并且可以与其他优化方法形成互补。
4. Winograd 卷积计算优化
    Winograd 方法是按照 Winograd 算法的原理将卷积运行进行转变，达到减少卷积运算中乘法的计算总量。其主要是通过将卷积中的乘法使用加法来替换，并把一部分替换出来的加法放到 weight 的提前处理中，从而达到加速卷积计算的目的。Winograd 算法的优化局限为在一些特定的常用卷积参数才支持。

下面对Im2col+matmul的方法进行记录：
输入：                                          <br />                        
input [N, in_channels, in_h, in_w]              <br />
weights [out_channels, kernel_h, kernel_w]      <br />
bias(optional) [out_channels]                   <br />

计算卷积分为三个过程：                            <br />

1. im2col：将input进行重排 [out_w * out_h, in_channels * kernel_w * kernel_h], weight进行重排 [out_channels, in_channels * kernel_w * kernel_h]
2. gemm: 对上面两个矩阵进行matmul，得到结果为 [out_channels, out_w * out_h]的矩阵
3. col2im: 将结果矩阵重新排列回去，得到结果 [out_channels, out_h, out_w]

参考链接：
[MegEngine Inference 卷积优化之 Im2col 和 winograd 优化](https://www.cnblogs.com/megengine/p/16405753.html)
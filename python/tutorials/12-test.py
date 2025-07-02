import torch
import triton
import triton.language as tl
from triton.runtime import driver


def is_cuda():
    target = driver.active.get_current_target()
    return target is not None and target.backend == "cuda"


def supports_pdl():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9



@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector. 指向第一个输入向量的指针。
               y_ptr,  # *Pointer* to second input vector. 指向第二个输入向量的指针。
               output_ptr,  # *Pointer* to output vector. 指向输出向量的指针。
               n_elements,  # Size of the vector. 向量的大小。
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process. 每个程序应处理的元素数量。
               # NOTE: `constexpr` so it can be used as a shape value. 注意：`constexpr` 因此它可以用作形状值。
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0. 使用 1D 启动网格，因此轴为 0。

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

@triton.jit
def matmul_kernel(

        a_ptr, b_ptr, c_ptr,

        M, N, K,
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,

        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m


    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
       
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    tl.extra.cuda.gdc_launch_dependents()
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)



    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b, activation=""):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=2,
        ACTIVATION="none"  #
    )

    return c




if __name__ == "__main__":
    a = torch.randn(1024, 7168).cuda().to(torch.float16)
    b = torch.randn(7168, 4096).cuda().to(torch.float16)
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    d = torch.randn((M, N), device=a.device, dtype=torch.float16)
    o = torch.randn((M, N), device=a.device, dtype=torch.float16)
    def compute():
        grid1 = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
        n_elements = o.numel()
        matmul_kernel[grid1](
            a, b, c,  #
            M, N, K,  #
            a.stride(0), a.stride(1),  #
            b.stride(0), b.stride(1),  #
            c.stride(0), c.stride(1),  #
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            
            BLOCK_SIZE_K=64,
            GROUP_SIZE_M=2,
            ACTIVATION="none",
            launch_pdl=True
        )
        add_kernel[grid2](d, c, o, n_elements, BLOCK_SIZE=1024,launch_pdl=True)
        return o

    triton.testing.do_bench_cudagraph(compute, rep=100)
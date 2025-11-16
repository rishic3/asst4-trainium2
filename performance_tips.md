For students who are passing correctness and struggling to optimize performance, here are some ideas to guide your thinking. These are based on the most common mistakes/misconceptions we're seeing in office hours.

Redundant data transfers: one of the biggest differences between NKI and standard programming models is the lack of an hardware-managed data cache. As a result, you might be tempted to use access patterns with nested DMAs like this (with syntax edited for clarity):

# a, b are HBM tensors of shape (n, n)
for i in range(n):
    res = 0
    for j in range(n):
        a_i_sbuf = dma_copy(a[i, :])
        b_j_sbuf = dma_copy(b[:, j])
        res += math_op(a_i_sbuf, b_j_sbuf)
    c[i] = dma_copy(res)

For normal CPU programs, successive accesses to/from the same memory locations would automatically result in a cache hit. But in the NKI code above, the dma_copy(a[i, :]) would execute the same DMA operation for every inner loop iteration. In that operation, each element of a slice of a is read from HBM to SBUF, and this would happen n times.

To fix this, you (the programmer) have to manage all data movement yourself. (Much like how you are fully responsible for moving data between CUDA global memory and CUDA shared memory yourself.) In this example -- supposing n is small enough for a and b to fit in SBUF -- you should be able to hoist DMAs out of those inner loops so each element is only transferred once.

Reducing the number of DMA transfers:
Similar to eliminating redundant data transfers by hoisting, it is also a good idea to check whether you are calling each dma_copy with sufficient data. As stated in the Part 1 spec:

The caveat is that there is an overhead cost when setting up a DMA transfer and assigning DMA engines to work on them. In order to reduce this setup overhead, efficient implementations should aim to move a large amount of data in each transfer to amortize DMA transfer overhead.
A rule of thumb: 128 * 128 would be too small, but 128 * 128 * H * W would be too large. (Check both your input image X and the weight matrix!) You will need to find a suitable amount of data in between by experimentation. If the dma_copy is too small, allocate a large enough SBUF buffer and move more data between SBUF <-> HBM with bigger chunks. Allocate the larger SBUF at one of the parent-level for loops (hoisting).

For example, consider a NumPy array a in HBM with shape (M, K, 128, 128). Instead of doing this: 

for m in range(M):
    for k in range(K):
        # allocate an sbuf with shape (128, 128)
        sbuf = dma_copy(a[m, k])
        # compute with sbuf
Consider changing this to:

for m in range(M):
    # allocate an sbuf with shape (K, 128, 128)
    sbuf = dma_copy(a[m, :])
    for k in range(K):
        # compute with sbuf[k]
Load fewer rows from X: Building on the last two points, for max-pooling, you need to compute your output tile at least two rows at a time. Note that, to calculate each line of output, you would need 3 rows from the input matrix (whether you are loading 3 rows all at once or within an inner for loop ranging on filter_height). However, to calculate two output rows at a time, you would only need 1 additional row (so a total of 4 rows), because the middle two rows from the input can be used for both output rows. Because of this, we recommend loading those 4 rows at once and correctly indexing into them when you need them.

Using the correct API to transfer data within PSUM/SBUF: As mentioned in #692 (Thanks Nash and Aditya!), you should be using nisa.tensor_copy instead of nisa.dma_copy when copying data from SBUF -> SBUF.

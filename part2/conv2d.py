import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A fused convolution - maxpool kernel that you need to implement for Part 2.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.
    pool_size: the size of the pool filter and pool stride.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: pool_size == 1 || pool_size == 2
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height // pool_size
out_pool_width = out_width // pool_size

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]

"""
@nki.compiler.skip_middle_end_transformations
@nki.jit
def fused_conv2d_maxpool(X, W, bias, pool_size=1):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height // pool_size
    out_pool_width = out_width // pool_size
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == out_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax

    tile_c_in = c_in_pmax
    tile_c_out = nl.tile_size.gemm_stationary_fmax
    rows_per_block = pool_size if pool_size > 1 else 1
    num_row_blocks = out_height // rows_per_block
    bias_2d = bias.reshape((out_channels, 1))

    for b in nl.affine_range(batch_size):
        for oc_block in nl.affine_range(out_channels // tile_c_out):
            c_out_start = oc_block * tile_c_out
            c_out_end = c_out_start + tile_c_out

            bias_tile = nl.ndarray((tile_c_out, 1), dtype=bias.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                src=bias_2d[c_out_start:c_out_end],
                dst=bias_tile,
            )

            for block_idx in nl.affine_range(num_row_blocks):
                row_buffer = nl.ndarray(
                    (tile_c_out, rows_per_block * out_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf,
                )

                for row_in_block in nl.affine_range(rows_per_block):
                    out_row = block_idx * rows_per_block + row_in_block
                    res_psum = nl.zeros(
                        (tile_c_out, out_width), nl.float32, buffer=nl.psum
                    )

                    for fh in nl.affine_range(filter_height):
                        h_idx = out_row + fh
                        for fw in nl.affine_range(filter_width):
                            w_start = fw
                            for cin_block in nl.affine_range(n_tiles_c_in):
                                c_in_start = cin_block * tile_c_in
                                c_in_end = c_in_start + tile_c_in

                                rhs_tile = nl.ndarray(
                                    (tile_c_in, out_width),
                                    dtype=X.dtype,
                                    buffer=nl.sbuf,
                                )
                                w_tile = nl.ndarray(
                                    (tile_c_out, tile_c_in),
                                    dtype=W.dtype,
                                    buffer=nl.sbuf,
                                )

                                w_slice = W[
                                    c_out_start:c_out_end,
                                    c_in_start:c_in_end,
                                    fh,
                                    fw,
                                ]
                                nisa.dma_copy(src=w_slice, dst=w_tile)

                                w_tile_psum = nisa.nc_transpose(w_tile)
                                lhs_tile = nisa.tensor_copy(
                                    src=w_tile_psum, dtype=W.dtype
                                )

                                x_slice = X[
                                    b,
                                    c_in_start:c_in_end,
                                    h_idx,
                                    w_start : w_start + out_width,
                                ]
                                nisa.dma_copy(src=x_slice, dst=rhs_tile)

                                res_psum += nisa.nc_matmul(lhs_tile, rhs_tile)

                    row_tile = nl.copy(res_psum, dtype=X.dtype)
                    row_offset = row_in_block * out_width
                    nisa.dma_copy(
                        src=row_tile,
                        dst=row_buffer[:, row_offset : row_offset + out_width],
                    )

                if pool_size == 1:
                    for w in nl.affine_range(out_width):
                        col = row_buffer[:, w : w + 1]
                        biased = nisa.tensor_scalar(col, nl.add, bias_tile)
                        nisa.dma_copy(src=biased, dst=row_buffer[:, w : w + 1])

                    dest = X_out[
                        b,
                        c_out_start:c_out_end,
                        block_idx,
                        :,
                    ]
                    nisa.dma_copy(
                        src=row_buffer[:, 0:out_width],
                        dst=dest,
                    )
                else:
                    pooled_tile = nl.ndarray(
                        (tile_c_out, out_pool_width),
                        dtype=X.dtype,
                        buffer=nl.sbuf,
                    )

                    for pw in nl.affine_range(out_pool_width):
                        base = pw * pool_size
                        row0_col0 = row_buffer[:, base : base + 1]
                        row0_col1 = row_buffer[:, base + 1 : base + 2]
                        row0_max = nisa.tensor_tensor(row0_col0, row0_col1, op=nl.max)

                        row1_base = out_width + base
                        row1_col0 = row_buffer[:, row1_base : row1_base + 1]
                        row1_col1 = row_buffer[:, row1_base + 1 : row1_base + 2]
                        row1_max = nisa.tensor_tensor(row1_col0, row1_col1, op=nl.max)

                        pooled_col = nisa.tensor_tensor(row0_max, row1_max, op=nl.max)
                        nisa.dma_copy(
                            src=pooled_col,
                            dst=pooled_tile[:, pw : pw + 1],
                        )

                    for w in nl.affine_range(out_pool_width):
                        col = pooled_tile[:, w : w + 1]
                        biased = nisa.tensor_scalar(col, nl.add, bias_tile)
                        nisa.dma_copy(src=biased, dst=pooled_tile[:, w : w + 1])

                    dest = X_out[
                        b,
                        c_out_start:c_out_end,
                        block_idx,
                        :,
                    ]
                    nisa.dma_copy(
                        src=pooled_tile,
                        dst=dest,
                    )

    return X_out


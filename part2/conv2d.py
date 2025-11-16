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
    
    # 128 in/out channels per tile, 2 rows at a time
    IC_TILE = c_in_pmax
    OC_TILE = c_in_pmax
    H_TILE = 2

    n_oc_tiles = out_channels // OC_TILE
    n_ic_tiles = in_channels // IC_TILE
    n_h_tiles = out_height // H_TILE
    
    # hold all weights (transposed) in sbuf
    w_tr_sbuf = nl.ndarray(
        shape=(IC_TILE, OC_TILE, n_ic_tiles, n_oc_tiles, filter_height, filter_width),
        dtype=X.dtype,
        buffer=nl.sbuf
    )
    
    # allocate biases in sbuf
    bias_sbuf = nl.ndarray(
        shape=(OC_TILE, n_oc_tiles),
        dtype=nl.float32,
        buffer=nl.sbuf
    )
    
    # preload and transpose all weights into sbuf, matmuls will consume directly
    for oc_tile_idx in nl.affine_range(n_oc_tiles):
        oc_start = oc_tile_idx * OC_TILE
        oc_end = (oc_tile_idx + 1) * OC_TILE
        
        # load bias for this output channel tile
        nisa.dma_copy(
            dst=bias_sbuf[:, oc_tile_idx],
            src=bias[oc_start:oc_end]
        )
        
        # for each input-channel tile we will copy weights to sbuf and transpose once
        for ic_tile_idx in nl.affine_range(n_ic_tiles):
            ic_start = ic_tile_idx * IC_TILE
            ic_end = (ic_tile_idx + 1) * IC_TILE
            
            # weight buffer for this (oc tile, ic tile)
            w_raw_sbuf = nl.ndarray(
                shape=(OC_TILE, IC_TILE, filter_height, filter_width),
                dtype=X.dtype,
                buffer=nl.sbuf
            )

            nisa.dma_copy(
                dst=w_raw_sbuf,
                src=W[oc_start:oc_end, ic_start:ic_end, :, :]
            )
            
            # transpose the [oc, ic] matrix (becomes [ic, oc]) for each spatial position
            # store into weight buffer that lives for entire kernel
            for kernel_row in nl.affine_range(filter_height):
                for kernel_col in nl.affine_range(filter_width):
                    w_slice = w_raw_sbuf[:, :, kernel_row, kernel_col]
                    # tensor engine transpose, to psum
                    w_slice_tr_psum = nisa.nc_transpose(data=w_slice)
                    w_tr_sbuf[:, :, ic_tile_idx, oc_tile_idx, kernel_row, kernel_col] = nisa.tensor_copy(w_slice_tr_psum, dtype=w_raw_sbuf.dtype)
    
    # convolution + maxpool
    for batch_idx in nl.affine_range(batch_size):
        for oc_tile_idx in nl.affine_range(n_oc_tiles):
            oc_start = oc_tile_idx * OC_TILE
            oc_end = (oc_tile_idx + 1) * OC_TILE
            
            for h_tile_idx in nl.affine_range(n_h_tiles):
                h_start = h_tile_idx * H_TILE
                h_end = (h_tile_idx + 1) * H_TILE
                
                # need extra rows for filter window
                # e.g. 2 out rows, 3x3 filter => 4 input rows
                in_patch_h = H_TILE + filter_height - 1

                # load all ic tiles once upfront here into X_sbuf, reused all oc tiles
                # note: moved block dimension into free dimension here (ic_tile first in shape) to
                # avoid warning block dimension deprecation warning
                X_sbuf = nl.ndarray(
                    shape=(nl.par_dim(IC_TILE), n_ic_tiles, in_patch_h, input_width),
                    dtype=X.dtype,
                    buffer=nl.sbuf
                )
                for ic_tile_idx in nl.affine_range(n_ic_tiles):
                    ic_start = ic_tile_idx * IC_TILE
                    ic_end = (ic_tile_idx + 1) * IC_TILE
                    nisa.dma_copy(
                        dst=X_sbuf[:, ic_tile_idx, :, :],
                        src=X[batch_idx, ic_start:ic_end, h_start:h_end + filter_height - 1, :]
                    )
                
                # accumulator for this (batch, oc tile, h tile) region
                acc_psum = nl.zeros(
                    shape=(OC_TILE, H_TILE, out_width),
                    dtype=nl.float32,
                    buffer=nl.psum
                )
                
                # accumulation loop
                # contributions from all ic tiles, filter positions
                for ic_tile_idx in nl.affine_range(n_ic_tiles):
                    for f_row in nl.affine_range(filter_height):
                        for f_col in nl.affine_range(filter_width):
                            # pre-transposed weight slice for this tile
                            # IC_TILE x OC_TILE
                            w_tile_tr = w_tr_sbuf[:, :, ic_tile_idx, oc_tile_idx, f_row, f_col]
                            
                            # shifted input window for filter pos
                            # (0,0) -> rows[0:2], cols[0:out_width]
                            # (0,1) -> rows[0:2], cols[1:out_width+1]
                            # (1,0) -> rows[1:3], cols[0:out_width]
                            #...
                            # IC_TILE x H_TILE * out_width
                            x_win = X_sbuf[:, ic_tile_idx, f_row:f_row + H_TILE, f_col:f_col + out_width]
                            acc_psum += nisa.nc_matmul(w_tile_tr, x_win)

                if pool_size == 2:
                    # expand dims per tile
                    # 128 x 2 x out_width -> 128 x 1 x 2 x out_width//2 x 2
                    acc_psum_shape = (OC_TILE, H_TILE // pool_size, pool_size, out_pool_width, pool_size)
                    acc_for_pool = acc_psum.reshape(shape=acc_psum_shape)
                    pooled_psum = nisa.tensor_reduce(
                        op=nki.language.maximum,
                        data=acc_for_pool,
                        axis=(2, 4)
                    )  # 128 x 1 x out_width//2

                    pooled_sbuf = nisa.tensor_copy(pooled_psum, dtype=X.dtype)
                    pooled_plus_bias = nisa.tensor_tensor(
                        pooled_sbuf,
                        bias_sbuf[:, oc_tile_idx],  # 128 x 1
                        nki.language.add
                    )
                    
                    # copy to output, pooled coordinates are squashed by pool size
                    pooled_h_start = h_tile_idx * (H_TILE // pool_size)
                    pooled_h_end = (h_tile_idx + 1) * (H_TILE // pool_size)
                    nisa.dma_copy(
                        dst=X_out[batch_idx, oc_start:oc_end, pooled_h_start:pooled_h_end, :],
                        src=pooled_plus_bias
                    )
                else:
                    # just add bias in sbuf and copy to out
                    conv_sbuf = nisa.tensor_copy(acc_psum, dtype=X.dtype)
                    conv_plus_bias = nisa.tensor_tensor(
                        conv_sbuf,
                        bias_sbuf[:, oc_tile_idx],
                        nki.language.add
                    )
                    nisa.dma_copy(
                        dst=X_out[batch_idx, oc_start:oc_end, h_start:h_end, :],
                        src=conv_plus_bias
                    )
    
    return X_out

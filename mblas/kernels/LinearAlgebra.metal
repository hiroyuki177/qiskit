// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <complex_matrix/complex.h>
#include <complex_matrix/Constants.h>
#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;

namespace cm {
	
kernel void transpose(
	device const complex* mat		[[buffer(0)]],
	device complex* out   			[[buffer(1)]],
	constant uint& n_elements  		[[buffer(2)]],
	constant uint& n_dims  			[[buffer(3)]],
	constant uint* dim_size  		[[buffer(4)]],
	constant uint* idx_weights	  	[[buffer(5)]],
	uint group_id 					[[threadgroup_position_in_grid]],
	uint tid 						[[thread_position_in_threadgroup]]
) {
	uint x = group_id * MAX_GROUP_SIZE + tid;
	  
	if (x >= n_elements) return;

	threadgroup uint cached_dim_size[MAX_GROUP_SIZE];
	threadgroup uint cached_idx_weights[MAX_GROUP_SIZE];
	cached_dim_size[tid] = tid >= n_dims ? 0 : dim_size[tid];
	cached_idx_weights[tid] = tid >= n_dims ? 0 : idx_weights[tid];
	
	uint y = 0, idx = n_dims - 1, rem;
	while(x > 0) {
	  rem = x % cached_dim_size[idx];
	  x = x / cached_dim_size[idx];
	  y += rem * cached_idx_weights[idx--];
	}

	out[y] = mat[x];
}
	
kernel void matmul(
	device const complex* mat1Data 		[[buffer(0)]],
	device const complex* mat2Data 		[[buffer(1)]],
	device complex* outputData 			[[buffer(2)]],
    constant array<ulong2, 2>& strides  [[buffer(3)]],
	constant uint3& sizes 				[[buffer(4)]],
    uint2 group_id 						[[threadgroup_position_in_grid]],
	uint2 tid 							[[thread_position_in_threadgroup]]
) {
	uint col = group_id.x * TILE_DIM + tid.x;
	uint row = group_id.y * TILE_DIM + tid.y;
	
	complex zero{0.0f, 0.0f};
	complex sum = zero;

  threadgroup complex A_tile[TILE_DIM][TILE_DIM];
  threadgroup complex B_tile[TILE_DIM][TILE_DIM];

  uint numTiles = (sizes.y + TILE_DIM - 1) / TILE_DIM;
  for (uint t = 0; t < numTiles; t++) {
	uint tiledCol = t * TILE_DIM + tid.x;
	if (row < sizes.x && tiledCol < sizes.y) {
	  A_tile[tid.y][tid.x] =
		  mat1Data[row * strides[0].x + tiledCol * strides[0].y];
	} else {
	  A_tile[tid.y][tid.x] = zero;
	}

	uint tiledRow = t * TILE_DIM + tid.y;
	if (tiledRow < sizes.y && col < sizes.z) {
	  B_tile[tid.y][tid.x] =
		  mat2Data[tiledRow * strides[1].x + col * strides[1].y];
	} else {
	  B_tile[tid.y][tid.x] = zero;
	}

	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint k = 0; k < TILE_DIM; k++) {
		sum = sum + complex::multiply(A_tile[tid.y][k], B_tile[k][tid.x]);
	}

	threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (row < sizes.x && col < sizes.z) {
	outputData[row * sizes.z + col] = sum;
  }
}

/**
	Faster matrix multiplication for dimensions are multiples of 8. The warp size is 32.
 */
kernel void matmul_simd_8(
	device const float* mat1Data 		[[buffer(0)]],
	device const float* mat2Data 		[[buffer(1)]],
	device float* outputData 			[[buffer(2)]],
	constant array<ulong2, 2>& strides  [[buffer(3)]],
	constant uint3& sizes 				[[buffer(4)]],
	uint2 group_id 						[[threadgroup_position_in_grid]],
	uint2 tid 							[[thread_position_in_threadgroup]]
) {
	float num;
	uint y = tid.x / 8, x = tid.x % 8;
	uint row = group_id.y * 8 + y, col = group_id.x * 8 + x, idx;
	
	threadgroup float tileA_re[8][8], tileA_im[8][8], tileA_im_neg[8][8];
	threadgroup float tileB_re[8][8], tileB_im[8][8];
	
	simdgroup_float8x8 simdA_re, simdA_im, simdA_im_neg;
	simdgroup_float8x8 simdB_re, simdB_im;
	simdgroup_float8x8 simdC_re(0.0f), simdC_im(0.0f);

	for (uint t = 0; t < sizes.y; t += 8) {
		idx = row * strides[0].x + (t + x) * strides[0].y;
		tileA_re[y][x] = mat1Data[idx];
		num = mat1Data[idx + 1]; tileA_im[y][x] = num; tileA_im_neg[y][x] = -num;
		idx += 4 * strides[0].x;
		tileA_re[y + 4][x] = mat1Data[idx];
		num = mat1Data[idx + 1]; tileA_im[y + 4][x] = num; tileA_im_neg[y + 4][x] = -num;
		
		idx = (t + y) * strides[1].x + col * strides[1].y;
		tileB_re[y][x] = mat2Data[idx];
		tileB_im[y][x] = mat2Data[idx + 1];
		tileB_re[y + 4][x] = mat2Data[idx + 4 * strides[1].x];
		tileB_im[y + 4][x] = mat2Data[idx + 4 * strides[1].x + 1];
		
		
		threadgroup_barrier(mem_flags::mem_threadgroup);
		
		simdgroup_load(simdA_re, 	 (threadgroup float *)tileA_re);
		simdgroup_load(simdA_im, 	 (threadgroup float *)tileA_im);
		simdgroup_load(simdA_im_neg, (threadgroup float *)tileA_im_neg);
		simdgroup_load(simdB_re, 	 (threadgroup float *)tileB_re);
		simdgroup_load(simdB_im, 	 (threadgroup float *)tileB_im);
		
		simdgroup_multiply_accumulate(simdC_re, simdA_re, simdB_re, simdC_re);
		simdgroup_multiply_accumulate(simdC_re, simdA_im_neg, simdB_im, simdC_re);
		simdgroup_multiply_accumulate(simdC_im, simdA_re, simdB_im, simdC_im);
		simdgroup_multiply_accumulate(simdC_im, simdA_im, simdB_re, simdC_im);
	}
	
	threadgroup_barrier(mem_flags::mem_threadgroup);
	
	// for temporary storage
	simdgroup_store(simdC_re, (threadgroup float *)tileA_re);
	simdgroup_store(simdC_im, (threadgroup float *)tileA_im);
	
	threadgroup_barrier(mem_flags::mem_threadgroup);
	
	idx = row * sizes.z * 2 + col * 2; // double strides
	outputData[idx] 				  = tileA_re[y][x];
	outputData[idx + 1] 			  = tileA_im[y][x];
	outputData[idx + sizes.z * 8] 	  = tileA_re[y + 4][x];
	outputData[idx + sizes.z * 8 + 1] = tileA_im[y + 4][x];
}

/**
	Faster matrix multiplication for dimensions are multiples of 16. The warp size is 32.
 */
kernel void matmul_simd_16(
	device const float* mat1Data 		[[buffer(0)]],
	device const float* mat2Data 		[[buffer(1)]],
	device float* outputData 			[[buffer(2)]],
	constant array<ulong2, 2>& strides  [[buffer(3)]],
	constant uint3& sizes 				[[buffer(4)]],
	uint2 group_id 						[[threadgroup_position_in_grid]],
	uint sgid							[[simdgroup_index_in_threadgroup]],
	uint tid							[[thread_index_in_simdgroup]]
) {
	uint block_y = sgid / 2, block_x = sgid % 2;
	uint y = tid / 8, x = tid % 8;
	uint row = group_id.y * 16 + block_y * 8 + y, col = group_id.x * 16 + block_x * 8 + x, idx;
	
	threadgroup float tileA[6][8][8]; // 2 for real, 2 for imag, 2 for negative img
	threadgroup float tileB[4][8][8]; // 2 for real, 2 for imag
	
	simdgroup_float8x8 simdA_re, simdA_im, simdA_im_neg;
	simdgroup_float8x8 simdB_re, simdB_im;
	simdgroup_float8x8 simdC_re(0.0f), simdC_im(0.0f);

	for (uint t = 0; t < sizes.y; t += 8) {
		idx = (row + block_x * 4) * strides[0].x + (t + x) * strides[0].y;
		tileA[block_y][y + block_x * 4][x]     = mat1Data[idx];
		tileA[block_y + 2][y + block_x * 4][x] = mat1Data[idx + 1];
		tileA[block_y + 4][y + block_x * 4][x] = -mat1Data[idx + 1];
		
		idx = (t + y + block_y * 4) * strides[1].x + col * strides[1].y;
		tileB[block_x][y + block_y * 4][x]     = mat2Data[idx];
		tileB[block_x + 2][y + block_y * 4][x] = mat2Data[idx + 1];
		
		threadgroup_barrier(mem_flags::mem_threadgroup);
		
		simdgroup_load(simdA_re, 	 (threadgroup float *)tileA[block_y]);
		simdgroup_load(simdA_im, 	 (threadgroup float *)tileA[block_y + 2]);
		simdgroup_load(simdA_im_neg, (threadgroup float *)tileA[block_y + 4]);
		simdgroup_load(simdB_re, 	 (threadgroup float *)tileB[block_x]);
		simdgroup_load(simdB_im, 	 (threadgroup float *)tileB[block_x + 2]);
		
		simdgroup_multiply_accumulate(simdC_re, simdA_re, simdB_re, simdC_re);
		simdgroup_multiply_accumulate(simdC_re, simdA_im_neg, simdB_im, simdC_re);
		simdgroup_multiply_accumulate(simdC_im, simdA_re, simdB_im, simdC_im);
		simdgroup_multiply_accumulate(simdC_im, simdA_im, simdB_re, simdC_im);
	}

	threadgroup_barrier(mem_flags::mem_threadgroup);
	
	// for temporary storage
	simdgroup_store(simdC_re, (threadgroup float *)tileA[sgid]);
	simdgroup_store(simdC_im, (threadgroup float *)tileB[sgid]);
	
	threadgroup_barrier(mem_flags::mem_threadgroup);
	
	idx = row * sizes.z * 2 + col * 2; // double strides
	outputData[idx] 				  = tileA[sgid][y][x];
	outputData[idx + 1] 			  = tileB[sgid][y][x];
	outputData[idx + sizes.z * 8] 	  = tileA[sgid][y + 4][x];
	outputData[idx + sizes.z * 8 + 1] = tileB[sgid][y + 4][x];
}
	
} // namespace cm

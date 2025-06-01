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
using namespace metal;

namespace cm {
	

kernel void ctrl_map_matrix(
	device const complex* base		[[buffer(0)]],
	device complex* target   		[[buffer(1)]],
	constant uint& dim_base  		[[buffer(2)]],
	constant uint& dim_ctrl  		[[buffer(3)]],
	constant uint& ctrl_pos  		[[buffer(4)]],
	uint2 group_id 					[[threadgroup_position_in_grid]],
	uint2 tid 						[[thread_position_in_threadgroup]]
) {
	uint i = group_id.y * TILE_DIM + tid.y;
	uint j = group_id.x * TILE_DIM + tid.x;
	uint dim = dim_base * dim_ctrl;
	
	if (i >= dim_base || j >= dim_base) return;
	
	target[(i * dim_ctrl + ctrl_pos) * dim + j * dim_ctrl + ctrl_pos] = base[i * dim_base + j];
}


kernel void ctrl_diag_matrix(
	device complex* mat      		[[buffer(0)]],
	constant uint& dim         		[[buffer(1)]],
	constant complex& scalar 		[[buffer(2)]],
	constant uint& dim_ctrl  		[[buffer(3)]],
	constant uint& ctrl_pos  		[[buffer(4)]],
	uint group_id 					[[threadgroup_position_in_grid]],
	uint tid          				[[thread_position_in_threadgroup]]

) {
	uint x = group_id * GROUP_SIZE + tid;
	
	if (x >= dim || x % dim_ctrl == ctrl_pos) return;
	
	mat[x * dim + x] = scalar;
}
	
} // namespace cm





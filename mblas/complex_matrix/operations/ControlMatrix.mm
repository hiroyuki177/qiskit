// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <Foundation/Foundation.h>
#include <complex_matrix/Stream.h>
#include <complex_matrix/Library.h>
#include <complex_matrix/Device.h>
#include <complex_matrix/complex.h>
#include <complex_matrix/Constants.h>
#include <complex_matrix/Api.h>

namespace cm {
	
void control_mat(tensor *base_mat, tensor *out_mat, uint32_t dim_ctrl, uint32_t ctrl_pos) {
	assert(VALID_2_TENSORS(base_mat, out_mat));
	
	uint32_t dim_base = base_mat->rows;
	uint32_t dim = dim_base * dim_ctrl;
	complex one(1.0f, 0.0f);
	
	auto stream = getCurrentStream();
	auto device = Device::getInstance().device();
	auto encoder = stream->commandEncoder();
	auto diag_pso = lib.getPipelineState("cm::ctrl_diag_matrix");
	auto map_pso = lib.getPipelineState("cm::ctrl_map_matrix");
	@autoreleasepool {
		auto base_buffer = INPUT_TENSOR_TO_BUFFER(base_mat, dim_base * dim_base * sizeof(complex));
		auto out_buffer = OUTPUT_TENSOR_TO_BUFFER(out_mat, dim * dim * sizeof(complex));
		
		[encoder setComputePipelineState:diag_pso];
		[encoder setBuffer:out_buffer offset:0 atIndex:0];
		[encoder setBytes:&dim length:sizeof(uint32_t) atIndex:1];
		[encoder setBytes:&one length:sizeof(complex) atIndex:2];
		[encoder setBytes:&dim_ctrl length:sizeof(uint32_t) atIndex:3];
		[encoder setBytes:&ctrl_pos length:sizeof(uint32_t) atIndex:4];
		uint32_t grid_size = (dim + GROUP_SIZE - 1) / GROUP_SIZE;
		MTLSize threadsPerThreadgroup = MTLSizeMake(GROUP_SIZE, 1, 1);
		MTLSize threadgroupsPerGrid = MTLSizeMake(grid_size, 1, 1);
		[encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
		
		[encoder setComputePipelineState:map_pso];
		[encoder setBuffer:base_buffer offset:0 atIndex:0];
		[encoder setBuffer:out_buffer offset:0 atIndex:1];
		[encoder setBytes:&dim_base length:sizeof(uint32_t) atIndex:2];
		[encoder setBytes:&dim_ctrl length:sizeof(uint32_t) atIndex:3];
		[encoder setBytes:&ctrl_pos length:sizeof(uint32_t) atIndex:4];
		uint32_t gridSizePerDim = (dim_base + TILE_DIM - 1) / TILE_DIM;
		threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
		threadgroupsPerGrid = MTLSizeMake(gridSizePerDim, gridSizePerDim, 1);
		[encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
		stream->synchronize(out_mat->type == ARRAY_TYPE ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT_AND_CONTINUE);
		
		if(IS_PRIVATE_BUFFER(out_mat))
			out_mat->tensor = (void *)CFBridgingRetain(out_buffer);
	}
}
	
} // namespace cm


void get_control_matrix(tensor *t, tensor *o, uint32_t dc, uint32_t cp) {
	cm::control_mat(t, o, dc, cp);
}

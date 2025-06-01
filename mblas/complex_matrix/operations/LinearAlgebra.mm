// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <Foundation/Foundation.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <complex_matrix/Stream.h>
#include <complex_matrix/Library.h>
#include <complex_matrix/Device.h>
#include <complex_matrix/complex.h>
#include <complex_matrix/Constants.h>
#include <complex_matrix/Api.h>

#include <array>

namespace cm {

// tanspose n-dims array with restriction that n should no more than MAX_GROUP_SIZE
void transpose(tensor *data, tensor *out_data, uint32_t *dims, uint32_t *axes, uint32_t n_dims) {
	assert(VALID_2_TENSORS(data, out_data));
	assert(n_dims > 1 && n_dims <= MAX_GROUP_SIZE);
	
	uint32_t n_elements = 1;
	uint32_t *idx_weights = (uint32_t *)malloc(n_dims * sizeof(uint32_t));
	for (int i = n_dims - 1; i >= 0 ; --i) {
		idx_weights[axes[i]] = n_elements;
		n_elements *= dims[axes[i]];
	}
	
	auto stream = getCurrentStream();
	auto device = Device::getInstance().device();
	auto encoder = stream->commandEncoder();
	auto pso = lib.getPipelineState("cm::transpose");
	auto allow_no_copy = data->tensor != out_data->tensor;
	@autoreleasepool {
		auto in_buffer = INPUT_TENSOR_TO_BUFFER(data, n_elements * sizeof(complex));
		auto out_buffer = OUTPUT_TENSOR_TO_BUFFER_CHECKED(out_data, allow_no_copy, n_elements * sizeof(complex));
		
		[encoder setComputePipelineState:pso];
		[encoder setBuffer:in_buffer offset:0 atIndex:0];
		[encoder setBuffer:out_buffer offset:0 atIndex:1];
		[encoder setBytes:&n_elements length:sizeof(uint32_t) atIndex:2];
		[encoder setBytes:&n_dims length:sizeof(uint32_t) atIndex:3];
		[encoder setBytes:dims length:n_dims * sizeof(uint32_t) atIndex:4];
		[encoder setBytes:idx_weights length:n_dims * sizeof(uint32_t) atIndex:5];
		uint32_t grid_size = (n_elements + MAX_GROUP_SIZE - 1) / MAX_GROUP_SIZE;
		MTLSize threadsPerThreadgroup = MTLSizeMake(MAX_GROUP_SIZE, 1, 1);
		MTLSize threadgroupsPerGrid = MTLSizeMake(grid_size, 1, 1);
		[encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

		PROCESS_OUT_BUFFER_RETURN(out_data, allow_no_copy, out_buffer, in_buffer);
	}
	free(idx_weights);
}

void matrix_multiplication(tensor *A, tensor *B, tensor *C, int32_t A_transpose, int32_t B_transpose, int32_t prefer_naive_matmul) {
	assert(VALID_3_TENSORS(A, B, C));
	
	uint32_t x = A->rows, yA = A->cols, yB = B->rows, z = B->cols;
	std::array<uint64_t, 4> strides = {yA, 1, z, 1};
	if(A_transpose) { yA = x; x = A->cols; strides[0] = 1; strides[1] = x; }
	if(B_transpose) { z = yB; yB = B->cols; strides[2] = 1; strides[3] = yB; }
	std::array<uint32_t, 4> sizes = {x, yA, z, 0};
	
	assert(yA == yB);
	
	auto stream = getCurrentStream();
	auto device = Device::getInstance().device();
	auto encoder = stream->commandEncoder();
	auto allow_no_copy = C->tensor != A->tensor && C->tensor != B->tensor;

	@autoreleasepool {
		auto A_buffer = INPUT_TENSOR_TO_BUFFER(A, x * yA * sizeof(complex));
		auto B_buffer = INPUT_TENSOR_TO_BUFFER(B, yB * z * sizeof(complex));
		auto out_buffer = OUTPUT_TENSOR_TO_BUFFER_CHECKED(C, allow_no_copy, x * z * sizeof(complex));
		
		MTLComputePipelineState_t pso;
		MTLSize threadsPerThreadgroup, threadgroupsPerGrid;
		if (!prefer_naive_matmul && [device supportsFamily:MTLGPUFamilyApple7] && x % 8 == 0 && yA % 8 == 0 && z % 8 == 0) {
			if (x % 16 == 0 && yA % 16 == 0 && z % 16 == 0) {
				pso = lib.getPipelineState("cm::matmul_simd_16");
				threadsPerThreadgroup = MTLSizeMake(128, 1, 1);
				threadgroupsPerGrid = MTLSizeMake(z / 16, x / 16, 1);
			} else {
				pso = lib.getPipelineState("cm::matmul_simd_8");
				threadsPerThreadgroup = MTLSizeMake(32, 1, 1);
				threadgroupsPerGrid = MTLSizeMake(z / 8, x / 8, 1);
			}
			assert([pso threadExecutionWidth] == WARP_SIZE);
			for (auto& stride : strides) stride *= sizeof(complex) / sizeof(float); // double the stride
		} else {
			pso = lib.getPipelineState("cm::matmul");
			threadsPerThreadgroup = MTLSizeMake(TILE_DIM, TILE_DIM, 1);
			threadgroupsPerGrid = MTLSizeMake((z + TILE_DIM - 1) / TILE_DIM, (x + TILE_DIM - 1) / TILE_DIM, 1);
		}
		
		[encoder setComputePipelineState:pso];
		[encoder setBuffer:A_buffer offset:0 atIndex:0];
		[encoder setBuffer:B_buffer offset:0 atIndex:1];
		[encoder setBuffer:out_buffer offset:0 atIndex:2];
		[encoder setBytes:&strides length:sizeof(std::array<uint64_t, 4>) atIndex:3];
		[encoder setBytes:&sizes length:sizeof(std::array<uint32_t, 4>) atIndex:4];
		[encoder dispatchThreadgroups:threadgroupsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];	
		
		PROCESS_OUT_BUFFER_RETURN(C, allow_no_copy, out_buffer, C->tensor == A->tensor ? A_buffer : B_buffer);
	}
}

void matmul_mps(complex *A, complex *B, complex *C, matmul_options& options) {
	uint32_t x = options.A_dims[0], yA = options.A_dims[1], yB = options.B_dims[0], z = options.B_dims[1];
	std::array<uint64_t, 4> strides = {yA, 1, z, 1};
	if(options.A_transpose) { yA = x; x = options.A_dims[1]; strides[0] = 1; strides[1] = x; }
	if(options.B_transpose) { z = yB; yB = options.B_dims[1]; strides[2] = 1; strides[3] = yB; }
	NSNumber *X = @(x), *Y = @(yA), *Z = @(z);
	
	assert(yA == yB);
	
	auto stream = getCurrentStream();
	auto device = Device::getInstance().device();

	@autoreleasepool {
		auto A_buffer = [device newBufferWithBytesNoCopy:A length:x * yA * sizeof(complex) options:MTLResourceStorageModeShared deallocator:nullptr];
		auto B_buffer = [device newBufferWithBytesNoCopy:B length:yB * z * sizeof(complex) options:MTLResourceStorageModeShared deallocator:nullptr];
		auto o_buffer = [device newBufferWithBytesNoCopy:C length:x * z * sizeof(complex) options:MTLResourceStorageModeShared deallocator:nullptr];
		
		auto graph = [[MPSGraph alloc] init];
		auto MA = [graph placeholderWithShape:@ [X, Y] dataType:MPSDataTypeComplexFloat32 name:@"A"];
		auto MB = [graph placeholderWithShape:@ [Y, Z] dataType:MPSDataTypeComplexFloat32 name:@"B"];
		auto MC = [graph matrixMultiplicationWithPrimaryTensor:MA secondaryTensor:MB name:nil];
		auto DataA = [[MPSGraphTensorData alloc] initWithMTLBuffer:A_buffer shape:@ [X, Y] dataType:MPSDataTypeComplexFloat32];
		auto DataB = [[MPSGraphTensorData alloc] initWithMTLBuffer:B_buffer shape:@ [Y, Z] dataType:MPSDataTypeComplexFloat32];
		auto DataC = [[MPSGraphTensorData alloc] initWithMTLBuffer:o_buffer shape:@ [X, Z] dataType:MPSDataTypeComplexFloat32];
		
		NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
			MA: DataA, MB: DataB
		};
		
		NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
			MC: DataC
		};

		[graph encodeToCommandBuffer:stream->commandBuffer() feeds:feeds targetOperations:nil resultsDictionary:results executionDescriptor:nil];
		stream->synchronize(SyncType::COMMIT_AND_WAIT);
	}
}

} // namespace cm

void transpose(tensor *d, tensor *o, uint32_t *dims, uint32_t *ax, uint32_t n_dims) {
	cm::transpose(d, o, dims, ax, n_dims);
}

void matmul(tensor *A, tensor *B, tensor *C, uint32_t TA, uint32_t TB, uint32_t metal) {
	cm::matrix_multiplication(A, B, C, TA, TB, metal);
}

void matmul_mps(void *A, void *B, void *C, struct matmul_options options) {
	cm::matmul_mps((cm::complex *)A, (cm::complex *)B,(cm::complex *)C, options);
}

// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef Types_h
#define Types_h

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

typedef MPSCommandBuffer* MPSCommandBuffer_t;
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLBuffer> MTLBuffer_t;
typedef id<MTLDevice> MTLDevice_t;
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLBlitCommandEncoder> MTLBlitCommandEncoder_t;


#endif /* Types_h */

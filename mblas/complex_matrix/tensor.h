// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef tensor_h
#define tensor_h

#include <stdint.h>

enum TensorType {
	BUFFER_TYPE,
	ARRAY_TYPE
};

/* Tensor type to define the data. The ownership of tensor data should be managed by user as well.
 */
typedef struct tensor {
	void *tensor;
	enum TensorType type;
	uint32_t rows; // number of rows
	uint32_t cols; // number of columns
	uint32_t ref;  // invalid for output fields.
} tensor;

#define IS_PRIVATE_BUFFER(ptr) ((ptr)->type == BUFFER_TYPE && !(ptr)->tensor)

#define VALID_2_TENSORS(a, b) (a->tensor != b->tensor || a->type != BUFFER_TYPE)
#define VALID_3_TENSORS(a, b, c) (VALID_2_TENSORS(a, b) && VALID_2_TENSORS(a, c) && VALID_2_TENSORS(b, c))

/* Macro to determine the construcion of variable MTLBuffer_t in_buffer.
 */
#define INPUT_TENSOR_TO_BUFFER(ptr, size) 									\
	(ptr)->type == ARRAY_TYPE 												\
		? [device newBufferWithBytesNoCopy:(ptr)->tensor 					\
									length:(size) 							\
								   options:MTLResourceStorageModeShared 	\
							   deallocator:nullptr] 						\
		: ( (ptr)->ref ? (__bridge MTLBuffer_t) (ptr)->tensor 				\
					   : (MTLBuffer_t) CFBridgingRelease((ptr)->tensor)); 	\
if ((ptr)->type == BUFFER_TYPE && !(ptr)->ref) { (ptr)->tensor = nil; }

/* Macro to determine the construcion of variable MTLBuffer_t out_buffer.
 */
#define OUTPUT_TENSOR_TO_BUFFER(ptr, size) 							\
(ptr)->type == ARRAY_TYPE 											\
? [device newBufferWithBytesNoCopy:(ptr)->tensor	 				\
							length:(size) 							\
						   options:MTLResourceStorageModeShared 	\
					   deallocator:nullptr] 						\
: (!(ptr)->tensor 													\
   ? [device newBufferWithLength:(size) 							\
						 options:MTLResourceStorageModePrivate] 	\
   : (__bridge MTLBuffer_t) (ptr)->tensor)

/* Macro to determine the construcion of variable MTLBuffer_t out_buffer. Used for kernel handle functions that
 * allow the memory buffer in GPU can be kept for further operations. Several variables such as device are
 * required and should be declared.
 */
#define OUTPUT_TENSOR_TO_BUFFER_CHECKED(ptr, allow_no_copy, size) 							\
((ptr)->tensor && (ptr)->type != ARRAY_TYPE)												\
? (__bridge MTLBuffer_t) (ptr)->tensor 														\
: ((ptr)->tensor && (allow_no_copy) 														\
   ? [device newBufferWithBytesNoCopy:(ptr)->tensor											\
							   length:(size)												\
							  options:MTLResourceStorageModeShared							\
						  deallocator:nullptr]												\
   : [device newBufferWithLength:(size) options:MTLResourceStorageModePrivate])


/* Macro to process the copy and return procedure according to variable MTLBuffer_t out_buffer. Used for kernel
 * handle functions that allow the memory buffer in GPU can be kept for further operations. Several variables
 * such as device are required and should be declared.
 */
#define PROCESS_OUT_BUFFER_RETURN(out_ptr, allow_no_copy, out_buffer, to_buffer) 			\
if ((out_ptr)->type != ARRAY_TYPE) { 														\
	stream->synchronize(SyncType::COMMIT_AND_CONTINUE); 									\
	if (!(out_ptr)->tensor)																	\
		(out_ptr)->tensor = (void *)CFBridgingRetain(out_buffer); 							\
} else { 																					\
	if (!(allow_no_copy)) { 																\
		stream->synchronize(SyncType::COMMIT_AND_CONTINUE); 								\
		auto blitEncoder = [stream->commandBuffer() blitCommandEncoder]; 					\
		[blitEncoder copyFromBuffer:(out_buffer) 											\
					   sourceOffset:0 														\
						   toBuffer:(to_buffer) 											\
				  destinationOffset:0 														\
							   size:(out_buffer).length]; 									\
		[blitEncoder endEncoding]; 															\
	} 																						\
	stream->synchronize(SyncType::COMMIT_AND_WAIT); 										\
}




#endif /* tensor_h */

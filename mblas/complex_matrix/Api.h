// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef Api_h
#define Api_h

#include<complex_matrix/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

struct matmul_options {
	uint32_t A_dims[2];
	uint32_t B_dims[2];
	uint32_t A_transpose;
	uint32_t B_transpose;
	uint32_t prefer_naive_matmul;
	uint32_t keep_buffer;
};


// base matrix should differs from output matrix
__attribute__((visibility("default")))
void get_control_matrix(tensor *base, tensor *out_mat, uint32_t dim_ctrl, uint32_t ctrl_pos);

__attribute__((visibility("default")))
void transpose(tensor *data, tensor *out_data, uint32_t *dims, uint32_t *axes, uint32_t n_dims);

__attribute__((visibility("default")))
void matmul(tensor *A, tensor *B, tensor *C, uint32_t A_transpose, uint32_t B_transpose, uint32_t prefer_naive_matmul);

__attribute__((visibility("default")))
void matmul_mps(void *A, void *B, void *C, struct matmul_options options);

#ifdef __cplusplus
}
#endif
#endif /* Api_h */

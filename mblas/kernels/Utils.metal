// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <complex_matrix/Constants.h>
#include <metal_stdlib>
using namespace metal;

namespace cm {

kernel void inv_arrange_list(
	 constant uint* src      		[[buffer(0)]],
	 device uint* dest      		[[buffer(1)]],
	 constant uint& length  		[[buffer(2)]],
	 uint group_id 					[[threadgroup_position_in_grid]],
	 uint tid          				[[thread_position_in_threadgroup]]
) {
	uint x = group_id * GROUP_SIZE + tid;
	
	if (x >= length) return;
	dest[src[x]] = x;
}
	
} // namespace cm

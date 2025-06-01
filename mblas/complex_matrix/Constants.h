// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef Constants_h
#define Constants_h

#ifdef __METAL__
#include <metal_stdlib>
using namespace metal;
#endif

namespace cm {

#ifdef __METAL__
constant uint WARP_SIZE = 32;
constant uint TILE_DIM = 16;
constant uint GROUP_SIZE = TILE_DIM * TILE_DIM;
constant uint MAX_GROUP_SIZE = 1024;
#else
const uint32_t WARP_SIZE = 32;
const uint32_t TILE_DIM = 16;
const uint32_t GROUP_SIZE = TILE_DIM * TILE_DIM;
const uint32_t MAX_GROUP_SIZE = 1024;
#endif

} // namespace cm

#endif /* Constants_h */

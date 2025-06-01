// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <complex_matrix/Device.h>

namespace cm {

Device& Device::getInstance() {
    static Device device;
    return device;
}

Device::Device() : _device(nil) {
    // device may be an Intel GPU
    _device = MTLCreateSystemDefaultDevice();
}
    
Device::~Device() {
    _device = nil;
    
    assert(_device == nil);
}
    
} // namespace cm

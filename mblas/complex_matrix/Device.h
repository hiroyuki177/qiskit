// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef Device_h
#define Device_h

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <complex_matrix/Types.h>

namespace cm {

class Device {
    
public:
    Device(const Device& other) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&& other) = delete;
    Device& operator=(Device&&) = delete;

    static Device& getInstance();
    MTLDevice_t device() {
        return _device;
    }
    
 private:
    MTLDevice_t _device;
    Device();
    ~Device();
};

} // namespace cm

#endif /* Device_h */

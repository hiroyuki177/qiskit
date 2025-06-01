// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef Library_h
#define Library_h

#include <string>
#include <unordered_map>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <complex_matrix/Types.h>

namespace cm {

class Library {
    
public:
    Library(const Library& other) = delete;
    Library(Library&& other) = delete;
    Library& operator=(Library&&) = delete;
    static Library& getInstance();
    inline MTLComputePipelineState_t getPipelineState(const std::string& fname) {
        return getPipelineStateFunc(fname).first;
    }
    inline MTLFunction_t getFunction(const std::string& fname) {
        return getPipelineStateFunc(fname).second;
    }
    
 private:
    std::pair<MTLComputePipelineState_t, MTLFunction_t>
        getPipelineStateFunc(const std::string& fname);
    Library();
    ~Library();
    std::unordered_map<
        std::string,
        std::pair<MTLComputePipelineState_t, MTLFunction_t>>
        cplMap;
    MTLLibrary_t _library;
    
};

static Library& lib = Library::getInstance();

} // namespace cm

#endif /* Library_h */

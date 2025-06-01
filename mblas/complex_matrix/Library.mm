// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <complex_matrix/Library.h>
#include <complex_matrix/Device.h>
#include <complex_matrix/ComplexMatrixMetallib.h>

namespace cm {
    
Library& Library::getInstance() {
    static Library library;
    return library;
}
    
Library::Library() : _library(nil) {
    NSData* metallibData = [NSData dataWithBytes:metallib_data
                                          length:metallib_data_len];
    dispatch_queue_t queue = dispatch_queue_create(nil, DISPATCH_QUEUE_SERIAL);
	dispatch_data_t dispatchData = dispatch_data_create([metallibData bytes],
														[metallibData length],
														queue,
														DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError *error = nil;
    _library = [Device::getInstance().device() newLibraryWithData:dispatchData error:&error];
}
    
Library::~Library() {
    _library = nil;
	cplMap.clear();
    assert(_library == nil);
}
    
std::pair<MTLComputePipelineState_t, MTLFunction_t>
Library::getPipelineStateFunc(const std::string& fname) {
    auto it_cpl = cplMap.find(fname);
    if (it_cpl != cplMap.end())
        return it_cpl->second;
    
    NSError *error = nil;
    MTLFunction_t function = [_library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
    MTLComputePipelineState_t pipeline = [[_library device] newComputePipelineStateWithFunction:function error:&error];
    auto cpl = std::make_pair(pipeline, function);
    cplMap[fname] = cpl;
    return cpl;
}
    
    
} // namespace cm

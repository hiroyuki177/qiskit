// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#include <complex_matrix/Stream.h>


namespace cm {

Stream::Stream() {
    _commandQueue = [Device::getInstance().device() newCommandQueue];
}

Stream::~Stream() {
    _commandQueue = nil;
    
    assert(_commandBuffer == nil);
}
    
MPSCommandBuffer_t Stream::commandBuffer() {
    if(!_commandBuffer)
        _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue];
    
    return _commandBuffer;
}
    
MTLDevice_t Stream::device() const {
    return [_commandQueue device];
}
 
MTLComputeCommandEncoder_t Stream::commandEncoder() {
    if (!_commandEncoder)
        _commandEncoder = [commandBuffer() computeCommandEncoder];

    return _commandEncoder;
}

void Stream::synchronize(SyncType syncType) {
    endKernelCoalescing();
    switch (syncType) {
        case SyncType::NONE:
            // typically in GPU to GPU copies we won't commit explicitly
            break;
        case SyncType::COMMIT:
            commit();
            break;
        case SyncType::COMMIT_AND_WAIT:
            commitAndWait();
            break;
        case SyncType::COMMIT_AND_CONTINUE:
            commitAndContinue();
            break;
    }
}
    
void Stream::commit() {
    [commandBuffer() commit];
	_commandBuffer = nil;
}

void Stream::commitAndWait() {
    if (_commandBuffer) {
        [_commandBuffer commit];
        [_commandBuffer waitUntilCompleted];
        _commandBuffer = nil;
    }
}

void Stream::commitAndContinue() {
    assert(_commandBuffer);
    [_commandBuffer commitAndContinue];
}

void Stream::endKernelCoalescing() {
    if (_commandEncoder) {
        [_commandEncoder endEncoding];
        _commandEncoder = nil;
    }
}

void Stream::flush() {
    if (_commandBuffer) {
        [_commandBuffer commit];
        _commandBuffer = nil;
    }
}
    
    
Stream* StreamImpl::_stream = nullptr;

Stream* StreamImpl::getInstance() {
    if (_stream == nullptr)
        _stream = new Stream();
    
    return _stream;
}

StreamImpl::StreamImpl() {}

Stream* getCurrentStream() {
    return StreamImpl::getInstance();
}
    
} // namespace cm

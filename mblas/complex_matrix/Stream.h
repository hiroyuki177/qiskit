// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef Stream_h
#define Stream_h

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <complex_matrix/Device.h>
#include <complex_matrix/Types.h>


namespace cm {
    
enum class SyncType {
  NONE, // no commit to command buffer
  COMMIT, // commit and flush the command buffer
  COMMIT_AND_WAIT, // flush and wait for command buffer execution to finish
  COMMIT_AND_CONTINUE, // commit and continue with a new underlying command buffer
};


class Stream {

public:
    explicit Stream();
    ~Stream();

    MTLCommandQueue_t commandQueue() const {
        return _commandQueue;
    }

    MPSCommandBuffer_t commandBuffer();
    MTLComputeCommandEncoder_t commandEncoder();
    MTLDevice_t device() const;
    void synchronize(SyncType syncType);
    void endKernelCoalescing();
    // TODO: add operations that manipulate buffer data

private:
    MTLCommandQueue_t _commandQueue = nil;
    MPSCommandBuffer_t _commandBuffer = nil;
	MTLComputeCommandEncoder_t _commandEncoder = nil;

    void commit();
    void commitAndWait();
    void commitAndContinue();
    void flush();
};

Stream* getCurrentStream();

class StreamImpl {
public:
    static Stream* getInstance();

private:
    static Stream* _stream;
    StreamImpl();
};

} // namespace cm
#endif /* Stream_h */

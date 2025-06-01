// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

#ifndef complex_h
#define complex_h

#ifdef __METAL__
#include <metal_stdlib>
using namespace metal;
#else
#include <cstdlib>
#endif

namespace cm {

struct complex {
	float re, im;
	
	inline float sqabs() const { return re * re + im * im; }
	inline float abs() const { return sqrt(sqabs()); }
	inline complex conj() const { return complex{re, -im}; }

#ifdef __METAL__
	inline complex operator*(const thread complex& other) const {
#else
	inline complex operator*(const complex& other) const {
#endif
		return complex{re * other.re - im * other.im, re * other.im + im * other.re};
	}
		
#ifdef __METAL__
	inline complex operator*(const thread float& other) const {
#else
	inline complex operator*(const float& other) const {
#endif
		return complex{re * other, im * other};
	}
		
#ifdef __METAL__
	inline complex operator+(const thread complex& other) const {
#else
	inline complex operator+(const complex& other) const {
#endif
		return complex{re + other.re, im + other.im};
	}
		
#ifdef __METAL__
	inline complex operator-(const thread complex& other) const {
#else
	inline complex operator-(const complex& other) const {
#endif
		return complex{re - other.re, im - other.im};
	}
	
	inline complex operator!() const {
		return complex{re, -im};
	}
	
		
// for threadgroup address space
#ifdef __METAL__
	static inline complex multiply(const threadgroup complex& a, const threadgroup complex& b) {
		return complex{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
	}
#endif
		
};
		
} // namespace cm

#endif /* complex_h */

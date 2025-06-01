// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

extern crate bindgen;

use std::path::PathBuf;
use std::fs;

use bindgen::RustTarget;

fn main() {
    let libdir_path = PathBuf::from("../../mblas")
        .canonicalize()
        .expect("cannot canonicalize mblas path");
    let dylib = libdir_path.join("libComplexMatrix.dylib");
    let rust_target = match RustTarget::stable(70, 0) { 
        Ok(target) => target,
        Err(e) => panic!("unsupported Rust target: {}", e),
     };

    println!("cargo:rustc-link-search={}", libdir_path.to_str().unwrap());
    println!("cargo:rustc-link-lib=ComplexMatrix");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");

    if !std::process::Command::new("make")
        .current_dir(libdir_path)
        .output()
        .expect("could not spawn `make`")
        .status
        .success()
    {
        panic!("could not compile the library");
    }

    let outdir_path = PathBuf::from("../../qiskit")
        .canonicalize()
        .expect("cannot canonicalize qiskit path");
    fs::copy(&dylib, outdir_path.join("libComplexMatrix.dylib"))
        .expect("Failed to copy dynamic library");

    let bindings = bindgen::Builder::default()
        .rust_target(rust_target)
        .clang_arg("-I../../mblas")
        .header("include/wrapper.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from("src").join("bindings.rs");
    bindings
        .write_to_file(out_path)
        .expect("Couldn't write bindings!");
}

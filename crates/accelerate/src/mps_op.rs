// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use std::ffi::c_void;

use numpy::PyArrayMethods;
use numpy::PyUntypedArrayMethods;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::Python;

use num_traits::pow;
use num_complex::Complex;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArray1, PyArray2};

use qiskit_metalext::{get_control_matrix, matmul, transpose};
use qiskit_metalext::{tensor, TensorType_ARRAY_TYPE, TensorType_BUFFER_TYPE};

#[pyfunction]
#[pyo3(signature=(base_mat, n_ctrl_q, ctrl_state=None, /))]
pub unsafe fn compute_ctrl_matrix(
    py: Python,
    base_mat: PyReadonlyArray2<Complex<f32>>,
    n_ctrl_q: usize,
    ctrl_state: Option<usize>
) -> PyObject {
    let (dim_target, dim_ctrl) = (base_mat.shape()[0], pow(2_usize, n_ctrl_q));
    assert!(n_ctrl_q > 0 && dim_target > 0);

    let ctrl_pos = match ctrl_state {
        Some(pos) => pos,
        None => dim_ctrl - 1_usize,
    };

    let dim = dim_target * dim_ctrl;
    let mat = PyArray2::<Complex<f32>>::zeros_bound(py, [dim, dim], false);

    match dim_target {
        1..=32 => {
            let mut mat_mut = mat.as_array_mut();
            let mat_base = base_mat.as_array();
            (0..dim).for_each(|x| mat_mut[(x, x)] = Complex::<f32>::new(1.0_f32, 0.0_f32));
            (0..dim_target).for_each(|i| { 
                (0..dim_target).for_each(|j| mat_mut[(i * dim_ctrl + ctrl_pos, j * dim_ctrl + ctrl_pos)] = mat_base[(i, j)])
            });
        }
        _ => {
            let mat_ptr = mat.as_array_mut().as_ptr();
            let mut tensor_base = tensor { 
                tensor: base_mat.as_array().as_ptr() as *mut c_void, type_: TensorType_ARRAY_TYPE, rows: dim_target as u32, cols: dim_target as u32, ref_: 0 
            };
            let mut tensor_out = tensor { tensor: mat_ptr as *mut c_void, type_: TensorType_ARRAY_TYPE, rows: 0, cols: 0, ref_: 0 };
            get_control_matrix(&mut tensor_base,&mut tensor_out, dim_ctrl as u32, ctrl_pos as u32);
        }
    }
    mat.into()
}


#[pyfunction]
#[pyo3(signature=(vstate, oper, qargs=None, /))]
pub unsafe fn evolve_operator(
    py: Python,
    vstate: PyReadonlyArray1<Complex<f32>>,
    oper: PyReadonlyArray2<Complex<f32>>,
    qargs: Option<Vec<usize>>
) -> PyObject {
    let data = PyArray1::<Complex<f32>>::zeros_bound(py, vstate.dims(), false);
    let dim_op_l = oper.shape()[0] as u32;
    let dim_op_r = oper.shape()[1] as u32;
    let dim_vstate = vstate.shape()[0] as u32;
    let n_qubits = dim_vstate.trailing_zeros() as usize;
    let mut tensor_op = 
        tensor { tensor: oper.as_array().as_ptr() as *mut c_void, type_: TensorType_ARRAY_TYPE, rows: dim_op_l, cols: dim_op_r, ref_: 0 };
    let mut tensor_state = 
        tensor { tensor: vstate.as_array().as_ptr() as *mut c_void, type_: TensorType_ARRAY_TYPE, rows: dim_vstate, cols: 1, ref_: 0 };
    let mut tensor_data = 
        tensor { tensor: data.as_array_mut().as_ptr() as *mut c_void, type_: TensorType_ARRAY_TYPE, rows: dim_op_l, cols: 1, ref_: 0 };

    if let Some(qubits_to_apply) = qargs {
        let n_oper = qubits_to_apply.len();
        let mut flags = vec![false ; n_qubits];
        let mut axes = vec![0_u32 ; n_qubits];
        qubits_to_apply.into_iter().rev().enumerate().for_each(|(i, x)| {
            let y = n_qubits - x - 1;
            axes[i] = y as u32; 
            flags[y] = true;  
        });
        (0..n_qubits).filter(|&x| !flags[x]).fold(n_oper, |pos, num| {
            axes[pos] = num as u32;
            pos + 1
        });
        let mut axes_inv = vec![0_u32 ; n_qubits];
        (0..n_qubits).for_each(|i| axes_inv[axes[i] as usize] = i as u32);

        let mut dims: Vec<u32> = vec![2 ; n_qubits];
        let dim_others = dim_vstate / dim_op_r;
        let mut tensor_1 = 
            tensor { tensor: std::ptr::null_mut(), type_: TensorType_BUFFER_TYPE, rows: dim_op_r, cols: dim_others, ref_: 0 };
        let mut tensor_2 =  
            tensor { tensor: std::ptr::null_mut(), type_: TensorType_BUFFER_TYPE, rows: dim_op_r, cols: dim_others, ref_: 0 };
        transpose(&mut tensor_state, &mut tensor_1, dims.as_mut_ptr(), axes.as_mut_ptr(), n_qubits as u32);
        matmul(&mut tensor_op, &mut tensor_1, &mut tensor_2, 0, 0, 0);
        transpose(&mut tensor_2, &mut tensor_data, dims.as_mut_ptr(), axes_inv.as_mut_ptr(), n_qubits as u32);
    } else {
        matmul(&mut tensor_op, &mut tensor_state, &mut tensor_data, 0, 0, 0);
    }
    
    data.into()
}


pub fn mps_op(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(compute_ctrl_matrix))?;
    m.add_wrapped(wrap_pyfunction!(evolve_operator))?;
    Ok(())
}

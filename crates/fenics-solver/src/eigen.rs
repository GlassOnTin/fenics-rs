//! Generalized eigenvalue solver for structural modal analysis (K u = omega^2 M u).

use crate::cg::{dot, solve_cg, spmv};
use faer::sparse::SparseColMat;
use faer::Unbind;
use std::f64::consts::PI;

/// A natural vibration mode of a 3D structural system.
#[derive(Clone, Debug)]
pub struct VibrationMode {
    /// Natural frequency in Hertz (cycles per second)
    pub frequency_hz: f64,
    /// Angular frequency omega in radians per second (omega = 2 * pi * f)
    pub omega_rad_s: f64,
    /// Corresponding eigenvalue lambda = omega^2
    pub eigenvalue: f64,
    /// Mass-normalized mode shape vector (length = 3 * num_vertices)
    pub mode_shape: Vec<f64>,
}

/// Compute the lowest `num_modes` natural vibration modes of a structural system.
///
/// Parameters:
/// - `k`: Global 3D elasticity stiffness matrix
/// - `m`: Global 3D elasticity mass matrix
/// - `fixed_dofs`: List of constrained/fixed DOF indices (e.g. clamped root)
/// - `num_modes`: Number of lowest natural frequencies to compute
/// - `max_power_iters`: Maximum inverse power iterations per mode
/// - `tol`: Convergence tolerance
pub fn solve_vibration_modes(
    k: &SparseColMat<usize, f64>,
    m: &SparseColMat<usize, f64>,
    fixed_dofs: &[usize],
    num_modes: usize,
    max_power_iters: usize,
    tol: f64,
) -> Result<Vec<VibrationMode>, String> {
    let n_total = k.nrows();
    let mut is_fixed = vec![false; n_total];
    for &dof in fixed_dofs {
        if dof < n_total {
            is_fixed[dof] = true;
        }
    }

    // Map total DOFs -> free DOFs
    let mut free_dof_indices = Vec::new();
    let mut global_to_free = vec![None; n_total];
    for dof in 0..n_total {
        if !is_fixed[dof] {
            global_to_free[dof] = Some(free_dof_indices.len());
            free_dof_indices.push(dof);
        }
    }

    let n_free = free_dof_indices.len();
    if n_free == 0 {
        return Err("No free degrees of freedom in system".to_string());
    }

    // Extract submatrices for free DOFs: K_free, M_free
    let mut k_free_triplets = Vec::new();
    for col in 0..n_total {
        if let Some(col_free) = global_to_free[col] {
            let row_inds = k.as_ref().row_indices_of_col(col);
            let values = k.as_ref().values_of_col(col);
            for (row_idx, &val) in row_inds.zip(values) {
                if let Some(row_free) = global_to_free[row_idx.unbound()] {
                    k_free_triplets.push((row_free, col_free, val));
                }
            }
        }
    }
    let k_free = SparseColMat::try_new_from_triplets(n_free, n_free, &k_free_triplets).unwrap();

    let mut m_free_triplets = Vec::new();
    for col in 0..n_total {
        if let Some(col_free) = global_to_free[col] {
            let row_inds = m.as_ref().row_indices_of_col(col);
            let values = m.as_ref().values_of_col(col);
            for (row_idx, &val) in row_inds.zip(values) {
                if let Some(row_free) = global_to_free[row_idx.unbound()] {
                    m_free_triplets.push((row_free, col_free, val));
                }
            }
        }
    }
    let m_free = SparseColMat::try_new_from_triplets(n_free, n_free, &m_free_triplets).unwrap();

    let mut modes = Vec::with_capacity(num_modes);
    let mut free_mode_vectors: Vec<Vec<f64>> = Vec::with_capacity(num_modes);

    for mode_idx in 0..num_modes {
        // Deterministic pseudo-random initial vector based on mode index
        let mut v = vec![0.0; n_free];
        for i in 0..n_free {
            let s = ((i + 1) * 31 + (mode_idx + 1) * 97) % 1000;
            v[i] = (s as f64) / 1000.0 - 0.5;
        }

        // M-orthogonalize against existing modes
        for prev in &free_mode_vectors {
            let m_prev = spmv(&m_free, prev);
            let coeff = dot(&v, &m_prev);
            for i in 0..n_free {
                v[i] -= coeff * prev[i];
            }
        }

        // Normalize v: v^T M v == 1
        let mut mv = spmv(&m_free, &v);
        let norm_m = dot(&v, &mv).sqrt();
        if norm_m > 1e-14 {
            for x in &mut v {
                *x /= norm_m;
            }
        }

        let mut prev_rayleigh = 0.0;

        // Inverse Power Iteration loop
        for _iter in 0..max_power_iters {
            // Solve: K_free * w = M_free * v
            mv = spmv(&m_free, &v);
            let (mut w, _, _) = solve_cg(&k_free, &mv, Some(&v), 1e-9, 1000)?;

            // M-orthogonalize w against all lower modes
            for prev in &free_mode_vectors {
                let m_prev = spmv(&m_free, prev);
                let coeff = dot(&w, &m_prev);
                for i in 0..n_free {
                    w[i] -= coeff * prev[i];
                }
            }

            // Normalize w
            let mw = spmv(&m_free, &w);
            let mw_norm = dot(&w, &mw).sqrt();
            if mw_norm < 1e-15 {
                break;
            }
            for i in 0..n_free {
                w[i] /= mw_norm;
            }

            // Rayleigh quotient: lambda = w^T K w
            let kw = spmv(&k_free, &w);
            let rayleigh = dot(&w, &kw);

            v = w;

            if (rayleigh - prev_rayleigh).abs() / (rayleigh.abs().max(1.0)) < tol {
                break;
            }
            prev_rayleigh = rayleigh;
        }

        // Final Rayleigh quotient
        let kv = spmv(&k_free, &v);
        mv = spmv(&m_free, &v);
        let lambda = (dot(&v, &kv) / dot(&v, &mv)).max(0.0);
        let omega = lambda.sqrt();
        let freq_hz = omega / (2.0 * PI);

        // Expand to full 3N displacement vector
        let mut full_mode = vec![0.0; n_total];
        for (free_idx, &global_dof) in free_dof_indices.iter().enumerate() {
            full_mode[global_dof] = v[free_idx];
        }

        free_mode_vectors.push(v);
        modes.push(VibrationMode {
            frequency_hz: freq_hz,
            omega_rad_s: omega,
            eigenvalue: lambda,
            mode_shape: full_mode,
        });
    }

    Ok(modes)
}

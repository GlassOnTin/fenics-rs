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

/// Mean absolute diagonal entry, searched through the columns of a CSC matrix.
fn mean_abs_diag(mat: &SparseColMat<usize, f64>) -> f64 {
    let n = mat.nrows();
    if n == 0 {
        return 0.0;
    }
    let col = mat.as_ref();
    let mut acc = 0.0;
    let mut found = 0usize;
    for c in 0..n {
        let rows = col.row_indices_of_col(c);
        let vals = col.values_of_col(c);
        for (r, &v) in rows.zip(vals) {
            if r.unbound() == c {
                acc += v.abs();
                found += 1;
                break;
            }
        }
    }
    if found == 0 {
        return 0.0;
    }
    acc / found as f64
}

/// Dirichlet elimination without renumbering: entries on a constrained row or column are
/// dropped and, when `unit_diag` is set, the constrained diagonal becomes 1. Idempotent —
/// folding an already folded matrix changes nothing. Public because callers supplying their
/// own inner solve to [`solve_vibration_modes_with_inner`] fold their matrix the same way.
pub fn fold_dirichlet(
    mat: &SparseColMat<usize, f64>,
    fixed_dofs: &[usize],
    unit_diag: bool,
) -> SparseColMat<usize, f64> {
    let n = mat.nrows();
    let mut is_fixed = vec![false; n];
    for &dof in fixed_dofs {
        if dof < n {
            is_fixed[dof] = true;
        }
    }
    let mut triplets = Vec::new();
    for col in 0..n {
        let rows = mat.as_ref().row_indices_of_col(col);
        let values = mat.as_ref().values_of_col(col);
        for (row_idx, &val) in rows.zip(values) {
            if is_fixed[row_idx.unbound()] || is_fixed[col] || val == 0.0 {
                continue;
            }
            triplets.push((row_idx.unbound(), col, val));
        }
        if unit_diag && is_fixed[col] {
            triplets.push((col, col, 1.0));
        }
    }
    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

/// Lowest modes of a structural system, with the inner solve supplied by the caller.
///
/// The iteration runs in the Dirichlet frame on full-length vectors: `k` carries the
/// constraints folded in (constrained rows and columns zero, unit diagonal on them) and `m` has
/// constrained rows and columns zero. Both folds are idempotent and applied here before the
/// iteration, so callers pass whatever frame they hold — an unconstrained pair or their own
/// folded copy. Vectors keep the constrained entries at exactly 0: `m`'s rows vanish there and
/// `k`'s unit rows map the vanishing right-hand side to a vanishing solution entry.
///
/// `inner(b, x0)` must solve `k x = b` in this frame and is free to exploit its own
/// preconditioner over the same matrix. The default Jacobi-CG inner solve of
/// [`solve_vibration_modes`] saturates on large P2 systems; a caller holding a mesh-based
/// multilevel preconditioner passes one here. `x0` is the previous iterate, typically a good
/// starting guess.
pub fn solve_vibration_modes_with_inner(
    k: &SparseColMat<usize, f64>,
    m: &SparseColMat<usize, f64>,
    fixed_dofs: &[usize],
    num_modes: usize,
    max_power_iters: usize,
    tol: f64,
    inner: &dyn Fn(&[f64], &[f64]) -> Result<Vec<f64>, String>,
) -> Result<Vec<VibrationMode>, String> {
    let kd = fold_dirichlet(k, fixed_dofs, true);
    let md = fold_dirichlet(m, fixed_dofs, false);
    let n_total = kd.nrows();

    let mut is_fixed = vec![false; n_total];
    for &dof in fixed_dofs {
        if dof < n_total {
            is_fixed[dof] = true;
        }
    }
    let n_free = n_total - is_fixed.iter().filter(|&&f| f).count();
    if n_free == 0 {
        return Err("No free degrees of freedom in system".to_string());
    }

    // The mass is rescaled before the iteration. In unit systems where stiffness and density
    // live far apart (a mm-N-s model: density 7.85e-9 N·s²/mm⁴, stiffness entries ~1e5), the
    // eigen solver's inner CG solves K w = M v for a b = M v that carries the density's scale —
    // a 1e-9 relative residual on ‖Mv‖ ~ 1e-8 is a 1e-17 absolute target, past double
    // precision, and CG stalls with a near-zero curvature breakdown. Scaling M by
    // s = mean|diag K| / mean|diag M| puts both matrices on one numerical scale; the pair
    // (K, sM) has the same eigenvectors and eigenvalues λ/s, recovered exactly below. The
    // transformation is exact, not an approximation: frequencies are unaffected.
    let m_scale = {
        let kd_diag = mean_abs_diag(&kd);
        let md_diag = mean_abs_diag(&md);
        if kd_diag > 0.0 && md_diag > 0.0 {
            kd_diag / md_diag
        } else {
            1.0
        }
    };

    let mut modes = Vec::with_capacity(num_modes);
    let mut mode_vectors: Vec<Vec<f64>> = Vec::with_capacity(num_modes);
    let trace = std::env::var_os("FENICS_TRACE_EIGEN").is_some();

    for mode_idx in 0..num_modes {
        // Deterministic pseudo-random initial vector based on mode index (zero on the
        // constrained entries).
        let mut v = vec![0.0; n_total];
        for (i, val) in v.iter_mut().enumerate() {
            if is_fixed[i] {
                continue;
            }
            let s = ((i + 1) * 31 + (mode_idx + 1) * 97) % 1000;
            *val = (s as f64) / 1000.0 - 0.5;
        }

        // The deflated mass-inner-products of the converged modes do not change within a
        // mode's iteration — hoisted out of the power loop.
        let m_prev: Vec<Vec<f64>> = mode_vectors
            .iter()
            .map(|p| spmv(&md, p))
            .map(|y| y.into_iter().map(|y| y * m_scale).collect())
            .collect();

        // M-orthogonalize the initial vector against the converged modes.
        for (mp, prev) in m_prev.iter().zip(&mode_vectors) {
            let coeff = dot(&v, mp);
            for (i, val) in v.iter_mut().enumerate() {
                *val -= coeff * prev[i];
            }
        }

        // Normalize v: v^T s M v == 1
        let mut mv = spmv(&md, &v);
        for y in mv.iter_mut() {
            *y *= m_scale;
        }
        let norm_m = dot(&v, &mv).sqrt();
        if norm_m > 1e-14 {
            for x in &mut v {
                *x /= norm_m;
            }
        }

        let mut prev_rayleigh = 0.0f64;

        // Inverse Power Iteration loop
        for _iter in 0..max_power_iters {
            // Solve: K w = s M v
            mv = spmv(&md, &v);
            for y in mv.iter_mut() {
                *y *= m_scale;
            }
            let mut w = inner(&mv, &v)?;

            // The constrained entries sit at 0 by construction; clamp so a caller's
            // round-off cannot leak into the deflation.
            for (i, val) in w.iter_mut().enumerate() {
                if is_fixed[i] {
                    *val = 0.0;
                }
            }

            // M-orthogonalize w against all lower modes
            for (mp, prev) in m_prev.iter().zip(&mode_vectors) {
                let coeff = dot(&w, mp);
                for (i, val) in w.iter_mut().enumerate() {
                    *val -= coeff * prev[i];
                }
            }

            // Normalize w
            let mw = spmv(&md, &w);
            let mw_norm = dot(&w, &mw).sqrt() * m_scale.sqrt();
            if mw_norm < 1e-15 {
                break;
            }
            for val in &mut w {
                *val /= mw_norm;
            }

            // Rayleigh quotient: lambda = w^T K w
            let kw = spmv(&kd, &w);
            let rayleigh = dot(&w, &kw);
            if trace {
                eprintln!(
                    "eigen trace: mode {} iter {}: f = {:.6} Hz (true scale {:.4} Hz)",
                    mode_idx + 1,
                    _iter,
                    rayleigh.max(0.0).sqrt() / (2.0 * PI),
                    (rayleigh.max(0.0) * m_scale).sqrt() / (2.0 * PI)
                );
            }

            v = w;

            // Stop when the Rayleigh quotient has stopped moving, relative to its own
            // magnitude. The rescaled mass changes the quotient's absolute scale (m_scale
            // above), so the denominator may not anchor to a fixed 1.0 — that anchor made
            // the criterion absolute in the rescaled frame and stopped the iteration
            // wherever the scaled change dipped under 1e-7, up to a couple of percent off
            // the eigenvalue at the finer rungs.
            let denom = rayleigh.abs().max(prev_rayleigh.abs());
            if denom == 0.0 || (rayleigh - prev_rayleigh).abs() / denom < tol {
                break;
            }
            prev_rayleigh = rayleigh;
        }

        // Final Rayleigh quotient on the unscaled mass. The iterated vectors live on the free
        // subspace, where the scaled and unscaled quotients differ by exactly m_scale, so
        // dot(v, K v) / dot(v, M v) is the true eigenvalue.
        let kv = spmv(&kd, &v);
        let mu = spmv(&md, &v);
        let lambda = (dot(&v, &kv) / dot(&v, &mu)).max(0.0);
        let omega = lambda.sqrt();
        let freq_hz = omega / (2.0 * PI);

        // The mode shape in the full frame; constrained entries are exactly zero.
        let full_mode = v.clone();
        mode_vectors.push(v);
        modes.push(VibrationMode {
            frequency_hz: freq_hz,
            omega_rad_s: omega,
            eigenvalue: lambda,
            mode_shape: full_mode,
        });
    }

    Ok(modes)
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
///
/// The constraints are folded in here (see [`solve_vibration_modes_with_inner`]) and the inner
/// solve is Jacobi-preconditioned CG at 1e-9 — adequate for small systems. Large P2 meshes
/// exceed what diagonal preconditioning can reach (the residual flattens long before the
/// tolerance); pass a stronger inner solve through [`solve_vibration_modes_with_inner`].
pub fn solve_vibration_modes(
    k: &SparseColMat<usize, f64>,
    m: &SparseColMat<usize, f64>,
    fixed_dofs: &[usize],
    num_modes: usize,
    max_power_iters: usize,
    tol: f64,
) -> Result<Vec<VibrationMode>, String> {
    let kd = fold_dirichlet(k, fixed_dofs, true);
    solve_vibration_modes_with_inner(
        &kd,
        m,
        fixed_dofs,
        num_modes,
        max_power_iters,
        tol,
        &|b: &[f64], x0: &[f64]| solve_cg(&kd, b, Some(x0), 1e-9, 1000).map(|(w, _, _)| w),
    )
}

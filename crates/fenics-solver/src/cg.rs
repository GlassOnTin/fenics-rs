//! Preconditioned Conjugate Gradient (PCG) solver for symmetric positive-definite sparse systems.

use faer::sparse::SparseColMat;
use faer::Unbind;

/// Compute matrix-vector product y = A * x for a SparseColMat.
pub fn spmv(mat: &SparseColMat<usize, f64>, x: &[f64]) -> Vec<f64> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    assert_eq!(x.len(), ncols, "Vector dimension mismatch in SpMV");

    let mut y = vec![0.0; nrows];
    for col in 0..ncols {
        let x_val = x[col];
        if x_val != 0.0 {
            let row_inds = mat.as_ref().row_indices_of_col(col);
            let values = mat.as_ref().values_of_col(col);
            for (row_idx, &val) in row_inds.zip(values) {
                y[row_idx.unbound()] += val * x_val;
            }
        }
    }
    y
}

/// Dot product of two slices.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vector L2 norm.
#[inline]
pub fn norm2(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Solve A * x = b using Conjugate Gradient with Jacobi (diagonal) preconditioning.
///
/// Returns (solution_vector, iterations_performed, final_residual_norm).
pub fn solve_cg(
    mat: &SparseColMat<usize, f64>,
    b: &[f64],
    x0: Option<&[f64]>,
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, usize, f64), String> {
    let n = mat.nrows();
    assert_eq!(mat.ncols(), n, "Matrix must be square");
    assert_eq!(b.len(), n, "RHS length must match matrix dimension");

    // Extract inverse diagonal for Jacobi preconditioner: M^{-1}
    let mut inv_diag = vec![1.0; n];
    for col in 0..n {
        let row_inds = mat.as_ref().row_indices_of_col(col);
        let values = mat.as_ref().values_of_col(col);
        for (row_idx, &val) in row_inds.zip(values) {
            if row_idx.unbound() == col {
                if val.abs() > 1e-15 {
                    inv_diag[col] = 1.0 / val;
                }
                break;
            }
        }
    }

    let mut x = if let Some(init) = x0 {
        init.to_vec()
    } else {
        vec![0.0; n]
    };

    // r = b - A * x
    let ax = spmv(mat, &x);
    let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect();

    let b_norm = norm2(b);
    let initial_res = norm2(&r);
    if initial_res < tol || (b_norm > 0.0 && initial_res / b_norm < tol) {
        return Ok((x, 0, initial_res));
    }

    // z = M^{-1} * r
    let mut z: Vec<f64> = r.iter().zip(inv_diag.iter()).map(|(&ri, &mi)| ri * mi).collect();

    // p = z
    let mut p = z.clone();
    let mut rz_old = dot(&r, &z);

    for iter in 1..=max_iter {
        // Ap = A * p
        let ap = spmv(mat, &p);
        let p_ap = dot(&p, &ap);

        if p_ap.abs() < 1e-30 {
            return Err(format!("Breakdown in CG: curvature p^T A p is near zero at iteration {}", iter));
        }

        let alpha = rz_old / p_ap;

        // x = x + alpha * p
        // r = r - alpha * Ap
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let res = norm2(&r);
        let rel_res = if b_norm > 0.0 { res / b_norm } else { res };
        if rel_res < tol {
            return Ok((x, iter, res));
        }

        // z = M^{-1} * r
        for i in 0..n {
            z[i] = r[i] * inv_diag[i];
        }

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz_old;
        rz_old = rz_new;

        // p = z + beta * p
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
    }

    let final_res = norm2(&r);
    Err(format!(
        "CG failed to converge within {} iterations (final relative residual: {:.2e})",
        max_iter,
        if b_norm > 0.0 { final_res / b_norm } else { final_res }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_small_spd_system() {
        // [ 4  -1 ] [ x0 ] = [ 3 ]  => x0 = 1, x1 = 1
        // [-1   4 ] [ x1 ]   [ 3 ]
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 4.0),
        ];
        let mat = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let b = vec![3.0, 3.0];

        let (x, iters, res) = solve_cg(&mat, &b, None, 1e-10, 100).unwrap();
        assert!(iters <= 2);
        assert!(res < 1e-10);
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-8);
        assert_relative_eq!(x[1], 1.0, epsilon = 1e-8);
    }
}

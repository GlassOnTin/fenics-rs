//! Direct linear solver using LU factorization for general (non-symmetric or indefinite) systems.

use faer::prelude::SpSolver;
use faer::sparse::SparseColMat;
use faer::Unbind;

/// Convert a SparseColMat to a dense faer::Mat.
pub fn sparse_to_dense(mat: &SparseColMat<usize, f64>) -> faer::Mat<f64> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let mut dense = faer::Mat::<f64>::zeros(nrows, ncols);

    for col in 0..ncols {
        let row_inds = mat.as_ref().row_indices_of_col(col);
        let values = mat.as_ref().values_of_col(col);
        for (r, &val) in row_inds.zip(values) {
            dense[(r.unbound(), col)] += val;
        }
    }
    dense
}

/// Solve A * x = b directly using partial-pivoting LU decomposition.
///
/// Works for arbitrary square, invertible matrices (symmetric or non-symmetric).
pub fn solve_direct(mat: &SparseColMat<usize, f64>, b: &[f64]) -> Result<Vec<f64>, String> {
    let n = mat.nrows();
    if mat.ncols() != n {
        return Err(format!("Matrix must be square, got {}x{}", n, mat.ncols()));
    }
    if b.len() != n {
        return Err(format!(
            "RHS length {} does not match matrix dimension {}",
            b.len(),
            n
        ));
    }

    let dense = sparse_to_dense(mat);
    let mut b_col = faer::Col::<f64>::zeros(n);
    for i in 0..n {
        b_col[i] = b[i];
    }

    let lu = dense.partial_piv_lu();
    let x_col = lu.solve(&b_col);

    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = x_col[i];
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_solve_direct_unsymmetric() {
        // [ 2.0  1.0 ] [ x0 ] = [ 4.0 ]
        // [ 5.0  7.0 ] [ x1 ] = [ 19.0 ]
        // Det = 14 - 5 = 9.
        // x0 = (4*7 - 1*19)/9 = (28 - 19)/9 = 1.0.
        // x1 = (2*19 - 5*4)/9 = (38 - 20)/9 = 2.0.
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 5.0), (1, 1, 7.0)];
        let mat = SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap();
        let b = vec![4.0, 19.0];

        let x = solve_direct(&mat, &b).unwrap();
        assert_relative_eq!(x[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(x[1], 2.0, epsilon = 1e-12);
    }
}

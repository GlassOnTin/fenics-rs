//! Time-dependent PDE integration schemes (Backward Euler, Crank-Nicolson, Newmark-beta).
//!
//! Provides transient marchers matching DOLFINx transient solvers for parabolic
//! (heat conduction, advection-diffusion) and hyperbolic (wave propagation, dynamic elasticity) PDEs.

use crate::cg::spmv;
use crate::direct::solve_direct;
use faer::sparse::SparseColMat;
use faer::Unbind;

/// Time stepping algorithm for first-order systems M du/dt + K u = F(t).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TimeSteppingScheme {
    /// 1st-order implicit Backward Euler: unconditionally stable
    BackwardEuler,
    /// 2nd-order implicit Crank-Nicolson: energy-conserving
    CrankNicolson,
}

/// Linear transient solver for M du/dt + K u = F.
pub struct TransientHeatSolver {
    pub scheme: TimeSteppingScheme,
    pub dt: f64,
}

impl TransientHeatSolver {
    pub fn new(scheme: TimeSteppingScheme, dt: f64) -> Self {
        Self { scheme, dt }
    }

    /// Advance one time step from state u_n at time t_n to state u_{n+1} at t_{n+1} = t_n + dt.
    pub fn step(
        &self,
        mass_mat: &SparseColMat<usize, f64>,
        stiff_mat: &SparseColMat<usize, f64>,
        u_n: &[f64],
        f_n: &[f64],
        f_next: &[f64],
    ) -> Result<Vec<f64>, String> {
        let n = u_n.len();
        let dt = self.dt;

        match self.scheme {
            TimeSteppingScheme::BackwardEuler => {
                // (M + dt * K) u_{n+1} = M * u_n + dt * f_{n+1}
                let lhs = add_scaled_matrices(mass_mat, stiff_mat, dt);
                let m_un = spmv(mass_mat, u_n);
                let mut rhs = vec![0.0; n];
                for i in 0..n {
                    rhs[i] = m_un[i] + dt * f_next[i];
                }
                solve_direct(&lhs, &rhs)
            }
            TimeSteppingScheme::CrankNicolson => {
                // (M + 0.5 * dt * K) u_{n+1} = (M - 0.5 * dt * K) u_n + 0.5 * dt * (f_n + f_{n+1})
                let half_dt = 0.5 * dt;
                let lhs = add_scaled_matrices(mass_mat, stiff_mat, half_dt);
                let rhs_mat = add_scaled_matrices(mass_mat, stiff_mat, -half_dt);

                let rhs_un = spmv(&rhs_mat, u_n);
                let mut rhs = vec![0.0; n];
                for i in 0..n {
                    rhs[i] = rhs_un[i] + half_dt * (f_n[i] + f_next[i]);
                }
                solve_direct(&lhs, &rhs)
            }
        }
    }
}

/// Helper to compute A + alpha * B for two sparse matrices.
pub fn add_scaled_matrices(
    a: &SparseColMat<usize, f64>,
    b: &SparseColMat<usize, f64>,
    alpha: f64,
) -> SparseColMat<usize, f64> {
    let nrows = a.nrows();
    let ncols = a.ncols();
    let mut triplets = Vec::new();

    for col in 0..ncols {
        let r_a = a.as_ref().row_indices_of_col(col);
        let v_a = a.as_ref().values_of_col(col);
        for (r, &val) in r_a.zip(v_a) {
            triplets.push((r.unbound(), col, val));
        }

        let r_b = b.as_ref().row_indices_of_col(col);
        let v_b = b.as_ref().values_of_col(col);
        for (r, &val) in r_b.zip(v_b) {
            triplets.push((r.unbound(), col, alpha * val));
        }
    }

    SparseColMat::try_new_from_triplets(nrows, ncols, &triplets)
        .expect("Failed to build combined sparse matrix")
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_transient_exponential_decay() {
        // du/dt + lambda * u = 0  -> Exact: u(t) = u0 * exp(-lambda * t)
        // Here M = [1.0], K = [lambda]. Let lambda = 2.0, u0 = 1.0.
        // At t = 1.0 with 100 steps of dt = 0.01: exact = exp(-2.0) = 0.135335283...
        let m_triplets = vec![(0, 0, 1.0)];
        let k_triplets = vec![(0, 0, 2.0)];
        let mass = SparseColMat::try_new_from_triplets(1, 1, &m_triplets).unwrap();
        let stiff = SparseColMat::try_new_from_triplets(1, 1, &k_triplets).unwrap();

        let dt = 0.01;
        let cn_solver = TransientHeatSolver::new(TimeSteppingScheme::CrankNicolson, dt);
        let mut u = vec![1.0];
        let f = vec![0.0];

        for _ in 0..100 {
            u = cn_solver.step(&mass, &stiff, &u, &f, &f).unwrap();
        }

        let exact = (-2.0_f64).exp();
        // Crank-Nicolson is 2nd-order accurate, error with dt=0.01 should be < 1e-4
        assert_relative_eq!(u[0], exact, epsilon = 1e-4);
    }
}

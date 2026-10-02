//! Newton-Raphson nonlinear variational solver with backtracking line search.
//!
//! Solves arbitrary nonlinear systems of PDE equations F(u) = 0 using
//! tangent Jacobians J(u) = dF/du and quadratic convergence near roots.

use crate::cg::norm2;
use crate::direct::solve_direct;
use faer::sparse::SparseColMat;

/// Parameters governing Newton-Raphson iterations.
#[derive(Clone, Debug)]
pub struct NewtonSolver {
    pub max_iterations: usize,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    pub relaxation_factor: f64,
    pub backtracking: bool,
}

impl Default for NewtonSolver {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            absolute_tolerance: 1e-10,
            relative_tolerance: 1e-9,
            relaxation_factor: 1.0,
            backtracking: true,
        }
    }
}

/// Results of a Newton-Raphson solve.
#[derive(Clone, Debug)]
pub struct NewtonResult {
    pub converged: bool,
    pub iterations: usize,
    pub initial_residual: f64,
    pub final_residual: f64,
    pub solution: Vec<f64>,
}

impl NewtonSolver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Solve nonlinear problem F(u) = 0.
    ///
    /// - `u_init`: Initial guess for the state vector
    /// - `residual_fn`: Evaluates the residual vector R(u) = F(u)
    /// - `jacobian_fn`: Evaluates the tangent Jacobian matrix J(u) = dF/du
    pub fn solve<F, J>(
        &self,
        u_init: &[f64],
        mut residual_fn: F,
        mut jacobian_fn: J,
    ) -> Result<NewtonResult, String>
    where
        F: FnMut(&[f64]) -> Vec<f64>,
        J: FnMut(&[f64]) -> SparseColMat<usize, f64>,
    {
        let mut u = u_init.to_vec();
        let mut r = residual_fn(&u);
        let r0_norm = norm2(&r);
        let mut current_norm = r0_norm;

        if r0_norm <= self.absolute_tolerance {
            return Ok(NewtonResult {
                converged: true,
                iterations: 0,
                initial_residual: r0_norm,
                final_residual: r0_norm,
                solution: u,
            });
        }

        for iter in 0..self.max_iterations {
            let jac = jacobian_fn(&u);

            // Linear system: J(u) * delta_u = -R(u)
            let mut neg_r = vec![0.0; r.len()];
            for i in 0..r.len() {
                neg_r[i] = -r[i];
            }

            let delta_u = solve_direct(&jac, &neg_r)?;

            // Backtracking line search
            let mut alpha = self.relaxation_factor;
            let mut u_next = vec![0.0; u.len()];
            let mut r_next = Vec::new();
            let mut next_norm = f64::MAX;

            if self.backtracking {
                let mut search_iters = 0;
                while search_iters < 10 {
                    for i in 0..u.len() {
                        u_next[i] = u[i] + alpha * delta_u[i];
                    }
                    r_next = residual_fn(&u_next);
                    next_norm = norm2(&r_next);

                    if next_norm < current_norm || alpha <= 0.05 {
                        break;
                    }
                    alpha *= 0.5;
                    search_iters += 1;
                }
            } else {
                for i in 0..u.len() {
                    u_next[i] = u[i] + alpha * delta_u[i];
                }
                r_next = residual_fn(&u_next);
                next_norm = norm2(&r_next);
            }

            u = u_next;
            r = r_next;
            current_norm = next_norm;

            // Check convergence criteria
            if current_norm <= self.absolute_tolerance
                || (r0_norm > 0.0 && (current_norm / r0_norm) <= self.relative_tolerance)
            {
                return Ok(NewtonResult {
                    converged: true,
                    iterations: iter + 1,
                    initial_residual: r0_norm,
                    final_residual: current_norm,
                    solution: u,
                });
            }
        }

        Err(format!(
            "Newton-Raphson did not converge within {} iterations (final residual: {:.3e}, rel: {:.3e})",
            self.max_iterations,
            current_norm,
            current_norm / (r0_norm + 1e-15)
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_newton_nonlinear_system() {
        let residual = |u: &[f64]| -> Vec<f64> {
            let x = u[0];
            let y = u[1];
            vec![x * x + y * y - 4.0, (x - 1.0).exp() + y - 2.0]
        };

        let jacobian = |u: &[f64]| -> SparseColMat<usize, f64> {
            let x = u[0];
            let y = u[1];
            // df1/dx = 2x, df1/dy = 2y
            // df2/dx = e^(x-1), df2/dy = 1
            let triplets = vec![
                (0, 0, 2.0 * x),
                (0, 1, 2.0 * y),
                (1, 0, (x - 1.0).exp()),
                (1, 1, 1.0),
            ];
            SparseColMat::try_new_from_triplets(2, 2, &triplets).unwrap()
        };

        let solver = NewtonSolver::default();
        let u_init = [1.5, 1.5];
        let res = solver.solve(&u_init, residual, jacobian).unwrap();
        println!(
            "Newton finished in {} iterations, root = {:?}",
            res.iterations, res.solution
        );

        assert!(res.converged);
        assert!(res.iterations <= 10);
        let root = res.solution;
        let r_final = residual(&root);
        assert_relative_eq!(r_final[0], 0.0, epsilon = 1e-8);
        assert_relative_eq!(r_final[1], 0.0, epsilon = 1e-8);
    }
}

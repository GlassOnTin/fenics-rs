//! High-performance sparse solvers and PDE problem solutions.

pub mod cg;
pub mod elasticity_solver;
pub mod poisson_solver;

pub use cg::{dot, norm2, solve_cg, spmv};
pub use elasticity_solver::{solve_elasticity_3d, ElasticitySolution};
pub use poisson_solver::{
    compute_l2_error_2d, compute_l2_error_3d, solve_poisson_2d, solve_poisson_3d, PoissonSolution,
};

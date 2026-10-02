//! High-performance sparse solvers and PDE problem solutions.

pub mod cg;
pub mod direct;
pub mod eigen;
pub mod elasticity_solver;
pub mod heat_solver;
pub mod hyperelastic_solver;
pub mod newton;
pub mod poisson_solver;
pub mod stokes_solver;
pub mod transient;

pub use cg::{dot, norm2, solve_cg, spmv};
pub use direct::{solve_direct, sparse_to_dense};
pub use eigen::{solve_vibration_modes, VibrationMode};
pub use elasticity_solver::{solve_elasticity_3d, ElasticitySolution};
pub use heat_solver::{solve_transient_heat_2d, TransientHeatSolution};
pub use hyperelastic_solver::{solve_hyperelastic_2d, HyperelasticBC, NeoHookeanMaterial};
pub use newton::{NewtonResult, NewtonSolver};
pub use poisson_solver::{
    compute_l2_error_2d, compute_l2_error_3d, solve_poisson_2d, solve_poisson_3d, PoissonSolution,
};
pub use stokes_solver::{solve_stokes_2d, StokesBC, StokesSolution};
pub use transient::{add_scaled_matrices, TimeSteppingScheme, TransientHeatSolver};

//! Variational assembly and sparse matrix construction for finite element methods.

pub mod poisson;

pub use poisson::{
    apply_dirichlet_bc, assemble_rhs_2d, assemble_rhs_3d, assemble_stiffness_2d,
    assemble_stiffness_3d, boundary_dofs_2d, boundary_dofs_3d, DirichletBC,
};

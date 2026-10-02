//! Variational assembly and sparse matrix construction for finite element methods.

pub mod boundary;
pub mod elasticity;
pub mod mass;
pub mod poisson;

pub use boundary::{
    assemble_neumann_2d, assemble_robin_matrix_2d, assemble_surface_traction_3d,
};
pub use elasticity::{
    apply_vector_dirichlet_bc, assemble_elasticity_body_force_3d, assemble_elasticity_stiffness_3d,
    compute_element_stresses, compute_vertex_von_mises, ElasticMaterial, StressState,
    VectorDirichletBC,
};
pub use mass::{assemble_elasticity_mass_3d, assemble_mass_2d, assemble_mass_3d};
pub use poisson::{
    apply_dirichlet_bc, assemble_rhs_2d, assemble_rhs_3d, assemble_stiffness_2d,
    assemble_stiffness_3d, boundary_dofs_2d, boundary_dofs_3d, DirichletBC,
};


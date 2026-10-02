//! High-level 3D linear elasticity solver and displacement/stress recovery.

use crate::cg::solve_cg;
use fenics_assembly::elasticity::{
    apply_vector_dirichlet_bc, assemble_elasticity_body_force_3d, assemble_elasticity_stiffness_3d,
    compute_element_stresses, compute_vertex_von_mises, ElasticMaterial, StressState,
    VectorDirichletBC,
};
use fenics_mesh::TetrahedronMesh;

/// Complete solution to a 3D linear elasticity problem.
#[derive(Clone, Debug)]
pub struct ElasticitySolution {
    /// Nodal displacement components: [u_x0, u_y0, u_z0, u_x1, u_y1, u_z1, ...]
    pub displacements: Vec<f64>,
    /// Deformed vertex coordinates: original_x + u_x, original_y + u_y, original_z + u_z
    pub deformed_vertices: Vec<[f64; 3]>,
    /// Element-wise stress and strain states
    pub element_stresses: Vec<StressState>,
    /// Smoothed nodal (vertex) Von Mises stresses
    pub vertex_von_mises: Vec<f64>,
    /// Peak Von Mises stress in the structure
    pub max_von_mises: f64,
    /// Number of PCG iterations to reach convergence
    pub iterations: usize,
    /// Final residual norm
    pub residual: f64,
}

/// Solve the 3D linear elasticity Navier-Cauchy problem $-\nabla \cdot \boldsymbol{\sigma} = \mathbf{f}$.
pub fn solve_elasticity_3d(
    mesh: &TetrahedronMesh,
    material: &ElasticMaterial,
    body_force: [f64; 3],
    bc: &VectorDirichletBC,
    tol: f64,
    max_iter: usize,
) -> Result<ElasticitySolution, String> {
    let k_raw = assemble_elasticity_stiffness_3d(mesh, material);
    let f_raw = assemble_elasticity_body_force_3d(mesh, body_force);

    let (k, f) = apply_vector_dirichlet_bc(&k_raw, &f_raw, bc);

    let (displacements, iterations, residual) = solve_cg(&k, &f, None, tol, max_iter)?;

    let mut deformed_vertices = Vec::with_capacity(mesh.num_vertices());
    for (i, &vert) in mesh.vertices.iter().enumerate() {
        deformed_vertices.push([
            vert[0] + displacements[3 * i],
            vert[1] + displacements[3 * i + 1],
            vert[2] + displacements[3 * i + 2],
        ]);
    }

    let element_stresses = compute_element_stresses(mesh, &displacements, material);
    let vertex_von_mises = compute_vertex_von_mises(mesh, &element_stresses);

    let max_von_mises = vertex_von_mises
        .iter()
        .copied()
        .fold(0.0_f64, |acc, val| acc.max(val));

    Ok(ElasticitySolution {
        displacements,
        deformed_vertices,
        element_stresses,
        vertex_von_mises,
        max_von_mises,
        iterations,
        residual,
    })
}

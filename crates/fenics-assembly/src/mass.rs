//! Mass matrix assembly for scalar and vector finite element problems.

use faer::sparse::SparseColMat;
use fenics_element::jacobian::AffineSimplexMap;
use fenics_mesh::{TetrahedronMesh, TriangleMesh};
use rayon::prelude::*;

/// Assemble the 2D scalar mass matrix M_{ij} = \int_\Omega \phi_i \phi_j dx.
pub fn assemble_mass_2d(mesh: &TriangleMesh) -> SparseColMat<usize, f64> {
    let n_dofs = mesh.num_vertices();

    // For linear P1 triangle:
    // int_K phi_i phi_j dx = Area(K) / 12 for i != j
    // int_K phi_i^2 dx     = Area(K) / 6  for i == j
    let triplets: Vec<(usize, usize, f64)> = mesh
        .cells
        .par_iter()
        .flat_map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
            ];
            let map = match AffineSimplexMap::from_triangle_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            let area = 0.5 * map.det_jacobian.abs();
            let mut local_triplets = Vec::with_capacity(9);

            for i in 0..3 {
                for j in 0..3 {
                    let factor = if i == j { 1.0 / 6.0 } else { 1.0 / 12.0 };
                    local_triplets.push((cell[i], cell[j], area * factor));
                }
            }
            local_triplets
        })
        .collect();

    SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

/// Assemble the 3D scalar mass matrix M_{ij} = \int_\Omega \phi_i \phi_j dx.
pub fn assemble_mass_3d(mesh: &TetrahedronMesh) -> SparseColMat<usize, f64> {
    let n_dofs = mesh.num_vertices();

    // For linear P1 tetrahedron:
    // int_K phi_i phi_j dx = Vol(K) / 20 for i != j
    // int_K phi_i^2 dx     = Vol(K) / 10 for i == j
    let triplets: Vec<(usize, usize, f64)> = mesh
        .cells
        .par_iter()
        .flat_map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
                mesh.vertices[cell[3]],
            ];
            let map = match AffineSimplexMap::from_tetrahedron_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            let vol = (1.0 / 6.0) * map.det_jacobian.abs();
            let mut local_triplets = Vec::with_capacity(16);

            for i in 0..4 {
                for j in 0..4 {
                    let factor = if i == j { 1.0 / 10.0 } else { 1.0 / 20.0 };
                    local_triplets.push((cell[i], cell[j], vol * factor));
                }
            }
            local_triplets
        })
        .collect();

    SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

/// Assemble the 3D vector mass matrix for elasticity: M_{(i,alpha), (j,beta)} = rho * int_K phi_i phi_j delta_{alpha,beta} dx.
/// Total DOFs = 3 * mesh.num_vertices().
pub fn assemble_elasticity_mass_3d(
    mesh: &TetrahedronMesh,
    density: f64,
) -> SparseColMat<usize, f64> {
    let n_dofs = 3 * mesh.num_vertices();

    let triplets: Vec<(usize, usize, f64)> = mesh
        .cells
        .par_iter()
        .flat_map(|cell| {
            let verts = [
                mesh.vertices[cell[0]],
                mesh.vertices[cell[1]],
                mesh.vertices[cell[2]],
                mesh.vertices[cell[3]],
            ];
            let map = match AffineSimplexMap::from_tetrahedron_vertices(&verts) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            let vol = (1.0 / 6.0) * map.det_jacobian.abs();
            let mut local_triplets = Vec::with_capacity(48);

            for (i, &vi) in cell.iter().enumerate() {
                for (j, &vj) in cell.iter().enumerate() {
                    let factor = if i == j { 1.0 / 10.0 } else { 1.0 / 20.0 };
                    let m_val = density * vol * factor;

                    // Same scalar mass on each displacement component (x, y, z)
                    for alpha in 0..3 {
                        local_triplets.push((3 * vi + alpha, 3 * vj + alpha, m_val));
                    }
                }
            }
            local_triplets
        })
        .collect();

    SparseColMat::try_new_from_triplets(n_dofs, n_dofs, &triplets).unwrap()
}

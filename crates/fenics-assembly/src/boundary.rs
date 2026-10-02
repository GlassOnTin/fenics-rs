//! Exterior boundary integral assembly (ds measures) for Neumann, Robin, and surface traction BCs.

use faer::sparse::SparseColMat;
use fenics_element::quadrature::interval_quadrature;
use fenics_mesh::{BoundaryFacet, TetrahedronMesh, TriangleMesh};

/// Assemble the scalar Neumann boundary vector F_i = \int_{\Gamma_N} g(x) * v_i ds.
pub fn assemble_neumann_2d<F>(
    mesh: &TriangleMesh,
    boundary_facets: &[BoundaryFacet<2>],
    mut flux_fn: F,
) -> Vec<f64>
where
    F: FnMut(&[f64; 2]) -> f64,
{
    let n_dofs = mesh.num_vertices();
    let mut rhs = vec![0.0; n_dofs];
    let quad = interval_quadrature(3); // 2-point Gauss rule on [0, 1]

    for facet in boundary_facets {
        let v0 = mesh.vertices[facet.vertices[0]];
        let v1 = mesh.vertices[facet.vertices[1]];

        let dx = v1[0] - v0[0];
        let dy = v1[1] - v0[1];
        let length = (dx * dx + dy * dy).sqrt();

        // 1D integration along the edge parameterized by s in [0, 1]:
        // x(s) = (1 - s) * v0 + s * v1
        // ds = length * ds_ref
        for q in &quad.points {
            let s = q.point[0];
            let x_q = [
                (1.0 - s) * v0[0] + s * v1[0],
                (1.0 - s) * v0[1] + s * v1[1],
            ];
            let g_val = flux_fn(&x_q);
            let w = q.weight * length;

            // P1 shape functions on the 1D edge:
            // phi_0 = 1 - s, phi_1 = s
            rhs[facet.vertices[0]] += w * g_val * (1.0 - s);
            rhs[facet.vertices[1]] += w * g_val * s;
        }
    }

    rhs
}

/// Assemble the 3D surface traction vector F_i = \int_{\Gamma_T} t(x) . v_i ds.
pub fn assemble_surface_traction_3d<F>(
    mesh: &TetrahedronMesh,
    boundary_facets: &[BoundaryFacet<3>],
    mut traction_fn: F,
) -> Vec<f64>
where
    F: FnMut(&[f64; 3]) -> [f64; 3],
{
    let n_nodes = mesh.num_vertices();
    let mut rhs = vec![0.0; 3 * n_nodes];

    // Reference triangle quadrature for facet integration:
    // Area of triangle = 0.5 * ||(v1 - v0) x (v2 - v0)||
    for facet in boundary_facets {
        let v0 = mesh.vertices[facet.vertices[0]];
        let v1 = mesh.vertices[facet.vertices[1]];
        let v2 = mesh.vertices[facet.vertices[2]];

        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        let cross = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let area = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();

        // 3-point edge-midpoint rule on boundary facet
        let midpoints = [
            [
                0.5 * (v0[0] + v1[0]),
                0.5 * (v0[1] + v1[1]),
                0.5 * (v0[2] + v1[2]),
            ],
            [
                0.5 * (v1[0] + v2[0]),
                0.5 * (v1[1] + v2[1]),
                0.5 * (v1[2] + v2[2]),
            ],
            [
                0.5 * (v2[0] + v0[0]),
                0.5 * (v2[1] + v0[1]),
                0.5 * (v2[2] + v0[2]),
            ],
        ];

        let w = area / 3.0;
        for mid in &midpoints {
            let t = traction_fn(mid);
            // Linear element distributes equally to 3 facet vertices
            for node_idx in &facet.vertices {
                rhs[3 * node_idx] += (w / 3.0) * t[0];
                rhs[3 * node_idx + 1] += (w / 3.0) * t[1];
                rhs[3 * node_idx + 2] += (w / 3.0) * t[2];
            }
        }
    }

    rhs
}

/// Assemble Robin boundary matrix A_R = \int_{\Gamma_R} gamma * u * v ds.
pub fn assemble_robin_matrix_2d(
    mesh: &TriangleMesh,
    boundary_facets: &[BoundaryFacet<2>],
    gamma: f64,
) -> SparseColMat<usize, f64> {
    let n = mesh.num_vertices();
    let mut triplets = Vec::new();

    for facet in boundary_facets {
        let v0 = mesh.vertices[facet.vertices[0]];
        let v1 = mesh.vertices[facet.vertices[1]];

        let dx = v1[0] - v0[0];
        let dy = v1[1] - v0[1];
        let length = (dx * dx + dy * dy).sqrt();

        let n0 = facet.vertices[0];
        let n1 = facet.vertices[1];

        // 1D mass matrix on interval of length L:
        // [ L/3   L/6 ]
        // [ L/6   L/3 ]
        triplets.push((n0, n0, gamma * length / 3.0));
        triplets.push((n0, n1, gamma * length / 6.0));
        triplets.push((n1, n0, gamma * length / 6.0));
        triplets.push((n1, n1, gamma * length / 3.0));
    }

    SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use fenics_mesh::generators::unit_square;

    #[test]
    fn test_neumann_boundary_integral() {
        let mesh = unit_square(4, 4);
        let bnd_facets = mesh.extract_boundary_facets();

        // Integrate constant flux g = 1.0 around the unit square perimeter (Perimeter = 4.0)
        let rhs = assemble_neumann_2d(&mesh, &bnd_facets, |_pt| 1.0);
        let total_integral: f64 = rhs.iter().sum();

        // Total boundary integral should equal the perimeter = 4.0
        assert_relative_eq!(total_integral, 4.0, epsilon = 1e-12);
    }
}

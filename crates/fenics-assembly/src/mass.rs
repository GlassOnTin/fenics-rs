//! Mass matrix assembly for scalar and vector finite element problems.

use faer::sparse::SparseColMat;
use fenics_element::jacobian::AffineSimplexMap;
use fenics_element::{FiniteElement, P2Tetrahedron};
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

/// Assemble the 3D vector mass matrix for a quadratic (P2, 10-node) tetrahedral mesh:
/// M_{(i,alpha),(j,beta)} = rho * int phi_i phi_j delta_{alpha,beta} dV over all P2 nodes.
/// Total DOFs = 3 * vertices.len().
///
/// `vertices` are the positions of every P2 node (corners first, then edge midpoints, the
/// numbering a `P2Mesh`-style mesh uses), `cells` the per-element connectivity in
/// `P2Tetrahedron` order (4 corners, then edges 01, 12, 02, 03, 13, 23). The mesh type is a
/// plain pair because the midside nodes are not corner vertices of a `TetrahedronMesh`.
pub fn assemble_elasticity_mass_p2_3d(
    vertices: &[[f64; 3]],
    cells: &[[usize; 10]],
    density: f64,
) -> SparseColMat<usize, f64> {
    let n_dofs = 3 * vertices.len();

    // The P2 mass products are quartic polynomials; no degree-4 tetrahedral rule is needed
    // because the reference products are exact closed forms (see below), applied per element.
    let products = reference_mass_products_p2();

    let triplets: Vec<(usize, usize, f64)> = cells
        .par_iter()
        .flat_map(|cell| {
            let corners = [
                vertices[cell[0]],
                vertices[cell[1]],
                vertices[cell[2]],
                vertices[cell[3]],
            ];
            let map = match AffineSimplexMap::from_tetrahedron_vertices(&corners) {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };

            // The reference products carry the reference tetrahedron's volume (they are raw
            // cartesian integrals over the unit-barycentric simplex, volume 1/6), so the
            // physical entry scales by the jacobian magnitude |det J|, not by the volume.
            let det = map.det_jacobian.abs();
            let mut local_triplets = Vec::with_capacity(300);

            for (i, &vi) in cell.iter().enumerate() {
                for (j, &vj) in cell.iter().enumerate() {
                    let m_val = density * det * products[i * 10 + j];

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

/// Exact reference-tetrahedron products int phi_a phi_b dRefV for the P2 basis, `[f64; 100]`
/// (a * 10 + b), built once per process.
///
/// On the unit-barycentric reference tetrahedron the monomial moments are exact for every
/// degree: int lambda_0^{m0} lambda_1^{m1} lambda_2^{m2} lambda_3^{m3} dV
/// = m0! m1! m2! m3! / (m0+m1+m2+m3+3)! (the reference volume 1/6 included — the m=0 moment
/// reads 1/6). Each P2 basis function is a combination of two such monomials (unit-coordinate
/// node i: 2 lambda_i^2 - lambda_i; half-half edge node i-j: 4 lambda_i lambda_j), so the
/// products follow term by term. Node positions are read from `P2Tetrahedron` itself, so the
/// classification follows the element's node layout rather than a hard-coded copy of it.
fn reference_mass_products_p2() -> &'static [f64; 100] {
    static PRODUCTS: std::sync::OnceLock<[f64; 100]> = std::sync::OnceLock::new();
    PRODUCTS.get_or_init(build_reference_mass_products_p2)
}

fn build_reference_mass_products_p2() -> [f64; 100] {
    const FACTORIALS: [f64; 12] = [
        1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0, 3628800.0, 39916800.0,
    ];
    let nodes = P2Tetrahedron.nodal_coordinates();
    let terms: Vec<Vec<([u8; 4], f64)>> = nodes
        .iter()
        .map(|&c| {
            let lam = [1.0 - c[0] - c[1] - c[2], c[0], c[1], c[2]];
            let unit: Vec<usize> = (0..4).filter(|&i| (lam[i] - 1.0).abs() < 1e-9).collect();
            let half: Vec<usize> = (0..4).filter(|&i| (lam[i] - 0.5).abs() < 1e-9).collect();
            let mono = |i: usize, k: u8| {
                let mut m = [0u8; 4];
                m[i] = k;
                m
            };
            match (unit.len(), half.len()) {
                (1, 0) => vec![(mono(unit[0], 2), 2.0), (mono(unit[0], 1), -1.0)],
                (0, 2) => {
                    let mut m = [0u8; 4];
                    m[half[0]] = 1;
                    m[half[1]] = 1;
                    vec![(m, 4.0)]
                }
                _ => {
                    panic!("P2Tetrahedron nodes are corner or edge-midpoint barycentric positions")
                }
            }
        })
        .collect();

    let mut products = [0.0f64; 100];
    for (a, ta) in terms.iter().enumerate() {
        for (b, tb) in terms.iter().enumerate() {
            let mut acc = 0.0f64;
            for &(ma, ca) in ta.iter() {
                for &(mb, cb) in tb.iter() {
                    let m = [ma[0] + mb[0], ma[1] + mb[1], ma[2] + mb[2], ma[3] + mb[3]];
                    let deg = (m[0] + m[1] + m[2] + m[3]) as usize;
                    acc += ca
                        * cb
                        * FACTORIALS[m[0] as usize]
                        * FACTORIALS[m[1] as usize]
                        * FACTORIALS[m[2] as usize]
                        * FACTORIALS[m[3] as usize]
                        / FACTORIALS[deg + 3];
                }
            }
            products[a * 10 + b] = acc;
        }
    }
    products
}

#[cfg(test)]
mod p2_tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer::Unbind;

    /// Closed forms of the reference-tetrahedron products, from the barycentric moments
    /// int lambda^m = prod m_i! / (sum m_i + 3)! applied to the explicit basis: phi_i (unit
    /// node) = lambda_i (2 lambda_i - 1), phi_ij (half-half node) = 4 lambda_i lambda_j:
    ///   int phi_i^2          = 4/210 - 4/120 + 1/60    = 1/420
    ///   int phi_i            = 2/60 - 1/24             = -1/120   (row sum)
    ///   int phi_ij           = 4 * 1/120               = 1/30     (row sum)
    ///   int phi_ij^2         = 16 * (2!2!)/(7!)        = 4/315
    ///   int phi_i phi_j      = 4/1260 - 4/360 + 1/120  = 1/2520
    ///   int phi_i phi_ij     = 8/840 - 4/360           = -1/630
    ///   int phi_ij phi_ik    = 16 * (2!)/(7!)          = 2/315    (edges sharing a corner)
    ///   int phi_ij phi_kl    = 16 * (1!1!1!1!)/(7!)    = 1/315    (opposite edges)
    ///   sum of every product = int (sum phi)^2 = int 1 = 1/6
    #[test]
    fn p2_reference_products_match_closed_forms() {
        let products = reference_mass_products_p2();
        let at = |a: usize, b: usize| products[a * 10 + b];
        assert_relative_eq!(at(0, 0), 1.0 / 420.0, epsilon = 1e-14);
        assert_relative_eq!(at(4, 4), 4.0 / 315.0, epsilon = 1e-14);
        assert_relative_eq!(at(0, 1), 1.0 / 2520.0, epsilon = 1e-14);
        assert_relative_eq!(at(0, 4), -1.0 / 630.0, epsilon = 1e-14);
        assert_relative_eq!(at(4, 6), 2.0 / 315.0, epsilon = 1e-14);
        assert_relative_eq!(at(4, 9), 1.0 / 315.0, epsilon = 1e-14);
        for a in 0..10 {
            let row: f64 = (0..10).map(|b| at(a, b)).sum();
            let expected = if a < 4 { -1.0 / 120.0 } else { 1.0 / 30.0 };
            assert_relative_eq!(row, expected, epsilon = 1e-14);
        }
        let total: f64 = (0..100).map(|ab| products[ab]).sum();
        assert_relative_eq!(total, 1.0 / 6.0, epsilon = 1e-13);
    }

    /// One-tet mesh whose corners coincide with the reference element (identity map,
    /// |det J| = 1): with density 1 the physical mass matrix is the raw reference product
    /// (M(0,0) = 1/420), the all-ones vector sees 1^T M 1 = 3 · int (sum phi)^2 = 3 · 1/6 —
    /// each of the three displacement components contributes the constant mode.
    #[test]
    fn p2_mass_conserves_volume_on_one_tet() {
        let vertices: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        let cell: Vec<[usize; 10]> = vec![(0..10).collect::<Vec<_>>().try_into().unwrap()];
        let m = assemble_elasticity_mass_p2_3d(&vertices, &cell, 1.0);
        let ones = vec![1.0; 30];
        let col = m.as_ref();
        let mut m1 = 0.0;
        for (c, &x) in ones.iter().enumerate() {
            let rows = col.row_indices_of_col(c);
            let vals = col.values_of_col(c);
            let local: f64 = rows
                .zip(vals)
                .map(|(r, &v)| (v * x) * ones[r.unbound()])
                .sum();
            m1 += local;
        }
        assert_relative_eq!(m1, 3.0 / 6.0, epsilon = 1e-13);
        assert_relative_eq!(*m.get(0, 0).unwrap(), 1.0 / 420.0, epsilon = 1e-13);
    }
}

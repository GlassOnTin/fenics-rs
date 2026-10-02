//! Polyhedral seed geometries (Icosahedron, Octahedron, Dodecahedron) and volumetric tetrahedralization.

use fenics_mesh::TetrahedronMesh;

/// Generate surface vertices and triangular faces of a regular icosahedron.
/// Returns (vertices, triangle_face_indices).
pub fn regular_icosahedron(radius: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;

    // 12 vertices of icosahedron
    let mut raw_vertices = vec![
        [-1.0, phi, 0.0],
        [1.0, phi, 0.0],
        [-1.0, -phi, 0.0],
        [1.0, -phi, 0.0],
        [0.0, -1.0, phi],
        [0.0, 1.0, phi],
        [0.0, -1.0, -phi],
        [0.0, 1.0, -phi],
        [phi, 0.0, -1.0],
        [phi, 0.0, 1.0],
        [-phi, 0.0, -1.0],
        [-phi, 0.0, 1.0],
    ];

    // Normalize to target radius
    let current_r = (1.0 + phi * phi).sqrt();
    let scale = radius / current_r;
    for v in &mut raw_vertices {
        v[0] *= scale;
        v[1] *= scale;
        v[2] *= scale;
    }

    // 20 triangular faces oriented counter-clockwise (outward normal)
    let faces = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];

    (raw_vertices, faces)
}

/// Generate surface vertices and triangular faces of a regular octahedron.
pub fn regular_octahedron(radius: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let vertices = vec![
        [radius, 0.0, 0.0],
        [-radius, 0.0, 0.0],
        [0.0, radius, 0.0],
        [0.0, -radius, 0.0],
        [0.0, 0.0, radius],
        [0.0, 0.0, -radius],
    ];

    let faces = vec![
        [0, 2, 4],
        [2, 1, 4],
        [1, 3, 4],
        [3, 0, 4],
        [2, 0, 5],
        [1, 2, 5],
        [3, 1, 5],
        [0, 3, 5],
    ];

    (vertices, faces)
}

/// Generate surface vertices and triangulated faces of a regular dodecahedron.
/// Triangulated using 12 pentagonal face centers, resulting in 60 symmetric triangular faces.
pub fn regular_dodecahedron(radius: f64) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let inv_phi = 1.0 / phi;

    // 20 vertices
    let mut verts = vec![
        // 8 cube vertices
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        // 4 on y-z planes
        [0.0, -inv_phi, -phi],
        [0.0, -inv_phi, phi],
        [0.0, inv_phi, -phi],
        [0.0, inv_phi, phi],
        // 4 on z-x planes
        [-inv_phi, -phi, 0.0],
        [-inv_phi, phi, 0.0],
        [inv_phi, -phi, 0.0],
        [inv_phi, phi, 0.0],
        // 4 on x-y planes
        [-phi, 0.0, -inv_phi],
        [-phi, 0.0, inv_phi],
        [phi, 0.0, -inv_phi],
        [phi, 0.0, inv_phi],
    ];

    let base_r = 3.0_f64.sqrt();
    for v in &mut verts {
        v[0] *= radius / base_r;
        v[1] *= radius / base_r;
        v[2] *= radius / base_r;
    }

    // 12 pentagonal faces (defined by vertex indices)
    let pentagons = [
        [3, 11, 7, 15, 13],
        [7, 19, 5, 9, 11],
        [1, 9, 5, 14, 12],
        [3, 13, 2, 17, 1],
        [1, 17, 16, 0, 12],
        [6, 18, 4, 14, 15],
        [7, 15, 6, 18, 19],
        [5, 19, 18, 4, 14],
        [2, 13, 15, 6, 10],
        [0, 16, 2, 10, 8],
        [4, 8, 0, 12, 14],
        [8, 10, 6, 4, 0],
    ];

    let mut faces = Vec::with_capacity(60);
    // Triangulate each pentagon into 5 triangles by inserting the face centroid
    for pent in &pentagons {
        let center_idx = verts.len();
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        for &idx in pent {
            cx += verts[idx][0];
            cy += verts[idx][1];
            cz += verts[idx][2];
        }
        // Normalize centroid to sphere surface so dodecahedron is spherically symmetric
        let c_len = (cx * cx + cy * cy + cz * cz).sqrt();
        let c_scale = radius / c_len;
        verts.push([cx * c_scale, cy * c_scale, cz * c_scale]);

        for k in 0..5 {
            let next_k = (k + 1) % 5;
            faces.push([pent[k], pent[next_k], center_idx]);
        }
    }

    (verts, faces)
}

/// Convert a closed surface triangle mesh into a solid 3D volumetric TetrahedronMesh
/// by star-triangulating from the center point [0, 0, 0].
pub fn star_tetrahedralize(
    surface_vertices: &[[f64; 3]],
    surface_triangles: &[[usize; 3]],
) -> TetrahedronMesh {
    let mut vertices = surface_vertices.to_vec();
    let center_idx = vertices.len();
    vertices.push([0.0, 0.0, 0.0]); // Origin center

    let mut cells = Vec::with_capacity(surface_triangles.len());

    for tri in surface_triangles {
        let v0 = vertices[tri[0]];
        let v1 = vertices[tri[1]];
        let v2 = vertices[tri[2]];

        // Ensure positive orientation (det(J) > 0)
        // Vector from center to v0, v1, v2
        let det = v0[0] * (v1[1] * v2[2] - v1[2] * v2[1]) - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
            + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);

        if det > 0.0 {
            cells.push([center_idx, tri[0], tri[1], tri[2]]);
        } else {
            // Flip to maintain positive determinant
            cells.push([center_idx, tri[1], tri[0], tri[2]]);
        }
    }

    TetrahedronMesh::new(vertices, cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_icosahedron_properties() {
        let (verts, faces) = regular_icosahedron(1.0);
        assert_eq!(verts.len(), 12);
        assert_eq!(faces.len(), 20);

        // Every vertex must be at distance radius = 1.0 from origin
        for v in &verts {
            let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert_relative_eq!(r, 1.0, epsilon = 1e-12);
        }

        // Solid tetrahedral mesh
        let mesh = star_tetrahedralize(&verts, &faces);
        assert_eq!(mesh.num_vertices(), 13);
        assert_eq!(mesh.num_cells(), 20);

        // All tetrahedra must have positive volume
        let vol = mesh.total_volume();
        // Regular icosahedron volume of radius R=1: (5/12) * (3 + sqrt(5)) * s^3, approx 2.536
        assert!(vol > 2.0 && vol < 3.0);
        println!("Solid Icosahedron Volume: {:.4}", vol);
    }

    #[test]
    fn test_octahedron_properties() {
        let (verts, faces) = regular_octahedron(1.0);
        assert_eq!(verts.len(), 6);
        assert_eq!(faces.len(), 8);

        let mesh = star_tetrahedralize(&verts, &faces);
        assert_eq!(mesh.num_vertices(), 7);
        assert_eq!(mesh.num_cells(), 8);

        // Octahedron volume with R=1: (4/3) * R^3 = 1.33333
        assert_relative_eq!(mesh.total_volume(), 4.0 / 3.0, epsilon = 1e-12);
    }
}

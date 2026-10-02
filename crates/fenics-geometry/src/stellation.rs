//! Stellation engine for creating star polyhedra and volumetric stellated finite element meshes.

use fenics_mesh::TetrahedronMesh;

/// Stellate a polyhedron by erecting symmetric pyramids over each triangular face.
///
/// Parameters:
/// - `vertices`: base polyhedron vertices
/// - `faces`: base triangular faces
/// - `height_factor`: relative height of the star pyramids (e.g. 0.3 to 1.5).
///
/// Returns: (stellated_surface_vertices, stellated_surface_triangles).
pub fn stellate_surface(
    vertices: &[[f64; 3]],
    faces: &[[usize; 3]],
    height_factor: f64,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let mut out_verts = vertices.to_vec();
    let mut out_faces = Vec::with_capacity(faces.len() * 3);

    for face in faces {
        let v0 = vertices[face[0]];
        let v1 = vertices[face[1]];
        let v2 = vertices[face[2]];

        // Face centroid
        let c = [
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ];

        // Edge vectors
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

        // Cross product e1 x e2 gives outward normal
        let mut n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if n_len > 1e-12 {
            n[0] /= n_len;
            n[1] /= n_len;
            n[2] /= n_len;
        }

        // Average edge length
        let len01 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
        let apex_height = height_factor * len01;

        let apex = [
            c[0] + n[0] * apex_height,
            c[1] + n[1] * apex_height,
            c[2] + n[2] * apex_height,
        ];

        let apex_idx = out_verts.len();
        out_verts.push(apex);

        // 3 side faces of the pyramid
        out_faces.push([face[0], face[1], apex_idx]);
        out_faces.push([face[1], face[2], apex_idx]);
        out_faces.push([face[2], face[0], apex_idx]);
    }

    (out_verts, out_faces)
}

/// Generate a solid volumetric finite element mesh of a stellated polyhedron.
///
/// Composed of:
/// 1. The core interior tetrahedra (connecting each base face to the origin).
/// 2. The exterior stellation pyramids (each base face [v0, v1, v2] forms a solid tetrahedron with apex P).
pub fn solid_stellated_polyhedron(
    base_vertices: &[[f64; 3]],
    base_faces: &[[usize; 3]],
    height_factor: f64,
) -> TetrahedronMesh {
    let mut vertices = base_vertices.to_vec();
    let center_idx = vertices.len();
    vertices.push([0.0, 0.0, 0.0]); // Core origin

    // Cells: base cells + apex cells
    let mut cells = Vec::with_capacity(base_faces.len() * 2);

    for face in base_faces {
        let v0 = vertices[face[0]];
        let v1 = vertices[face[1]];
        let v2 = vertices[face[2]];

        // 1. Core tetrahedron: [center, v0, v1, v2]
        let det_core = v0[0] * (v1[1] * v2[2] - v1[2] * v2[1])
            - v0[1] * (v1[0] * v2[2] - v1[2] * v2[0])
            + v0[2] * (v1[0] * v2[1] - v1[1] * v2[0]);

        if det_core > 0.0 {
            cells.push([center_idx, face[0], face[1], face[2]]);
        } else {
            cells.push([center_idx, face[1], face[0], face[2]]);
        }

        // 2. Pyramid tetrahedron: [v0, v1, v2, apex]
        let c = [
            (v0[0] + v1[0] + v2[0]) / 3.0,
            (v0[1] + v1[1] + v2[1]) / 3.0,
            (v0[2] + v1[2] + v2[2]) / 3.0,
        ];
        let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let mut n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if n_len > 1e-12 {
            n[0] /= n_len;
            n[1] /= n_len;
            n[2] /= n_len;
        }

        let edge_len = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
        let apex_height = height_factor * edge_len;

        let apex = [
            c[0] + n[0] * apex_height,
            c[1] + n[1] * apex_height,
            c[2] + n[2] * apex_height,
        ];

        let apex_idx = vertices.len();
        vertices.push(apex);

        // Check orientation of [face[0], face[1], face[2], apex]
        let a = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
        let b = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
        let c_vec = [apex[0] - v0[0], apex[1] - v0[1], apex[2] - v0[2]];

        let det_pyr = a[0] * (b[1] * c_vec[2] - b[2] * c_vec[1])
            - a[1] * (b[0] * c_vec[2] - b[2] * c_vec[0])
            + a[2] * (b[0] * c_vec[1] - b[1] * c_vec[0]);

        if det_pyr > 0.0 {
            cells.push([face[0], face[1], face[2], apex_idx]);
        } else {
            cells.push([face[1], face[0], face[2], apex_idx]);
        }
    }

    TetrahedronMesh::new(vertices, cells)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polyhedra::regular_icosahedron;

    #[test]
    fn test_stellated_icosahedron() {
        let (base_verts, base_faces) = regular_icosahedron(1.0);
        let mesh = solid_stellated_polyhedron(&base_verts, &base_faces, 0.5);

        // 12 base verts + 1 center + 20 apices = 33 vertices
        assert_eq!(mesh.num_vertices(), 33);
        // 20 core tets + 20 stellation tets = 40 tets
        assert_eq!(mesh.num_cells(), 40);

        let vol = mesh.total_volume();
        println!("Solid Stellated Icosahedron Volume: {:.4}", vol);
        assert!(vol > 2.5); // Must be strictly greater than base icosahedron volume
    }
}

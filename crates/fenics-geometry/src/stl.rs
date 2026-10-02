//! Stereolithography (STL) format export for 3D printing and CAD interchange.

use fenics_mesh::TetrahedronMesh;

/// Export surface triangles as ASCII STL format string.
pub fn export_stl_ascii(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
    solid_name: &str,
) -> String {
    let mut out = String::with_capacity(triangles.len() * 256);
    out.push_str(&format!("solid {}\n", solid_name));

    for tri in triangles {
        let v0 = vertices[tri[0]];
        let v1 = vertices[tri[1]];
        let v2 = vertices[tri[2]];

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

        out.push_str(&format!(
            "  facet normal {:.6e} {:.6e} {:.6e}\n",
            n[0], n[1], n[2]
        ));
        out.push_str("    outer loop\n");
        out.push_str(&format!(
            "      vertex {:.6e} {:.6e} {:.6e}\n",
            v0[0], v0[1], v0[2]
        ));
        out.push_str(&format!(
            "      vertex {:.6e} {:.6e} {:.6e}\n",
            v1[0], v1[1], v1[2]
        ));
        out.push_str(&format!(
            "      vertex {:.6e} {:.6e} {:.6e}\n",
            v2[0], v2[1], v2[2]
        ));
        out.push_str("    endloop\n");
        out.push_str("  endfacet\n");
    }

    out.push_str(&format!("endsolid {}\n", solid_name));
    out
}

/// Export surface triangles as compact Binary STL byte buffer.
pub fn export_stl_binary(
    vertices: &[[f64; 3]],
    triangles: &[[usize; 3]],
) -> Vec<u8> {
    // 80-byte header + 4-byte triangle count + 50 bytes per triangle
    let total_bytes = 84 + triangles.len() * 50;
    let mut buf = Vec::with_capacity(total_bytes);

    // 80-byte header
    let header = b"fenics-rs binary STL generated mesh export                                     ";
    buf.extend_from_slice(&header[..80]);

    // Number of triangles (u32 little-endian)
    buf.extend_from_slice(&(triangles.len() as u32).to_le_bytes());

    for tri in triangles {
        let v0 = vertices[tri[0]];
        let v1 = vertices[tri[1]];
        let v2 = vertices[tri[2]];

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

        // Normal (3 x f32)
        buf.extend_from_slice(&(n[0] as f32).to_le_bytes());
        buf.extend_from_slice(&(n[1] as f32).to_le_bytes());
        buf.extend_from_slice(&(n[2] as f32).to_le_bytes());

        // Vertex 0 (3 x f32)
        buf.extend_from_slice(&(v0[0] as f32).to_le_bytes());
        buf.extend_from_slice(&(v0[1] as f32).to_le_bytes());
        buf.extend_from_slice(&(v0[2] as f32).to_le_bytes());

        // Vertex 1 (3 x f32)
        buf.extend_from_slice(&(v1[0] as f32).to_le_bytes());
        buf.extend_from_slice(&(v1[1] as f32).to_le_bytes());
        buf.extend_from_slice(&(v1[2] as f32).to_le_bytes());

        // Vertex 2 (3 x f32)
        buf.extend_from_slice(&(v2[0] as f32).to_le_bytes());
        buf.extend_from_slice(&(v2[1] as f32).to_le_bytes());
        buf.extend_from_slice(&(v2[2] as f32).to_le_bytes());

        // Attribute byte count (u16 = 0)
        buf.extend_from_slice(&0u16.to_le_bytes());
    }

    buf
}

/// Helper to export the exterior boundary of a solid 3D TetrahedronMesh directly to ASCII STL.
pub fn export_mesh_boundary_stl_ascii(mesh: &TetrahedronMesh, name: &str) -> String {
    let boundary_facets = mesh.extract_boundary_facets();
    let triangles: Vec<[usize; 3]> = boundary_facets.iter().map(|f| f.vertices).collect();
    export_stl_ascii(&mesh.vertices, &triangles, name)
}

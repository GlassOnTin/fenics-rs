//! Canonical benchmark mesh generators (Unit Interval, Unit Square, Unit Cube).

use crate::mesh::{IntervalMesh, SimplicialMesh, TetrahedronMesh, TriangleMesh};

/// Generate a 1D uniform mesh on [0, 1] with `n_cells` intervals.
pub fn unit_interval(n_cells: usize) -> IntervalMesh {
    assert!(n_cells > 0, "n_cells must be at least 1");
    let mut vertices = Vec::with_capacity(n_cells + 1);
    for i in 0..=n_cells {
        let x = i as f64 / n_cells as f64;
        vertices.push([x]);
    }

    let mut cells = Vec::with_capacity(n_cells);
    for i in 0..n_cells {
        cells.push([i, i + 1]);
    }

    SimplicialMesh::new(vertices, cells)
}

/// Generate a 2D uniform triangular mesh on [0, 1] x [0, 1] with `nx` x `ny` grid divisions.
/// Each rectangular grid cell is split into two triangles along the diagonal.
pub fn unit_square(nx: usize, ny: usize) -> TriangleMesh {
    assert!(nx > 0 && ny > 0, "nx and ny must be at least 1");
    let num_verts = (nx + 1) * (ny + 1);
    let mut vertices = Vec::with_capacity(num_verts);

    for j in 0..=ny {
        let y = j as f64 / ny as f64;
        for i in 0..=nx {
            let x = i as f64 / nx as f64;
            vertices.push([x, y]);
        }
    }

    let v_idx = |i: usize, j: usize| -> usize { j * (nx + 1) + i };

    let mut cells = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let v00 = v_idx(i, j);
            let v10 = v_idx(i + 1, j);
            let v01 = v_idx(i, j + 1);
            let v11 = v_idx(i + 1, j + 1);

            // Triangle 1: (v00, v10, v11)
            cells.push([v00, v10, v11]);
            // Triangle 2: (v00, v11, v01)
            cells.push([v00, v11, v01]);
        }
    }

    SimplicialMesh::new(vertices, cells)
}

/// Generate a 3D uniform tetrahedral mesh on [0, L] x [0, W] x [0, H] with `nx` x `ny` x `nz` grid divisions.
pub fn box_beam(
    length: f64,
    width: f64,
    height: f64,
    nx: usize,
    ny: usize,
    nz: usize,
) -> TetrahedronMesh {
    assert!(
        nx > 0 && ny > 0 && nz > 0,
        "nx, ny, and nz must be at least 1"
    );
    let num_verts = (nx + 1) * (ny + 1) * (nz + 1);
    let mut vertices = Vec::with_capacity(num_verts);

    for k in 0..=nz {
        let z = (k as f64 / nz as f64) * height;
        for j in 0..=ny {
            let y = (j as f64 / ny as f64) * width;
            for i in 0..=nx {
                let x = (i as f64 / nx as f64) * length;
                vertices.push([x, y, z]);
            }
        }
    }

    let v_idx =
        |i: usize, j: usize, k: usize| -> usize { k * (nx + 1) * (ny + 1) + j * (nx + 1) + i };

    let mut cells = Vec::with_capacity(6 * nx * ny * nz);
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p0 = v_idx(i, j, k);
                let p1 = v_idx(i + 1, j, k);
                let p2 = v_idx(i, j + 1, k);
                let p3 = v_idx(i + 1, j + 1, k);
                let p4 = v_idx(i, j, k + 1);
                let p5 = v_idx(i + 1, j, k + 1);
                let p6 = v_idx(i, j + 1, k + 1);
                let p7 = v_idx(i + 1, j + 1, k + 1);

                cells.push([p0, p1, p3, p7]);
                cells.push([p0, p3, p2, p7]);
                cells.push([p0, p2, p6, p7]);
                cells.push([p0, p6, p4, p7]);
                cells.push([p0, p4, p5, p7]);
                cells.push([p0, p5, p1, p7]);
            }
        }
    }

    SimplicialMesh::new(vertices, cells)
}

/// Generate a 3D uniform tetrahedral mesh on [0, 1]^3 with `nx` x `ny` x `nz` grid divisions.
/// Each cube cell is decomposed into 6 tetrahedra using the standard Kuhn triangulation.
pub fn unit_cube(nx: usize, ny: usize, nz: usize) -> TetrahedronMesh {
    box_beam(1.0, 1.0, 1.0, nx, ny, nz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_unit_interval_properties() {
        let mesh = unit_interval(10);
        assert_eq!(mesh.num_vertices(), 11);
        assert_eq!(mesh.num_cells(), 10);
        let bnd = mesh.extract_boundary_facets();
        assert_eq!(bnd.len(), 2);
    }

    #[test]
    fn test_unit_square_properties() {
        let mesh = unit_square(4, 4);
        assert_eq!(mesh.num_vertices(), 25);
        assert_eq!(mesh.num_cells(), 32);

        assert_relative_eq!(mesh.total_volume(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(mesh.boundary_measure(), 4.0, epsilon = 1e-12);

        let bnd = mesh.extract_boundary_facets();
        assert_eq!(bnd.len(), 16);
    }

    #[test]
    fn test_unit_cube_properties() {
        let mesh = unit_cube(3, 3, 3);
        assert_eq!(mesh.num_vertices(), 64);
        assert_eq!(mesh.num_cells(), 162);

        assert_relative_eq!(mesh.total_volume(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(mesh.boundary_measure(), 6.0, epsilon = 1e-12);

        let bnd = mesh.extract_boundary_facets();
        assert_eq!(bnd.len(), 108);
    }
}

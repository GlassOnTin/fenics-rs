//! Generic simplicial mesh data structure and topological entity extraction.

use std::collections::HashMap;

/// A generic simplicial mesh in DIM-dimensional space with NODES vertices per cell.
///
/// For 1D Interval: `DIM = 1, NODES = 2`
/// For 2D Triangle: `DIM = 2, NODES = 3`
/// For 3D Tetrahedron: `DIM = 3, NODES = 4`
#[derive(Clone, Debug)]
pub struct SimplicialMesh<const DIM: usize, const NODES: usize> {
    /// Vertex spatial coordinates: [x_0, x_1, ..., x_{DIM-1}]
    pub vertices: Vec<[f64; DIM]>,
    /// Cell connectivity: indices into the vertices array
    pub cells: Vec<[usize; NODES]>,
}

pub type IntervalMesh = SimplicialMesh<1, 2>;
pub type TriangleMesh = SimplicialMesh<2, 3>;
pub type TetrahedronMesh = SimplicialMesh<3, 4>;

/// A boundary facet (edge in 2D, triangle in 3D).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoundaryFacet<const FACET_NODES: usize> {
    /// Sorted vertex indices of the facet
    pub vertices: [usize; FACET_NODES],
    /// Index of the cell to which this facet belongs
    pub cell_idx: usize,
    /// Local facet index within the cell
    pub local_facet_idx: usize,
}

impl<const DIM: usize, const NODES: usize> SimplicialMesh<DIM, NODES> {
    pub fn new(vertices: Vec<[f64; DIM]>, cells: Vec<[usize; NODES]>) -> Self {
        Self { vertices, cells }
    }

    pub fn num_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn num_cells(&self) -> usize {
        self.cells.len()
    }
}

impl SimplicialMesh<1, 2> {
    /// Extract endpoints of the 1D mesh.
    pub fn extract_boundary_facets(&self) -> Vec<BoundaryFacet<1>> {
        let mut vertex_counts = HashMap::new();
        for (cell_idx, cell) in self.cells.iter().enumerate() {
            for (local_idx, &v) in cell.iter().enumerate() {
                vertex_counts
                    .entry(v)
                    .and_modify(|(count, _, _)| *count += 1)
                    .or_insert((1, cell_idx, local_idx));
            }
        }

        vertex_counts
            .into_iter()
            .filter_map(|(v, (count, cell_idx, local_facet_idx))| {
                if count == 1 {
                    Some(BoundaryFacet {
                        vertices: [v],
                        cell_idx,
                        local_facet_idx,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}

impl SimplicialMesh<2, 3> {
    /// Extract all exterior boundary edges.
    pub fn extract_boundary_facets(&self) -> Vec<BoundaryFacet<2>> {
        let mut edge_counts = HashMap::new();

        for (cell_idx, cell) in self.cells.iter().enumerate() {
            // Triangle has 3 edges: (1,2), (2,0), (0,1)
            let edges = [
                (0, [cell[1], cell[2]]),
                (1, [cell[2], cell[0]]),
                (2, [cell[0], cell[1]]),
            ];

            for (local_idx, mut edge) in edges {
                edge.sort_unstable();
                edge_counts
                    .entry(edge)
                    .and_modify(|(count, _, _)| *count += 1)
                    .or_insert((1, cell_idx, local_idx));
            }
        }

        edge_counts
            .into_iter()
            .filter_map(|(vertices, (count, cell_idx, local_facet_idx))| {
                if count == 1 {
                    Some(BoundaryFacet {
                        vertices,
                        cell_idx,
                        local_facet_idx,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute total domain area (sum of triangle areas).
    pub fn total_volume(&self) -> f64 {
        self.cells
            .iter()
            .map(|cell| {
                let v0 = self.vertices[cell[0]];
                let v1 = self.vertices[cell[1]];
                let v2 = self.vertices[cell[2]];
                0.5 * ((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])).abs()
            })
            .sum()
    }

    /// Compute total perimeter length of the boundary edges.
    pub fn boundary_measure(&self) -> f64 {
        let bnd = self.extract_boundary_facets();
        bnd.iter()
            .map(|f| {
                let p0 = self.vertices[f.vertices[0]];
                let p1 = self.vertices[f.vertices[1]];
                let dx = p1[0] - p0[0];
                let dy = p1[1] - p0[1];
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }
}

impl SimplicialMesh<3, 4> {
    /// Extract all exterior boundary triangles.
    pub fn extract_boundary_facets(&self) -> Vec<BoundaryFacet<3>> {
        let mut face_counts = HashMap::new();

        for (cell_idx, cell) in self.cells.iter().enumerate() {
            // Tetrahedron has 4 triangular faces:
            let faces = [
                (0, [cell[1], cell[2], cell[3]]),
                (1, [cell[0], cell[2], cell[3]]),
                (2, [cell[0], cell[1], cell[3]]),
                (3, [cell[0], cell[1], cell[2]]),
            ];

            for (local_idx, mut face) in faces {
                face.sort_unstable();
                face_counts
                    .entry(face)
                    .and_modify(|(count, _, _)| *count += 1)
                    .or_insert((1, cell_idx, local_idx));
            }
        }

        face_counts
            .into_iter()
            .filter_map(|(vertices, (count, cell_idx, local_facet_idx))| {
                if count == 1 {
                    Some(BoundaryFacet {
                        vertices,
                        cell_idx,
                        local_facet_idx,
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute total domain volume (sum of tetrahedron volumes).
    pub fn total_volume(&self) -> f64 {
        self.cells
            .iter()
            .map(|cell| {
                let v0 = self.vertices[cell[0]];
                let v1 = self.vertices[cell[1]];
                let v2 = self.vertices[cell[2]];
                let v3 = self.vertices[cell[3]];

                let a = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
                let b = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
                let c = [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]];

                let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                (det.abs()) / 6.0
            })
            .sum()
    }

    /// Compute total surface area of boundary faces.
    pub fn boundary_measure(&self) -> f64 {
        let bnd = self.extract_boundary_facets();
        bnd.iter()
            .map(|f| {
                let p0 = self.vertices[f.vertices[0]];
                let p1 = self.vertices[f.vertices[1]];
                let p2 = self.vertices[f.vertices[2]];

                let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
                let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];

                let cross = [
                    v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0],
                ];
                0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
            })
            .sum()
    }
}

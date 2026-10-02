//! Simplicial mesh representations, topological entity queries, and canonical mesh generators.

pub mod generators;
pub mod mesh;
pub mod tags;

pub use generators::{box_beam, unit_cube, unit_interval, unit_square};
pub use mesh::{BoundaryFacet, IntervalMesh, SimplicialMesh, TetrahedronMesh, TriangleMesh};
pub use tags::{mark_boundary_facets, MeshTags};


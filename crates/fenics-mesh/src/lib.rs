//! Simplicial mesh representations, topological entity queries, and canonical mesh generators.

pub mod generators;
pub mod mesh;

pub use generators::{unit_cube, unit_interval, unit_square};
pub use mesh::{BoundaryFacet, IntervalMesh, SimplicialMesh, TetrahedronMesh, TriangleMesh};

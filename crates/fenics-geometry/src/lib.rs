//! Polyhedral seed geometries, stellation engine, and STL export.

pub mod polyhedra;
pub mod stellation;
pub mod stl;

pub use polyhedra::{
    regular_dodecahedron, regular_icosahedron, regular_octahedron, star_tetrahedralize,
};
pub use stellation::{solid_stellated_polyhedron, stellate_surface};
pub use stl::{export_mesh_boundary_stl_ascii, export_stl_ascii, export_stl_binary};

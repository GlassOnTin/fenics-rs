//! Reference finite elements, Gaussian quadrature rules, and coordinate Jacobians.

pub mod dg;
pub mod jacobian;
pub mod lagrange;
pub mod nedelec;
pub mod quadrature;
pub mod raviart_thomas;
pub mod vector_element;

pub use dg::{DG0Interval, DG0Tetrahedron, DG0Triangle, DG1Tetrahedron, DG1Triangle};
pub use jacobian::AffineSimplexMap;
pub use lagrange::{
    FiniteElement, P1Interval, P1Tetrahedron, P1Triangle, P2Interval, P2Tetrahedron, P2Triangle,
    P3Interval, P3Triangle,
};
pub use nedelec::Nedelec1Triangle;
pub use quadrature::{
    interval_quadrature, tetrahedron_quadrature, triangle_quadrature, QuadraturePoint,
    QuadratureRule,
};
pub use raviart_thomas::RT1Triangle;
pub use vector_element::{VectorElement, VectorFiniteElement};

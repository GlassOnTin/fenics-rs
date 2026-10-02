//! Reference finite elements, Gaussian quadrature rules, and coordinate Jacobians.

pub mod jacobian;
pub mod lagrange;
pub mod quadrature;

pub use jacobian::AffineSimplexMap;
pub use lagrange::{FiniteElement, P1Interval, P1Tetrahedron, P1Triangle, P2Triangle};
pub use quadrature::{
    interval_quadrature, tetrahedron_quadrature, triangle_quadrature, QuadraturePoint,
    QuadratureRule,
};

//! Discontinuous Galerkin (DG) finite elements.
//!
//! DG elements do not enforce inter-element continuity across facet boundaries.
//! They are essential for advection-dominated flows, conservation laws,
//! flux formulations, and mixed finite element methods.

use crate::lagrange::FiniteElement;

/// Piecewise constant DG element on the 1D interval (DG0).
#[derive(Clone, Copy, Debug, Default)]
pub struct DG0Interval;

impl FiniteElement<1> for DG0Interval {
    fn num_dofs(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        0
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 1]> {
        vec![[0.5]]
    }

    fn evaluate_basis(&self, _xi: &[f64; 1]) -> Vec<f64> {
        vec![1.0]
    }

    fn evaluate_gradients(&self, _xi: &[f64; 1]) -> Vec<[f64; 1]> {
        vec![[0.0]]
    }
}

/// Piecewise constant DG element on the 2D triangle (DG0).
#[derive(Clone, Copy, Debug, Default)]
pub struct DG0Triangle;

impl FiniteElement<2> for DG0Triangle {
    fn num_dofs(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        0
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 2]> {
        vec![[1.0 / 3.0, 1.0 / 3.0]]
    }

    fn evaluate_basis(&self, _xi: &[f64; 2]) -> Vec<f64> {
        vec![1.0]
    }

    fn evaluate_gradients(&self, _xi: &[f64; 2]) -> Vec<[f64; 2]> {
        vec![[0.0, 0.0]]
    }
}

/// Piecewise constant DG element on the 3D tetrahedron (DG0).
#[derive(Clone, Copy, Debug, Default)]
pub struct DG0Tetrahedron;

impl FiniteElement<3> for DG0Tetrahedron {
    fn num_dofs(&self) -> usize {
        1
    }

    fn degree(&self) -> usize {
        0
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 3]> {
        vec![[0.25, 0.25, 0.25]]
    }

    fn evaluate_basis(&self, _xi: &[f64; 3]) -> Vec<f64> {
        vec![1.0]
    }

    fn evaluate_gradients(&self, _xi: &[f64; 3]) -> Vec<[f64; 3]> {
        vec![[0.0, 0.0, 0.0]]
    }
}

/// Piecewise linear DG element on the 2D triangle (DG1).
#[derive(Clone, Copy, Debug, Default)]
pub struct DG1Triangle;

impl FiniteElement<2> for DG1Triangle {
    fn num_dofs(&self) -> usize {
        3
    }

    fn degree(&self) -> usize {
        1
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    }

    fn evaluate_basis(&self, xi: &[f64; 2]) -> Vec<f64> {
        let x = xi[0];
        let y = xi[1];
        vec![1.0 - x - y, x, y]
    }

    fn evaluate_gradients(&self, _xi: &[f64; 2]) -> Vec<[f64; 2]> {
        vec![[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
    }
}

/// Piecewise linear DG element on the 3D tetrahedron (DG1).
#[derive(Clone, Copy, Debug, Default)]
pub struct DG1Tetrahedron;

impl FiniteElement<3> for DG1Tetrahedron {
    fn num_dofs(&self) -> usize {
        4
    }

    fn degree(&self) -> usize {
        1
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    fn evaluate_basis(&self, xi: &[f64; 3]) -> Vec<f64> {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        vec![1.0 - x - y - z, x, y, z]
    }

    fn evaluate_gradients(&self, _xi: &[f64; 3]) -> Vec<[f64; 3]> {
        vec![
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_dg0_elements() {
        let dg0_1d = DG0Interval;
        assert_eq!(dg0_1d.evaluate_basis(&[0.3]), vec![1.0]);
        assert_eq!(dg0_1d.evaluate_gradients(&[0.3]), vec![[0.0]]);

        let dg0_2d = DG0Triangle;
        assert_eq!(dg0_2d.evaluate_basis(&[0.2, 0.3]), vec![1.0]);
        assert_eq!(dg0_2d.evaluate_gradients(&[0.2, 0.3]), vec![[0.0, 0.0]]);

        let dg0_3d = DG0Tetrahedron;
        assert_eq!(dg0_3d.evaluate_basis(&[0.1, 0.2, 0.3]), vec![1.0]);
        assert_eq!(
            dg0_3d.evaluate_gradients(&[0.1, 0.2, 0.3]),
            vec![[0.0, 0.0, 0.0]]
        );
    }

    #[test]
    fn test_dg1_partition_of_unity() {
        let dg1_tri = DG1Triangle;
        let pt = [0.25, 0.45];
        let vals = dg1_tri.evaluate_basis(&pt);
        assert_relative_eq!(vals.iter().sum::<f64>(), 1.0);
    }
}

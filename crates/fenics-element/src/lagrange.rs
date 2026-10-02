//! Lagrange finite element basis functions on reference simplices.

/// Evaluation of Lagrange basis functions and their gradients on a reference simplex.
pub trait FiniteElement<const DIM: usize> {
    /// Number of degrees of freedom (nodes) per element.
    fn num_dofs(&self) -> usize;

    /// Polynomial degree of the element (e.g. 1 for P1, 2 for P2).
    fn degree(&self) -> usize;

    /// Local coordinates of the nodal degrees of freedom on the reference simplex.
    fn nodal_coordinates(&self) -> Vec<[f64; DIM]>;

    /// Evaluate all basis functions at a point xi in the reference simplex.
    fn evaluate_basis(&self, xi: &[f64; DIM]) -> Vec<f64>;

    /// Evaluate gradients of all basis functions with respect to reference coordinates xi.
    /// Returns a vector of gradients, each gradient being [f64; DIM].
    fn evaluate_gradients(&self, xi: &[f64; DIM]) -> Vec<[f64; DIM]>;
}

/// 1D Linear Lagrange Element (P1 on [0, 1]).
#[derive(Clone, Copy, Debug, Default)]
pub struct P1Interval;

impl FiniteElement<1> for P1Interval {
    fn num_dofs(&self) -> usize {
        2
    }

    fn degree(&self) -> usize {
        1
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 1]> {
        vec![[0.0], [1.0]]
    }

    fn evaluate_basis(&self, xi: &[f64; 1]) -> Vec<f64> {
        let x = xi[0];
        vec![1.0 - x, x]
    }

    fn evaluate_gradients(&self, _xi: &[f64; 1]) -> Vec<[f64; 1]> {
        vec![[-1.0], [1.0]]
    }
}

/// 2D Linear Lagrange Element (P1 on Triangle: (0,0), (1,0), (0,1)).
#[derive(Clone, Copy, Debug, Default)]
pub struct P1Triangle;

impl FiniteElement<2> for P1Triangle {
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

/// 3D Linear Lagrange Element (P1 on Tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)).
#[derive(Clone, Copy, Debug, Default)]
pub struct P1Tetrahedron;

impl FiniteElement<3> for P1Tetrahedron {
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

/// 2D Quadratic Lagrange Element (P2 on Triangle: 3 vertices + 3 edge midpoints = 6 DOFs).
#[derive(Clone, Copy, Debug, Default)]
pub struct P2Triangle;

impl FiniteElement<2> for P2Triangle {
    fn num_dofs(&self) -> usize {
        6
    }

    fn degree(&self) -> usize {
        2
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 2]> {
        vec![
            [0.0, 0.0], // Node 0: Vertex 0
            [1.0, 0.0], // Node 1: Vertex 1
            [0.0, 1.0], // Node 2: Vertex 2
            [0.5, 0.0], // Node 3: Midpoint 0-1
            [0.5, 0.5], // Node 4: Midpoint 1-2
            [0.0, 0.5], // Node 5: Midpoint 2-0
        ]
    }

    fn evaluate_basis(&self, xi: &[f64; 2]) -> Vec<f64> {
        let x = xi[0];
        let y = xi[1];
        let l0 = 1.0 - x - y;
        let l1 = x;
        let l2 = y;

        vec![
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l2 * l0,
        ]
    }

    fn evaluate_gradients(&self, xi: &[f64; 2]) -> Vec<[f64; 2]> {
        let x = xi[0];
        let y = xi[1];
        let l0 = 1.0 - x - y;
        let l1 = x;
        let l2 = y;

        // dl0/dx = -1, dl0/dy = -1
        // dl1/dx = 1,  dl1/dy = 0
        // dl2/dx = 0,  dl2/dy = 1

        // phi0 = l0*(2*l0 - 1) -> dphi0 = (4*l0 - 1) * dl0
        let g0 = [-(4.0 * l0 - 1.0), -(4.0 * l0 - 1.0)];
        // phi1 = l1*(2*l1 - 1) -> dphi1 = (4*l1 - 1) * dl1
        let g1 = [4.0 * l1 - 1.0, 0.0];
        // phi2 = l2*(2*l2 - 1) -> dphi2 = (4*l2 - 1) * dl2
        let g2 = [0.0, 4.0 * l2 - 1.0];
        // phi3 = 4*l0*l1 -> dphi3/dx = 4*(dl0/dx*l1 + l0*dl1/dx) = 4*(-l1 + l0)
        let g3 = [4.0 * (l0 - l1), -4.0 * l1];
        // phi4 = 4*l1*l2 -> dphi4/dx = 4*l2, dphi4/dy = 4*l1
        let g4 = [4.0 * l2, 4.0 * l1];
        // phi5 = 4*l2*l0 -> dphi5/dx = -4*l2, dphi5/dy = 4*(l0 - l2)
        let g5 = [-4.0 * l2, 4.0 * (l0 - l2)];

        vec![g0, g1, g2, g3, g4, g5]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_element_properties<const DIM: usize, E: FiniteElement<DIM>>(elem: &E, test_points: &[[f64; DIM]]) {
        let nodes = elem.nodal_coordinates();
        assert_eq!(nodes.len(), elem.num_dofs());

        // 1. Kronecker Delta property: phi_i(node_j) == delta_ij
        for (i, node_i) in nodes.iter().enumerate() {
            let values = elem.evaluate_basis(node_i);
            for (j, &val) in values.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(val, expected, epsilon = 1e-12);
            }
        }

        // 2. Partition of unity: sum(phi_i(xi)) == 1.0 and sum(grad phi_i) == 0.0 at interior points
        for pt in test_points {
            let vals = elem.evaluate_basis(pt);
            let sum_val: f64 = vals.iter().sum();
            assert_relative_eq!(sum_val, 1.0, epsilon = 1e-12);

            let grads = elem.evaluate_gradients(pt);
            let mut sum_grad = [0.0; DIM];
            for g in &grads {
                for d in 0..DIM {
                    sum_grad[d] += g[d];
                }
            }
            for d in 0..DIM {
                assert_relative_eq!(sum_grad[d], 0.0, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_p1_interval() {
        test_element_properties(&P1Interval, &[[0.25], [0.5], [0.75]]);
    }

    #[test]
    fn test_p1_triangle() {
        test_element_properties(&P1Triangle, &[[0.2, 0.2], [0.333, 0.333], [0.1, 0.7]]);
    }

    #[test]
    fn test_p1_tetrahedron() {
        test_element_properties(
            &P1Tetrahedron,
            &[[0.25, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.1, 0.1]],
        );
    }

    #[test]
    fn test_p2_triangle() {
        test_element_properties(&P2Triangle, &[[0.2, 0.2], [0.333, 0.333], [0.1, 0.7]]);
    }
}

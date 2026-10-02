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

/// 1D Quadratic Lagrange Element (P2 on [0, 1]: 2 vertices + 1 midpoint = 3 DOFs).
#[derive(Clone, Copy, Debug, Default)]
pub struct P2Interval;

impl FiniteElement<1> for P2Interval {
    fn num_dofs(&self) -> usize {
        3
    }

    fn degree(&self) -> usize {
        2
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 1]> {
        vec![[0.0], [1.0], [0.5]]
    }

    fn evaluate_basis(&self, xi: &[f64; 1]) -> Vec<f64> {
        let x = xi[0];
        let l0 = 1.0 - x;
        let l1 = x;
        vec![l0 * (2.0 * l0 - 1.0), l1 * (2.0 * l1 - 1.0), 4.0 * l0 * l1]
    }

    fn evaluate_gradients(&self, xi: &[f64; 1]) -> Vec<[f64; 1]> {
        let x = xi[0];
        vec![[4.0 * x - 3.0], [4.0 * x - 1.0], [4.0 - 8.0 * x]]
    }
}

/// 1D Cubic Lagrange Element (P3 on [0, 1]: 4 DOFs at 0, 1, 1/3, 2/3).
#[derive(Clone, Copy, Debug, Default)]
pub struct P3Interval;

impl FiniteElement<1> for P3Interval {
    fn num_dofs(&self) -> usize {
        4
    }

    fn degree(&self) -> usize {
        3
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 1]> {
        vec![[0.0], [1.0], [1.0 / 3.0], [2.0 / 3.0]]
    }

    fn evaluate_basis(&self, xi: &[f64; 1]) -> Vec<f64> {
        let x = xi[0];
        let l0 = 1.0 - x;
        let l1 = x;
        vec![
            0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0),
            0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0),
            4.5 * l0 * l1 * (3.0 * l0 - 1.0),
            4.5 * l0 * l1 * (3.0 * l1 - 1.0),
        ]
    }

    fn evaluate_gradients(&self, xi: &[f64; 1]) -> Vec<[f64; 1]> {
        let x = xi[0];
        // phi0(x) = -0.5 * (1-x) * (2 - 3x) * (1 - 3x) = -0.5 * (1 - x) * (2 - 9x + 9x^2)
        //         = -0.5 * (2 - 11x + 18x^2 - 9x^3) = -1 + 5.5x - 9x^2 + 4.5x^3
        // phi0'(x) = 5.5 - 18x + 13.5x^2 = (-11 + 36x - 27x^2) / -2 ? Let's check:
        // d/dx [ 0.5 * (1-x) * (3(1-x)-1) * (3(1-x)-2) ]:
        // Let u = 1 - x, du/dx = -1.
        // g(u) = 0.5 * u * (3u - 1) * (3u - 2) = 0.5 * (9u^3 - 9u^2 + 2u)
        // g'(u) = 0.5 * (27u^2 - 18u + 2)
        // dphi0/dx = -g'(1 - x) = -0.5 * (27(1-x)^2 - 18(1-x) + 2)
        let u0 = 1.0 - x;
        let dphi0 = -0.5 * (27.0 * u0 * u0 - 18.0 * u0 + 2.0);

        // phi1(x): let u = x, du/dx = 1.
        // dphi1/dx = 0.5 * (27.0 * x * x - 18.0 * x + 2.0);
        let dphi1 = 0.5 * (27.0 * x * x - 18.0 * x + 2.0);

        // phi2(x) = 4.5 * (1-x) * x * (2 - 3x) = 4.5 * (2x - 5x^2 + 3x^3)
        // phi2'(x) = 4.5 * (2 - 10x + 9x^2)
        let dphi2 = 4.5 * (2.0 - 10.0 * x + 9.0 * x * x);

        // phi3(x) = 4.5 * (1-x) * x * (3x - 1) = 4.5 * (-x + 4x^2 - 3x^3)
        // phi3'(x) = 4.5 * (-1 + 8x - 9x^2)
        let dphi3 = 4.5 * (-1.0 + 8.0 * x - 9.0 * x * x);

        vec![[dphi0], [dphi1], [dphi2], [dphi3]]
    }
}

/// 3D Quadratic Lagrange Element (P2 on Tetrahedron: 4 vertices + 6 edge midpoints = 10 DOFs).
#[derive(Clone, Copy, Debug, Default)]
pub struct P2Tetrahedron;

impl FiniteElement<3> for P2Tetrahedron {
    fn num_dofs(&self) -> usize {
        10
    }

    fn degree(&self) -> usize {
        2
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0], // Node 0: Vertex 0
            [1.0, 0.0, 0.0], // Node 1: Vertex 1
            [0.0, 1.0, 0.0], // Node 2: Vertex 2
            [0.0, 0.0, 1.0], // Node 3: Vertex 3
            [0.5, 0.0, 0.0], // Node 4: Midpoint 0-1
            [0.5, 0.5, 0.0], // Node 5: Midpoint 1-2
            [0.0, 0.5, 0.0], // Node 6: Midpoint 0-2
            [0.0, 0.0, 0.5], // Node 7: Midpoint 0-3
            [0.5, 0.0, 0.5], // Node 8: Midpoint 1-3
            [0.0, 0.5, 0.5], // Node 9: Midpoint 2-3
        ]
    }

    fn evaluate_basis(&self, xi: &[f64; 3]) -> Vec<f64> {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let l0 = 1.0 - x - y - z;
        let l1 = x;
        let l2 = y;
        let l3 = z;

        vec![
            l0 * (2.0 * l0 - 1.0),
            l1 * (2.0 * l1 - 1.0),
            l2 * (2.0 * l2 - 1.0),
            l3 * (2.0 * l3 - 1.0),
            4.0 * l0 * l1,
            4.0 * l1 * l2,
            4.0 * l0 * l2,
            4.0 * l0 * l3,
            4.0 * l1 * l3,
            4.0 * l2 * l3,
        ]
    }

    fn evaluate_gradients(&self, xi: &[f64; 3]) -> Vec<[f64; 3]> {
        let x = xi[0];
        let y = xi[1];
        let z = xi[2];
        let l0 = 1.0 - x - y - z;
        let l1 = x;
        let l2 = y;
        let l3 = z;

        let g0 = [-(4.0 * l0 - 1.0), -(4.0 * l0 - 1.0), -(4.0 * l0 - 1.0)];
        let g1 = [4.0 * l1 - 1.0, 0.0, 0.0];
        let g2 = [0.0, 4.0 * l2 - 1.0, 0.0];
        let g3 = [0.0, 0.0, 4.0 * l3 - 1.0];

        let g4 = [4.0 * (l0 - l1), -4.0 * l1, -4.0 * l1];
        let g5 = [4.0 * l2, 4.0 * l1, 0.0];
        let g6 = [-4.0 * l2, 4.0 * (l0 - l2), -4.0 * l2];
        let g7 = [-4.0 * l3, -4.0 * l3, 4.0 * (l0 - l3)];
        let g8 = [4.0 * l3, 0.0, 4.0 * l1];
        let g9 = [0.0, 4.0 * l3, 4.0 * l2];

        vec![g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
    }
}

/// 2D Cubic Lagrange Element (P3 on Triangle: 3 vertices + 6 edge nodes + 1 centroid = 10 DOFs).
#[derive(Clone, Copy, Debug, Default)]
pub struct P3Triangle;

impl FiniteElement<2> for P3Triangle {
    fn num_dofs(&self) -> usize {
        10
    }

    fn degree(&self) -> usize {
        3
    }

    fn nodal_coordinates(&self) -> Vec<[f64; 2]> {
        vec![
            [0.0, 0.0],             // Node 0: Vertex 0
            [1.0, 0.0],             // Node 1: Vertex 1
            [0.0, 1.0],             // Node 2: Vertex 2
            [1.0 / 3.0, 0.0],       // Node 3: Edge 0-1 near 0
            [2.0 / 3.0, 0.0],       // Node 4: Edge 0-1 near 1
            [2.0 / 3.0, 1.0 / 3.0], // Node 5: Edge 1-2 near 1
            [1.0 / 3.0, 2.0 / 3.0], // Node 6: Edge 1-2 near 2
            [0.0, 2.0 / 3.0],       // Node 7: Edge 2-0 near 2
            [0.0, 1.0 / 3.0],       // Node 8: Edge 2-0 near 0
            [1.0 / 3.0, 1.0 / 3.0], // Node 9: Face centroid
        ]
    }

    fn evaluate_basis(&self, xi: &[f64; 2]) -> Vec<f64> {
        let x = xi[0];
        let y = xi[1];
        let l0 = 1.0 - x - y;
        let l1 = x;
        let l2 = y;

        vec![
            0.5 * l0 * (3.0 * l0 - 1.0) * (3.0 * l0 - 2.0),
            0.5 * l1 * (3.0 * l1 - 1.0) * (3.0 * l1 - 2.0),
            0.5 * l2 * (3.0 * l2 - 1.0) * (3.0 * l2 - 2.0),
            4.5 * l0 * l1 * (3.0 * l0 - 1.0),
            4.5 * l0 * l1 * (3.0 * l1 - 1.0),
            4.5 * l1 * l2 * (3.0 * l1 - 1.0),
            4.5 * l1 * l2 * (3.0 * l2 - 1.0),
            4.5 * l2 * l0 * (3.0 * l2 - 1.0),
            4.5 * l2 * l0 * (3.0 * l0 - 1.0),
            27.0 * l0 * l1 * l2,
        ]
    }

    fn evaluate_gradients(&self, xi: &[f64; 2]) -> Vec<[f64; 2]> {
        let x = xi[0];
        let y = xi[1];
        let l0 = 1.0 - x - y;
        let l1 = x;
        let l2 = y;

        // Chain rule: grad(f) = [ df/dl1 - df/dl0, df/dl2 - df/dl0 ]
        let d_vertex = |l: f64| -> f64 { 0.5 * (27.0 * l * l - 18.0 * l + 2.0) };

        // phi0(l0): d/dl0 = d_vertex(l0), others 0
        let dl0_0 = d_vertex(l0);
        let g0 = [-dl0_0, -dl0_0];

        // phi1(l1): d/dl1 = d_vertex(l1), others 0
        let dl1_1 = d_vertex(l1);
        let g1 = [dl1_1, 0.0];

        // phi2(l2): d/dl2 = d_vertex(l2), others 0
        let dl2_2 = d_vertex(l2);
        let g2 = [0.0, dl2_2];

        // phi3 = 4.5 * l0 * l1 * (3*l0 - 1)
        // d/dl0 = 4.5 * l1 * (6*l0 - 1)
        // d/dl1 = 4.5 * l0 * (3*l0 - 1)
        let d3_l0 = 4.5 * l1 * (6.0 * l0 - 1.0);
        let d3_l1 = 4.5 * l0 * (3.0 * l0 - 1.0);
        let g3 = [d3_l1 - d3_l0, -d3_l0];

        // phi4 = 4.5 * l0 * l1 * (3*l1 - 1)
        // d/dl0 = 4.5 * l1 * (3*l1 - 1)
        // d/dl1 = 4.5 * l0 * (6*l1 - 1)
        let d4_l0 = 4.5 * l1 * (3.0 * l1 - 1.0);
        let d4_l1 = 4.5 * l0 * (6.0 * l1 - 1.0);
        let g4 = [d4_l1 - d4_l0, -d4_l0];

        // phi5 = 4.5 * l1 * l2 * (3*l1 - 1)
        // d/dl1 = 4.5 * l2 * (6*l1 - 1)
        // d/dl2 = 4.5 * l1 * (3*l1 - 1)
        let d5_l1 = 4.5 * l2 * (6.0 * l1 - 1.0);
        let d5_l2 = 4.5 * l1 * (3.0 * l1 - 1.0);
        let g5 = [d5_l1, d5_l2];

        // phi6 = 4.5 * l1 * l2 * (3*l2 - 1)
        // d/dl1 = 4.5 * l2 * (3*l2 - 1)
        // d/dl2 = 4.5 * l1 * (6*l2 - 1)
        let d6_l1 = 4.5 * l2 * (3.0 * l2 - 1.0);
        let d6_l2 = 4.5 * l1 * (6.0 * l2 - 1.0);
        let g6 = [d6_l1, d6_l2];

        // phi7 = 4.5 * l2 * l0 * (3*l2 - 1)
        // d/dl0 = 4.5 * l2 * (3*l2 - 1)
        // d/dl2 = 4.5 * l0 * (6*l2 - 1)
        let d7_l0 = 4.5 * l2 * (3.0 * l2 - 1.0);
        let d7_l2 = 4.5 * l0 * (6.0 * l2 - 1.0);
        let g7 = [-d7_l0, d7_l2 - d7_l0];

        // phi8 = 4.5 * l2 * l0 * (3*l0 - 1)
        // d/dl0 = 4.5 * l2 * (6*l0 - 1)
        // d/dl2 = 4.5 * l0 * (3*l0 - 1)
        let d8_l0 = 4.5 * l2 * (6.0 * l0 - 1.0);
        let d8_l2 = 4.5 * l0 * (3.0 * l0 - 1.0);
        let g8 = [-d8_l0, d8_l2 - d8_l0];

        // phi9 = 27.0 * l0 * l1 * l2
        // d/dl0 = 27 * l1 * l2
        // d/dl1 = 27 * l0 * l2
        // d/dl2 = 27 * l0 * l1
        let d9_l0 = 27.0 * l1 * l2;
        let d9_l1 = 27.0 * l0 * l2;
        let d9_l2 = 27.0 * l0 * l1;
        let g9 = [d9_l1 - d9_l0, d9_l2 - d9_l0];

        vec![g0, g1, g2, g3, g4, g5, g6, g7, g8, g9]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_element_properties<const DIM: usize, E: FiniteElement<DIM>>(
        elem: &E,
        test_points: &[[f64; DIM]],
    ) {
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
    fn test_p2_interval() {
        test_element_properties(&P2Interval, &[[0.25], [0.333], [0.75]]);
    }

    #[test]
    fn test_p3_interval() {
        test_element_properties(&P3Interval, &[[0.15], [0.45], [0.82]]);
    }

    #[test]
    fn test_p1_triangle() {
        test_element_properties(&P1Triangle, &[[0.2, 0.2], [0.333, 0.333], [0.1, 0.7]]);
    }

    #[test]
    fn test_p2_triangle() {
        test_element_properties(&P2Triangle, &[[0.2, 0.2], [0.333, 0.333], [0.1, 0.7]]);
    }

    #[test]
    fn test_p3_triangle() {
        test_element_properties(&P3Triangle, &[[0.2, 0.2], [0.333, 0.333], [0.1, 0.7]]);
    }

    #[test]
    fn test_p1_tetrahedron() {
        test_element_properties(
            &P1Tetrahedron,
            &[[0.25, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.1, 0.1]],
        );
    }

    #[test]
    fn test_p2_tetrahedron() {
        test_element_properties(
            &P2Tetrahedron,
            &[[0.25, 0.25, 0.25], [0.1, 0.2, 0.3], [0.5, 0.1, 0.1]],
        );
    }
}

//! Vector-valued and Tensor-valued finite elements.
//!
//! Replicates scalar basis functions across multiple vector/tensor components,
//! matching DOLFINx / Basix `VectorElement` and `TensorElement` semantics.

use crate::lagrange::FiniteElement;

/// Trait for vector-valued finite elements on reference simplices.
pub trait VectorFiniteElement<const DIM: usize, const VEC_DIM: usize> {
    /// Total number of DOFs (scalar DOFs * VEC_DIM).
    fn num_dofs(&self) -> usize;

    /// Polynomial degree.
    fn degree(&self) -> usize;

    /// Evaluate all vector basis functions at reference coordinate xi.
    /// Returns Vec of length num_dofs(), each having VEC_DIM components.
    fn evaluate_basis(&self, xi: &[f64; DIM]) -> Vec<[f64; VEC_DIM]>;

    /// Evaluate gradients of all vector basis functions.
    /// Returns [ [d/dx_0, d/dx_1, ...]_comp0, [d/dx_0, d/dx_1, ...]_comp1, ... ]
    fn evaluate_gradients(&self, xi: &[f64; DIM]) -> Vec<[[f64; DIM]; VEC_DIM]>;
}

/// Vector finite element constructed from a scalar finite element.
#[derive(Clone, Copy, Debug)]
pub struct VectorElement<const DIM: usize, const VEC_DIM: usize, E: FiniteElement<DIM>> {
    pub scalar_element: E,
}

impl<const DIM: usize, const VEC_DIM: usize, E: FiniteElement<DIM>> VectorElement<DIM, VEC_DIM, E> {
    pub fn new(scalar_element: E) -> Self {
        Self { scalar_element }
    }
}

impl<const DIM: usize, const VEC_DIM: usize, E: FiniteElement<DIM>>
    VectorFiniteElement<DIM, VEC_DIM> for VectorElement<DIM, VEC_DIM, E>
{
    fn num_dofs(&self) -> usize {
        self.scalar_element.num_dofs() * VEC_DIM
    }

    fn degree(&self) -> usize {
        self.scalar_element.degree()
    }

    fn evaluate_basis(&self, xi: &[f64; DIM]) -> Vec<[f64; VEC_DIM]> {
        let scalar_vals = self.scalar_element.evaluate_basis(xi);
        let num_scalar_dofs = scalar_vals.len();
        let mut basis = Vec::with_capacity(num_scalar_dofs * VEC_DIM);

        for val in scalar_vals {
            for comp in 0..VEC_DIM {
                let mut v = [0.0; VEC_DIM];
                v[comp] = val;
                basis.push(v);
            }
        }
        basis
    }

    fn evaluate_gradients(&self, xi: &[f64; DIM]) -> Vec<[[f64; DIM]; VEC_DIM]> {
        let scalar_grads = self.scalar_element.evaluate_gradients(xi);
        let num_scalar_dofs = scalar_grads.len();
        let mut grads = Vec::with_capacity(num_scalar_dofs * VEC_DIM);

        for grad in scalar_grads {
            for comp in 0..VEC_DIM {
                let mut g = [[0.0; DIM]; VEC_DIM];
                g[comp] = grad;
                grads.push(g);
            }
        }
        grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrange::{P1Triangle, P2Triangle};

    #[test]
    fn test_vector_element_p1_triangle() {
        let vec_elem = VectorElement::<2, 2, _>::new(P1Triangle);
        assert_eq!(vec_elem.num_dofs(), 6);
        assert_eq!(vec_elem.degree(), 1);

        let pt = [0.25, 0.25];
        let basis = vec_elem.evaluate_basis(&pt);
        assert_eq!(basis.len(), 6);

        // Sum of vector components should equal [1.0, 1.0] by partition of unity
        let mut sum = [0.0; 2];
        for v in &basis {
            sum[0] += v[0];
            sum[1] += v[1];
        }
        approx::assert_relative_eq!(sum[0], 1.0);
        approx::assert_relative_eq!(sum[1], 1.0);
    }

    #[test]
    fn test_vector_element_p2_stokes_velocity() {
        let vel_elem = VectorElement::<2, 2, _>::new(P2Triangle);
        assert_eq!(vel_elem.num_dofs(), 12);
    }
}

//! Symbolic representation and compile-time evaluation of variational forms.

use fenics_element::{
    jacobian::AffineSimplexMap,
    lagrange::{FiniteElement, P1Triangle},
    quadrature::QuadratureRule,
};

/// Type of differential integrand in a bilinear form a(u, v).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FormKernel {
    /// inner(grad(u), grad(v)) * dx  -> Stiffness matrix
    GradGrad,
    /// u * v * dx                    -> Mass matrix
    Mass,
    /// (inner(grad(u), grad(v)) - k^2 * u * v) * dx -> Helmholtz wave equation
    Helmholtz { k: f64 },
    /// (kappa * inner(grad(u), grad(v)) + c * u * v) * dx -> Reaction-Diffusion
    ReactionDiffusion { kappa: f64, c: f64 },
}

/// A bilinear form a(u, v) defined on a domain with measure dx.
#[derive(Clone, Debug)]
pub struct BilinearForm {
    pub kernel: FormKernel,
}

impl BilinearForm {
    pub fn new(kernel: FormKernel) -> Self {
        Self { kernel }
    }

    /// Compute the local element matrix A^K_{ij} for a 2D affine triangle.
    pub fn tabulate_element_2d(
        &self,
        map: &AffineSimplexMap<2>,
        elem: &P1Triangle,
        quad: &QuadratureRule<2>,
    ) -> [[f64; 3]; 3] {
        let n_dofs = elem.num_dofs();
        let mut a_elem = [[0.0; 3]; 3];
        let det_j = map.det_jacobian.abs();

        let ref_grads = elem.evaluate_gradients(&[0.0, 0.0]);
        let mut phys_grads = [[0.0, 0.0]; 3];
        for i in 0..n_dofs {
            phys_grads[i] = map.transform_gradient(&ref_grads[i]);
        }

        match self.kernel {
            FormKernel::GradGrad => {
                // For linear P1 triangles, physical gradients are constant across the element!
                let area = 0.5 * det_j;
                for i in 0..3 {
                    for j in 0..3 {
                        let dot_g = phys_grads[i][0] * phys_grads[j][0]
                            + phys_grads[i][1] * phys_grads[j][1];
                        a_elem[i][j] = dot_g * area;
                    }
                }
            }
            FormKernel::Mass => {
                for q in &quad.points {
                    let phi = elem.evaluate_basis(&q.point);
                    let weight_det = q.weight * det_j;
                    for i in 0..3 {
                        for j in 0..3 {
                            a_elem[i][j] += weight_det * phi[i] * phi[j];
                        }
                    }
                }
            }
            FormKernel::Helmholtz { k } => {
                let area = 0.5 * det_j;
                let k_sq = k * k;

                for i in 0..3 {
                    for j in 0..3 {
                        let dot_g = phys_grads[i][0] * phys_grads[j][0]
                            + phys_grads[i][1] * phys_grads[j][1];
                        a_elem[i][j] = dot_g * area;
                    }
                }

                for q in &quad.points {
                    let phi = elem.evaluate_basis(&q.point);
                    let weight_det = q.weight * det_j;
                    for i in 0..3 {
                        for j in 0..3 {
                            a_elem[i][j] -= k_sq * weight_det * phi[i] * phi[j];
                        }
                    }
                }
            }
            FormKernel::ReactionDiffusion { kappa, c } => {
                let area = 0.5 * det_j;

                for i in 0..3 {
                    for j in 0..3 {
                        let dot_g = phys_grads[i][0] * phys_grads[j][0]
                            + phys_grads[i][1] * phys_grads[j][1];
                        a_elem[i][j] = kappa * dot_g * area;
                    }
                }

                for q in &quad.points {
                    let phi = elem.evaluate_basis(&q.point);
                    let weight_det = q.weight * det_j;
                    for i in 0..3 {
                        for j in 0..3 {
                            a_elem[i][j] += c * weight_det * phi[i] * phi[j];
                        }
                    }
                }
            }
        }

        a_elem
    }
}

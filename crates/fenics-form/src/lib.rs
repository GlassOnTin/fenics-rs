//! Symbolic Variational Form DSL and compile-time weak form evaluation.

pub mod form;
#[macro_use]
pub mod macros;

pub use form::{BilinearForm, FormKernel};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use fenics_element::{
        jacobian::AffineSimplexMap, lagrange::P1Triangle, quadrature::triangle_quadrature,
    };

    #[test]
    fn test_var_form_macro_syntax() {
        let a = var_form!(inner(grad(u), grad(v)) * dx);
        assert_eq!(a.kernel, FormKernel::GradGrad);

        let m = var_form!(u * v * dx);
        assert_eq!(m.kernel, FormKernel::Mass);

        let k = 3.0;
        let h = var_form!((inner(grad(u), grad(v)) - k * k * u * v) * dx);
        assert_eq!(h.kernel, FormKernel::Helmholtz { k: 3.0 });
    }

    #[test]
    fn test_var_form_tabulation_against_analytical() {
        // Reference triangle: (0,0), (1,0), (0,1). Area = 0.5
        let verts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let map = AffineSimplexMap::from_triangle_vertices(&verts).unwrap();
        let elem = P1Triangle;
        let quad = triangle_quadrature(2);

        // 1. Stiffness via var_form!
        let a = var_form!(inner(grad(u), grad(v)) * dx);
        let k_local = a.tabulate_element_2d(&map, &elem, &quad);

        // Gradients on ref triangle:
        // phi0: [-1, -1] -> norm_sq = 2.0 -> K_00 = 2.0 * 0.5 = 1.0
        // phi1: [ 1,  0] -> norm_sq = 1.0 -> K_11 = 1.0 * 0.5 = 0.5
        // phi2: [ 0,  1] -> norm_sq = 1.0 -> K_22 = 1.0 * 0.5 = 0.5
        // phi0 . phi1 = -1.0 -> K_01 = -0.5
        // phi0 . phi2 = -1.0 -> K_02 = -0.5
        // phi1 . phi2 =  0.0 -> K_12 = 0.0
        assert_relative_eq!(k_local[0][0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(k_local[1][1], 0.5, epsilon = 1e-12);
        assert_relative_eq!(k_local[2][2], 0.5, epsilon = 1e-12);
        assert_relative_eq!(k_local[0][1], -0.5, epsilon = 1e-12);
        assert_relative_eq!(k_local[0][2], -0.5, epsilon = 1e-12);
        assert_relative_eq!(k_local[1][2], 0.0, epsilon = 1e-12);

        // 2. Mass via var_form!
        let m_form = var_form!(u * v * dx);
        let m_local = m_form.tabulate_element_2d(&map, &elem, &quad);

        // Mass matrix on ref triangle of area 0.5:
        // Diagonal: Area / 6 = 0.5 / 6 = 1/12
        // Off-diagonal: Area / 12 = 0.5 / 12 = 1/24
        assert_relative_eq!(m_local[0][0], 1.0 / 12.0, epsilon = 1e-12);
        assert_relative_eq!(m_local[1][1], 1.0 / 12.0, epsilon = 1e-12);
        assert_relative_eq!(m_local[0][1], 1.0 / 24.0, epsilon = 1e-12);
    }
}

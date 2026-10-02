//! Raviart-Thomas H(div) finite elements.
//!
//! Raviart-Thomas spaces provide normal flux continuity across element boundaries
//! with square-integrable divergence, fundamental for Darcy flow, mixed Poisson,
//! and groundwater simulations.

/// Lowest order Raviart-Thomas element on 2D triangle (RT0 / RT1 in FEniCS notation).
#[derive(Clone, Copy, Debug, Default)]
pub struct RT1Triangle;

impl RT1Triangle {
    /// Number of DOFs (3 edges in 2D triangle).
    pub fn num_dofs(&self) -> usize {
        3
    }

    /// Polynomial degree.
    pub fn degree(&self) -> usize {
        1
    }

    /// Evaluate vector basis functions [psi_0, psi_1, psi_2] at (x, y).
    pub fn evaluate_basis(&self, xi: &[f64; 2]) -> Vec<[f64; 2]> {
        let x = xi[0];
        let y = xi[1];
        vec![
            [x, y],       // Edge 0 (hypotenuse x+y=1)
            [x - 1.0, y], // Edge 1 (x=0)
            [x, y - 1.0], // Edge 2 (y=0)
        ]
    }

    /// Evaluate divergence div(psi_i) at (x, y).
    pub fn evaluate_div(&self, _xi: &[f64; 2]) -> Vec<f64> {
        vec![2.0, 2.0, 2.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_raviart_thomas_fluxes() {
        let rt = RT1Triangle;

        // On Edge 0: x + y = 1. Normal n = [1/sqrt(2), 1/sqrt(2)]
        let mid_e0 = [0.5, 0.5];
        let vals_e0 = rt.evaluate_basis(&mid_e0);
        let n0 = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];
        let flux0_e0 = vals_e0[0][0] * n0[0] + vals_e0[0][1] * n0[1];
        let flux1_e0 = vals_e0[1][0] * n0[0] + vals_e0[1][1] * n0[1];
        let flux2_e0 = vals_e0[2][0] * n0[0] + vals_e0[2][1] * n0[1];
        // Length of hypotenuse is sqrt(2), so flux * length = 1.0 for psi_0 and 0.0 for others
        assert_relative_eq!(flux0_e0 * 2.0_f64.sqrt(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(flux1_e0 * 2.0_f64.sqrt(), 0.0, epsilon = 1e-12);
        assert_relative_eq!(flux2_e0 * 2.0_f64.sqrt(), 0.0, epsilon = 1e-12);

        // On Edge 1: x = 0. Normal n = [-1, 0]
        let mid_e1 = [0.0, 0.5];
        let vals_e1 = rt.evaluate_basis(&mid_e1);
        let n1 = [-1.0, 0.0];
        let flux0_e1 = vals_e1[0][0] * n1[0] + vals_e1[0][1] * n1[1];
        let flux1_e1 = vals_e1[1][0] * n1[0] + vals_e1[1][1] * n1[1];
        let flux2_e1 = vals_e1[2][0] * n1[0] + vals_e1[2][1] * n1[1];
        assert_relative_eq!(flux0_e1, 0.0, epsilon = 1e-12);
        assert_relative_eq!(flux1_e1, 1.0, epsilon = 1e-12);
        assert_relative_eq!(flux2_e1, 0.0, epsilon = 1e-12);

        // Divergence is constantly 2.0 everywhere
        let divs = rt.evaluate_div(&[0.2, 0.3]);
        assert_eq!(divs, vec![2.0, 2.0, 2.0]);
    }
}

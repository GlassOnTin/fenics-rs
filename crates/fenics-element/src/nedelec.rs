//! Nédélec H(curl) edge finite elements.
//!
//! Nédélec elements enforce tangential continuity across cell facets,
//! with square-integrable curl. They are indispensable for Maxwell's equations
//! and computational electromagnetics without spurious unphysical modes.

/// Lowest order Nédélec edge element on 2D triangle (first kind, N1curl).
#[derive(Clone, Copy, Debug, Default)]
pub struct Nedelec1Triangle;

impl Nedelec1Triangle {
    /// Number of DOFs (3 edges on triangle).
    pub fn num_dofs(&self) -> usize {
        3
    }

    /// Polynomial degree.
    pub fn degree(&self) -> usize {
        1
    }

    /// Evaluate vector basis functions [w_0, w_1, w_2] at (x, y).
    /// - w_0 associated with Edge 1-2 (hypotenuse)
    /// - w_1 associated with Edge 2-0 (vertical y-axis)
    /// - w_2 associated with Edge 0-1 (horizontal x-axis)
    pub fn evaluate_basis(&self, xi: &[f64; 2]) -> Vec<[f64; 2]> {
        let x = xi[0];
        let y = xi[1];
        vec![
            [-y, x],       // Edge 1-2: lambda_1 grad lambda_2 - lambda_2 grad lambda_1
            [-y, x - 1.0], // Edge 2-0: lambda_2 grad lambda_0 - lambda_0 grad lambda_2
            [1.0 - y, x],  // Edge 0-1: lambda_0 grad lambda_1 - lambda_1 grad lambda_0
        ]
    }

    /// Evaluate scalar 2D curl: curl(w) = d(w_y)/dx - d(w_x)/dy.
    pub fn evaluate_curl(&self, _xi: &[f64; 2]) -> Vec<f64> {
        vec![2.0, 2.0, 2.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_nedelec_circulations() {
        let ned = Nedelec1Triangle;

        // 1. Edge 0-1: x goes from 0 to 1, y = 0. Tangent = [1, 0]
        let pt_e01 = [0.5, 0.0];
        let basis_e01 = ned.evaluate_basis(&pt_e01);
        assert_relative_eq!(basis_e01[0][0], 0.0); // w0 . [1, 0] = 0
        assert_relative_eq!(basis_e01[1][0], 0.0); // w1 . [1, 0] = 0
        assert_relative_eq!(basis_e01[2][0], 1.0); // w2 . [1, 0] = 1

        // 2. Edge 2-0: y goes from 1 to 0, x = 0. Tangent = [0, -1]
        let pt_e20 = [0.0, 0.5];
        let basis_e20 = ned.evaluate_basis(&pt_e20);
        let circ0 = -basis_e20[0][1];
        let circ1 = -basis_e20[1][1];
        let circ2 = -basis_e20[2][1];
        assert_relative_eq!(circ0, 0.0);
        assert_relative_eq!(circ1, 1.0);
        assert_relative_eq!(circ2, 0.0);

        // 3. Edge 1-2: from (1,0) to (0,1). Tangent = [-1, 1]
        let pt_e12 = [0.5, 0.5];
        let basis_e12 = ned.evaluate_basis(&pt_e12);
        let circ0_12 = -basis_e12[0][0] + basis_e12[0][1];
        let circ1_12 = -basis_e12[1][0] + basis_e12[1][1];
        let circ2_12 = -basis_e12[2][0] + basis_e12[2][1];
        assert_relative_eq!(circ0_12, 1.0);
        assert_relative_eq!(circ1_12, 0.0);
        assert_relative_eq!(circ2_12, 0.0);

        // 4. Curl is constant 2.0 everywhere
        let curls = ned.evaluate_curl(&[0.3, 0.4]);
        assert_eq!(curls, vec![2.0, 2.0, 2.0]);
    }
}

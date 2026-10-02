//! Geometric Jacobian and coordinate transformations for affine simplices.

use nalgebra::{SMatrix, SVector};

/// Affine mapping between reference simplex and physical simplex.
#[derive(Clone, Debug)]
pub struct AffineSimplexMap<const DIM: usize> {
    /// Column matrix representing the Jacobian J = dx / dxi
    pub jacobian: SMatrix<f64, DIM, DIM>,
    /// Inverse transpose of Jacobian J^{-T}, used to transform gradients: grad_x = J^{-T} * grad_xi
    pub inv_trans_jacobian: SMatrix<f64, DIM, DIM>,
    /// Determinant of the Jacobian matrix det(J)
    pub det_jacobian: f64,
}

impl AffineSimplexMap<2> {
    /// Construct map for a 2D triangle given its 3 vertices [v0, v1, v2].
    pub fn from_triangle_vertices(vertices: &[[f64; 2]; 3]) -> Result<Self, &'static str> {
        let v0 = SVector::<f64, 2>::new(vertices[0][0], vertices[0][1]);
        let v1 = SVector::<f64, 2>::new(vertices[1][0], vertices[1][1]);
        let v2 = SVector::<f64, 2>::new(vertices[2][0], vertices[2][1]);

        let col0 = v1 - v0;
        let col1 = v2 - v0;

        let jacobian = SMatrix::<f64, 2, 2>::from_columns(&[col0, col1]);
        let det = jacobian.determinant();

        if det.abs() < 1e-14 {
            return Err("Degenerate triangle: zero or near-zero area");
        }

        let inv = jacobian.try_inverse().ok_or("Failed to invert Jacobian")?;
        let inv_trans = inv.transpose();

        Ok(Self {
            jacobian,
            inv_trans_jacobian: inv_trans,
            det_jacobian: det,
        })
    }

    /// Map reference coordinate xi = [xi, eta] to physical coordinate x = [x, y].
    pub fn map_to_physical(&self, v0: &[f64; 2], xi: &[f64; 2]) -> [f64; 2] {
        let xi_vec = SVector::<f64, 2>::new(xi[0], xi[1]);
        let x_vec = SVector::<f64, 2>::new(v0[0], v0[1]) + self.jacobian * xi_vec;
        [x_vec[0], x_vec[1]]
    }

    /// Transform gradient on reference element to gradient in physical coordinates.
    pub fn transform_gradient(&self, grad_ref: &[f64; 2]) -> [f64; 2] {
        let g_ref = SVector::<f64, 2>::new(grad_ref[0], grad_ref[1]);
        let g_phys = self.inv_trans_jacobian * g_ref;
        [g_phys[0], g_phys[1]]
    }
}

impl AffineSimplexMap<3> {
    /// Construct map for a 3D tetrahedron given its 4 vertices [v0, v1, v2, v3].
    pub fn from_tetrahedron_vertices(vertices: &[[f64; 3]; 4]) -> Result<Self, &'static str> {
        let v0 = SVector::<f64, 3>::new(vertices[0][0], vertices[0][1], vertices[0][2]);
        let v1 = SVector::<f64, 3>::new(vertices[1][0], vertices[1][1], vertices[1][2]);
        let v2 = SVector::<f64, 3>::new(vertices[2][0], vertices[2][1], vertices[2][2]);
        let v3 = SVector::<f64, 3>::new(vertices[3][0], vertices[3][1], vertices[3][2]);

        let col0 = v1 - v0;
        let col1 = v2 - v0;
        let col2 = v3 - v0;

        let jacobian = SMatrix::<f64, 3, 3>::from_columns(&[col0, col1, col2]);
        let det = jacobian.determinant();

        if det.abs() < 1e-14 {
            return Err("Degenerate tetrahedron: zero or near-zero volume");
        }

        let inv = jacobian.try_inverse().ok_or("Failed to invert Jacobian")?;
        let inv_trans = inv.transpose();

        Ok(Self {
            jacobian,
            inv_trans_jacobian: inv_trans,
            det_jacobian: det,
        })
    }

    /// Map reference coordinate xi = [xi, eta, zeta] to physical coordinate x = [x, y, z].
    pub fn map_to_physical(&self, v0: &[f64; 3], xi: &[f64; 3]) -> [f64; 3] {
        let xi_vec = SVector::<f64, 3>::new(xi[0], xi[1], xi[2]);
        let x_vec = SVector::<f64, 3>::new(v0[0], v0[1], v0[2]) + self.jacobian * xi_vec;
        [x_vec[0], x_vec[1], x_vec[2]]
    }

    /// Transform gradient on reference element to gradient in physical coordinates.
    pub fn transform_gradient(&self, grad_ref: &[f64; 3]) -> [f64; 3] {
        let g_ref = SVector::<f64, 3>::new(grad_ref[0], grad_ref[1], grad_ref[2]);
        let g_phys = self.inv_trans_jacobian * g_ref;
        [g_phys[0], g_phys[1], g_phys[2]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_scaled_triangle_mapping() {
        // Triangle with side lengths 2.0 along X and 3.0 along Y. Area should be 0.5 * 2 * 3 = 3.0.
        // det(J) should be 2 * 3 = 6.0 (since reference triangle area is 0.5, physical area is 0.5 * det = 3.0).
        let verts = [[1.0, 1.0], [3.0, 1.0], [1.0, 4.0]];
        let map = AffineSimplexMap::from_triangle_vertices(&verts).unwrap();
        assert_relative_eq!(map.det_jacobian, 6.0, epsilon = 1e-12);

        // Map midpoint of reference hypotenuse (0.5, 0.5)
        let phys = map.map_to_physical(&verts[0], &[0.5, 0.5]);
        assert_relative_eq!(phys[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(phys[1], 2.5, epsilon = 1e-12);

        // Test gradient transformation:
        // If u_phys(x, y) = 3*x + 5*y, then grad_x = [3, 5]
        // u_ref(xi, eta) = u_phys(1 + 2*xi, 1 + 3*eta) = 3*(1 + 2*xi) + 5*(1 + 3*eta) = 8 + 6*xi + 15*eta
        // grad_xi = [6, 15]
        let grad_phys = map.transform_gradient(&[6.0, 15.0]);
        assert_relative_eq!(grad_phys[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(grad_phys[1], 5.0, epsilon = 1e-12);
    }

    #[test]
    fn test_scaled_tetrahedron_mapping() {
        // Tetrahedron with sides 2.0, 3.0, 4.0.
        // Reference volume = 1/6. Physical volume = (1/6) * 2 * 3 * 4 = 4.0.
        // det(J) = 24.0.
        let verts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 4.0],
        ];
        let map = AffineSimplexMap::from_tetrahedron_vertices(&verts).unwrap();
        assert_relative_eq!(map.det_jacobian, 24.0, epsilon = 1e-12);

        let grad_phys = map.transform_gradient(&[2.0, 6.0, 12.0]);
        assert_relative_eq!(grad_phys[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(grad_phys[1], 2.0, epsilon = 1e-12);
        assert_relative_eq!(grad_phys[2], 3.0, epsilon = 1e-12);
    }
}

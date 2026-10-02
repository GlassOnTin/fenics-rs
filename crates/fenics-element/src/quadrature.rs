//! Gaussian quadrature rules for reference simplices (Interval, Triangle, Tetrahedron).
//!
//! All reference simplices are defined on the standard unit domain:
//! - 1D: $[0, 1]$ (Length = 1)
//! - 2D: Triangles with vertices $(0,0), (1,0), (0,1)$ (Area = $1/2$)
//! - 3D: Tetrahedra with vertices $(0,0,0), (1,0,0), (0,1,0), (0,0,1)$ (Volume = $1/6$)

#[derive(Clone, Debug, PartialEq)]
pub struct QuadraturePoint<const DIM: usize> {
    pub point: [f64; DIM],
    pub weight: f64,
}

#[derive(Clone, Debug)]
pub struct QuadratureRule<const DIM: usize> {
    pub degree: usize,
    pub points: Vec<QuadraturePoint<DIM>>,
}

impl<const DIM: usize> QuadratureRule<DIM> {
    /// Return the sum of weights, which matches the volume of the reference simplex:
    /// 1.0 for 1D, 0.5 for 2D, 1.0/6.0 for 3D.
    pub fn total_weight(&self) -> f64 {
        self.points.iter().map(|p| p.weight).sum()
    }
}

/// Quadrature rules for the 1D reference interval [0, 1].
pub fn interval_quadrature(degree: usize) -> QuadratureRule<1> {
    match degree {
        0 | 1 => {
            // Midpoint rule (1 point, exact up to degree 1)
            QuadratureRule {
                degree: 1,
                points: vec![QuadraturePoint {
                    point: [0.5],
                    weight: 1.0,
                }],
            }
        }
        2 | 3 => {
            // 2-point Gauss-Legendre (exact up to degree 3)
            let c = 1.0 / (2.0 * 3.0_f64.sqrt());
            QuadratureRule {
                degree: 3,
                points: vec![
                    QuadraturePoint {
                        point: [0.5 - c],
                        weight: 0.5,
                    },
                    QuadraturePoint {
                        point: [0.5 + c],
                        weight: 0.5,
                    },
                ],
            }
        }
        _ => {
            // 3-point Gauss-Legendre (exact up to degree 5)
            let c = 0.5 * (3.0 / 5.0_f64).sqrt();
            QuadratureRule {
                degree: 5,
                points: vec![
                    QuadraturePoint {
                        point: [0.5 - c],
                        weight: 5.0 / 18.0,
                    },
                    QuadraturePoint {
                        point: [0.5],
                        weight: 8.0 / 18.0,
                    },
                    QuadraturePoint {
                        point: [0.5 + c],
                        weight: 5.0 / 18.0,
                    },
                ],
            }
        }
    }
}

/// Quadrature rules for the 2D reference triangle (0,0)-(1,0)-(0,1). Area = 0.5.
pub fn triangle_quadrature(degree: usize) -> QuadratureRule<2> {
    match degree {
        0 | 1 => {
            // Centroid rule (1 point, exact up to degree 1)
            QuadratureRule {
                degree: 1,
                points: vec![QuadraturePoint {
                    point: [1.0 / 3.0, 1.0 / 3.0],
                    weight: 0.5,
                }],
            }
        }
        2 => {
            // Edge-midpoints rule (3 points, exact up to degree 2)
            QuadratureRule {
                degree: 2,
                points: vec![
                    QuadraturePoint {
                        point: [0.5, 0.0],
                        weight: 1.0 / 6.0,
                    },
                    QuadraturePoint {
                        point: [0.5, 0.5],
                        weight: 1.0 / 6.0,
                    },
                    QuadraturePoint {
                        point: [0.0, 0.5],
                        weight: 1.0 / 6.0,
                    },
                ],
            }
        }
        _ => {
            // 4-point rule (exact up to degree 3)
            let w_center = -27.0 / 96.0;
            let w_outer = 25.0 / 96.0;
            QuadratureRule {
                degree: 3,
                points: vec![
                    QuadraturePoint {
                        point: [1.0 / 3.0, 1.0 / 3.0],
                        weight: w_center,
                    },
                    QuadraturePoint {
                        point: [0.6, 0.2],
                        weight: w_outer,
                    },
                    QuadraturePoint {
                        point: [0.2, 0.6],
                        weight: w_outer,
                    },
                    QuadraturePoint {
                        point: [0.2, 0.2],
                        weight: w_outer,
                    },
                ],
            }
        }
    }
}

/// Quadrature rules for the 3D reference tetrahedron (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1). Volume = 1/6.
pub fn tetrahedron_quadrature(degree: usize) -> QuadratureRule<3> {
    match degree {
        0 | 1 => {
            // Centroid rule (1 point, exact up to degree 1)
            QuadratureRule {
                degree: 1,
                points: vec![QuadraturePoint {
                    point: [0.25, 0.25, 0.25],
                    weight: 1.0 / 6.0,
                }],
            }
        }
        2 => {
            // Hammer-Marlowe-Stroud / Keast degree 2 rule (4 points)
            let sqrt5 = 5.0_f64.sqrt();
            let a = (5.0 - sqrt5) / 20.0;
            let b = (5.0 + 3.0 * sqrt5) / 20.0;
            let w = 1.0 / 24.0;
            QuadratureRule {
                degree: 2,
                points: vec![
                    QuadraturePoint {
                        point: [a, a, a],
                        weight: w,
                    },
                    QuadraturePoint {
                        point: [b, a, a],
                        weight: w,
                    },
                    QuadraturePoint {
                        point: [a, b, a],
                        weight: w,
                    },
                    QuadraturePoint {
                        point: [a, a, b],
                        weight: w,
                    },
                ],
            }
        }
        _ => {
            // Keast 5-point rule (degree 3)
            // Center weight: -2/15 = -0.13333333333333333
            // Outer weights: 3/40 = 0.075
            // Sum = -2/15 + 4 * 3/40 = 5/30 = 1/6
            let w_center = -2.0 / 15.0;
            let w_outer = 3.0 / 40.0;
            QuadratureRule {
                degree: 3,
                points: vec![
                    QuadraturePoint {
                        point: [0.25, 0.25, 0.25],
                        weight: w_center,
                    },
                    QuadraturePoint {
                        point: [0.5, 1.0 / 6.0, 1.0 / 6.0],
                        weight: w_outer,
                    },
                    QuadraturePoint {
                        point: [1.0 / 6.0, 0.5, 1.0 / 6.0],
                        weight: w_outer,
                    },
                    QuadraturePoint {
                        point: [1.0 / 6.0, 1.0 / 6.0, 0.5],
                        weight: w_outer,
                    },
                    QuadraturePoint {
                        point: [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
                        weight: w_outer,
                    },
                ],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simplex_volumes() {
        for deg in 1..=3 {
            assert_relative_eq!(interval_quadrature(deg).total_weight(), 1.0, epsilon = 1e-12);
            assert_relative_eq!(triangle_quadrature(deg).total_weight(), 0.5, epsilon = 1e-12);
            assert_relative_eq!(
                tetrahedron_quadrature(deg).total_weight(),
                1.0 / 6.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_polynomial_integration_triangle() {
        let quad = triangle_quadrature(2);
        // Integrate f(x, y) = x^2 + 2*y over reference triangle
        // Exact: \int_0^1 \int_0^{1-x} (x^2 + 2y) dy dx
        // \int_0^1 [x^2(1-x) + (1-x)^2] dx = \int_0^1 (x^2 - x^3 + 1 - 2x + x^2) dx
        // = [2/3 x^3 - 1/4 x^4 + x - x^2]_0^1 = 2/3 - 1/4 + 1 - 1 = 5/12 approx 0.416666666667
        let approx_val: f64 = quad
            .points
            .iter()
            .map(|p| (p.point[0].powi(2) + 2.0 * p.point[1]) * p.weight)
            .sum();
        assert_relative_eq!(approx_val, 5.0 / 12.0, epsilon = 1e-12);
    }

    #[test]
    fn test_polynomial_integration_tetrahedron() {
        let quad = tetrahedron_quadrature(2);
        // Integrate f(x, y, z) = x^2 over reference tetrahedron
        // Exact: \int_T x^2 dV = 1/60 approx 0.016666666667
        let approx_val: f64 = quad
            .points
            .iter()
            .map(|p| p.point[0].powi(2) * p.weight)
            .sum();
        assert_relative_eq!(approx_val, 1.0 / 60.0, epsilon = 1e-12);
    }
}

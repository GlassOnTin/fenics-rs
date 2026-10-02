//! Symbolic Unified Form Language (UFL) AST and Automatic Differentiation.
//!
//! Provides a symbolic expression engine capable of computing exact Fréchet derivatives
//! (Jacobians) of nonlinear variational forms F(u; v) = 0 for Newton-Raphson solvers.

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    /// Scalar constant.
    Constant(f64),
    /// Spatial coordinate x_d (e.g. x=0, y=1, z=2).
    SpatialCoordinate(usize),
    /// Unknown trial function / increment delta_u (linear argument).
    TrialFunction { id: usize },
    /// Test function v (linear argument).
    TestFunction { id: usize },
    /// State coefficient u (known function from previous iteration or step).
    Coefficient { id: usize },
    /// Negation: -expr
    Neg(Box<Expr>),
    /// Sum: a + b
    Add(Box<Expr>, Box<Expr>),
    /// Difference: a - b
    Sub(Box<Expr>, Box<Expr>),
    /// Product: a * b
    Mul(Box<Expr>, Box<Expr>),
    /// Division: a / b
    Div(Box<Expr>, Box<Expr>),
    /// Power: base^p
    Power(Box<Expr>, f64),
    /// Spatial gradient: grad(expr)
    Grad(Box<Expr>),
    /// Spatial divergence: div(expr)
    DivOp(Box<Expr>),
    /// Inner product: inner(a, b) = a : b or a . b
    Inner(Box<Expr>, Box<Expr>),
    /// Identity tensor of dimension DIM.
    Identity(usize),
    /// Tensor transpose: A^T
    Transpose(Box<Expr>),
    /// Tensor trace: tr(A)
    Trace(Box<Expr>),
    /// Matrix determinant: det(A)
    Det(Box<Expr>),
    /// Matrix inverse: inv(A)
    Inv(Box<Expr>),
}

// Ergonomic constructors
pub fn constant(val: f64) -> Expr {
    Expr::Constant(val)
}

pub fn trial_function(id: usize) -> Expr {
    Expr::TrialFunction { id }
}

pub fn test_function(id: usize) -> Expr {
    Expr::TestFunction { id }
}

pub fn coefficient(id: usize) -> Expr {
    Expr::Coefficient { id }
}

pub fn grad(expr: Expr) -> Expr {
    Expr::Grad(Box::new(expr))
}

pub fn div(expr: Expr) -> Expr {
    Expr::DivOp(Box::new(expr))
}

pub fn inner(a: Expr, b: Expr) -> Expr {
    Expr::Inner(Box::new(a), Box::new(b))
}

pub fn identity(dim: usize) -> Expr {
    Expr::Identity(dim)
}

pub fn transpose(a: Expr) -> Expr {
    Expr::Transpose(Box::new(a)).simplify()
}

pub fn tr(a: Expr) -> Expr {
    Expr::Trace(Box::new(a)).simplify()
}

pub fn det(a: Expr) -> Expr {
    Expr::Det(Box::new(a)).simplify()
}

pub fn inv(a: Expr) -> Expr {
    Expr::Inv(Box::new(a)).simplify()
}

impl std::ops::Add for Expr {
    type Output = Expr;
    fn add(self, rhs: Expr) -> Expr {
        Expr::Add(Box::new(self), Box::new(rhs)).simplify()
    }
}

impl std::ops::Sub for Expr {
    type Output = Expr;
    fn sub(self, rhs: Expr) -> Expr {
        Expr::Sub(Box::new(self), Box::new(rhs)).simplify()
    }
}

impl std::ops::Mul for Expr {
    type Output = Expr;
    fn mul(self, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(self), Box::new(rhs)).simplify()
    }
}

impl std::ops::Neg for Expr {
    type Output = Expr;
    fn neg(self) -> Expr {
        Expr::Neg(Box::new(self)).simplify()
    }
}

impl Expr {
    pub fn powf(self, p: f64) -> Expr {
        Expr::Power(Box::new(self), p).simplify()
    }

    /// Algebraic simplification and constant folding.
    pub fn simplify(&self) -> Expr {
        match self {
            Expr::Neg(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Constant(c) => Expr::Constant(-c),
                    Expr::Neg(sub) => *sub,
                    _ => Expr::Neg(Box::new(s)),
                }
            }
            Expr::Add(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Constant(ca), Expr::Constant(cb)) => Expr::Constant(ca + cb),
                    (Expr::Constant(c), _) if *c == 0.0 => sb,
                    (_, Expr::Constant(c)) if *c == 0.0 => sa,
                    _ => Expr::Add(Box::new(sa), Box::new(sb)),
                }
            }
            Expr::Sub(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Constant(ca), Expr::Constant(cb)) => Expr::Constant(ca - cb),
                    (_, Expr::Constant(c)) if *c == 0.0 => sa,
                    (Expr::Constant(c), _) if *c == 0.0 => Expr::Neg(Box::new(sb)).simplify(),
                    _ => Expr::Sub(Box::new(sa), Box::new(sb)),
                }
            }
            Expr::Mul(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Constant(ca), Expr::Constant(cb)) => Expr::Constant(ca * cb),
                    (Expr::Constant(c), _) if *c == 0.0 => Expr::Constant(0.0),
                    (_, Expr::Constant(c)) if *c == 0.0 => Expr::Constant(0.0),
                    (Expr::Constant(c), _) if *c == 1.0 => sb,
                    (_, Expr::Constant(c)) if *c == 1.0 => sa,
                    _ => Expr::Mul(Box::new(sa), Box::new(sb)),
                }
            }
            Expr::Div(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Constant(ca), Expr::Constant(cb)) if *cb != 0.0 => {
                        Expr::Constant(ca / cb)
                    }
                    (Expr::Constant(c), _) if *c == 0.0 => Expr::Constant(0.0),
                    (_, Expr::Constant(c)) if *c == 1.0 => sa,
                    _ => Expr::Div(Box::new(sa), Box::new(sb)),
                }
            }
            Expr::Power(base, p) => {
                let s = base.simplify();
                if *p == 0.0 {
                    Expr::Constant(1.0)
                } else if *p == 1.0 {
                    s
                } else {
                    match s {
                        Expr::Constant(c) => Expr::Constant(c.powf(*p)),
                        _ => Expr::Power(Box::new(s), *p),
                    }
                }
            }
            Expr::Grad(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Constant(_) => Expr::Constant(0.0),
                    _ => Expr::Grad(Box::new(s)),
                }
            }
            Expr::DivOp(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Constant(_) => Expr::Constant(0.0),
                    _ => Expr::DivOp(Box::new(s)),
                }
            }
            Expr::Inner(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Expr::Constant(c), _) if *c == 0.0 => Expr::Constant(0.0),
                    (_, Expr::Constant(c)) if *c == 0.0 => Expr::Constant(0.0),
                    _ => Expr::Inner(Box::new(sa), Box::new(sb)),
                }
            }
            Expr::Transpose(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Constant(c) if c.abs() < 1e-15 => Expr::Constant(0.0),
                    Expr::Identity(d) => Expr::Identity(d),
                    Expr::Transpose(sub) => *sub,
                    _ => Expr::Transpose(Box::new(s)),
                }
            }
            Expr::Trace(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Constant(c) if c.abs() < 1e-15 => Expr::Constant(0.0),
                    Expr::Identity(d) => Expr::Constant(d as f64),
                    _ => Expr::Trace(Box::new(s)),
                }
            }
            Expr::Det(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Identity(_) => Expr::Constant(1.0),
                    _ => Expr::Det(Box::new(s)),
                }
            }
            Expr::Inv(inner) => {
                let s = inner.simplify();
                match s {
                    Expr::Identity(d) => Expr::Identity(d),
                    _ => Expr::Inv(Box::new(s)),
                }
            }
            other => other.clone(),
        }
    }
}

/// Compute the exact Fréchet derivative of an expression with respect to coefficient `coeff_id`.
///
/// Returns the linearized bilinear or linear expression in direction of `trial_id`:
/// J(u; \delta u) = d/du [ F(u) ] \cdot \delta u
pub fn derivative(expr: &Expr, coeff_id: usize, trial_id: usize) -> Expr {
    match expr {
        Expr::Constant(_) | Expr::SpatialCoordinate(_) => Expr::Constant(0.0),
        Expr::TestFunction { .. } => Expr::Constant(0.0),
        Expr::TrialFunction { .. } => Expr::Constant(0.0),
        Expr::Identity(_) => Expr::Constant(0.0),
        Expr::Coefficient { id } => {
            if *id == coeff_id {
                Expr::TrialFunction { id: trial_id }
            } else {
                Expr::Constant(0.0)
            }
        }
        Expr::Neg(inner) => Expr::Neg(Box::new(derivative(inner, coeff_id, trial_id))).simplify(),
        Expr::Add(a, b) => {
            let da = derivative(a, coeff_id, trial_id);
            let db = derivative(b, coeff_id, trial_id);
            (da + db).simplify()
        }
        Expr::Sub(a, b) => {
            let da = derivative(a, coeff_id, trial_id);
            let db = derivative(b, coeff_id, trial_id);
            (da - db).simplify()
        }
        Expr::Mul(a, b) => {
            // Product rule: d(a * b) = da * b + a * db
            let da = derivative(a, coeff_id, trial_id);
            let db = derivative(b, coeff_id, trial_id);
            let term1 = Expr::Mul(Box::new(da), b.clone());
            let term2 = Expr::Mul(a.clone(), Box::new(db));
            (term1 + term2).simplify()
        }
        Expr::Div(a, b) => {
            // Quotient rule: d(a / b) = (da * b - a * db) / (b^2)
            let da = derivative(a, coeff_id, trial_id);
            let db = derivative(b, coeff_id, trial_id);
            let num = (da * (**b).clone()) - ((**a).clone() * db);
            let den = (**b).clone().powf(2.0);
            Expr::Div(Box::new(num), Box::new(den)).simplify()
        }
        Expr::Power(base, p) => {
            // Power rule: d(base^p) = p * base^(p-1) * d(base)
            let d_base = derivative(base, coeff_id, trial_id);
            let p_const = Expr::Constant(*p);
            let base_pow = (**base).clone().powf(p - 1.0);
            (p_const * base_pow * d_base).simplify()
        }
        Expr::Grad(inner) => {
            let d_inner = derivative(inner, coeff_id, trial_id);
            grad(d_inner).simplify()
        }
        Expr::DivOp(inner) => {
            let d_inner = derivative(inner, coeff_id, trial_id);
            div(d_inner).simplify()
        }
        Expr::Inner(a, b) => {
            // Bilinear inner product rule: d inner(a, b) = inner(da, b) + inner(a, db)
            let da = derivative(a, coeff_id, trial_id);
            let db = derivative(b, coeff_id, trial_id);
            let term1 = inner(da, (**b).clone());
            let term2 = inner((**a).clone(), db);
            (term1 + term2).simplify()
        }
        Expr::Transpose(inner) => {
            let d_inner = derivative(inner, coeff_id, trial_id);
            transpose(d_inner)
        }
        Expr::Trace(inner) => {
            let d_inner = derivative(inner, coeff_id, trial_id);
            tr(d_inner)
        }
        Expr::Det(inner) => {
            // Jacobi's formula: d(det(A)) = det(A) * tr(inv(A) * dA)
            let d_inner = derivative(inner, coeff_id, trial_id);
            (det((**inner).clone()) * tr(inv((**inner).clone()) * d_inner)).simplify()
        }
        Expr::Inv(inner) => {
            // d(inv(A)) = -inv(A) * dA * inv(A)
            let d_inner = derivative(inner, coeff_id, trial_id);
            let inv_a = inv((**inner).clone());
            (-inv_a.clone() * d_inner * inv_a).simplify()
        }
    }
}

/// A variational form integral with measure (dx for domain, ds for exterior boundary).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Measure {
    /// Domain interior integral \int_\Omega ... dx
    Dx,
    /// Exterior boundary facet integral \int_{\partial \Omega} ... ds
    Ds,
}

/// Complete variational form AST: integrand expression and domain measure.
#[derive(Clone, Debug)]
pub struct Form {
    pub integrand: Expr,
    pub measure: Measure,
}

impl Form {
    pub fn new(integrand: Expr, measure: Measure) -> Self {
        Self { integrand, measure }
    }

    /// Compute the Fréchet derivative of this form with respect to a coefficient function u.
    pub fn derivative(&self, coeff_id: usize, trial_id: usize) -> Form {
        Form {
            integrand: derivative(&self.integrand, coeff_id, trial_id),
            measure: self.measure,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_poisson_jacobian() {
        // Nonlinear Poisson: F(u; v) = inner((1 + u^2) * grad(u), grad(v))
        let u = coefficient(0);
        let v = test_function(0);
        let one = constant(1.0);

        let integrand = inner(
            (one + u.clone().powf(2.0)) * grad(u.clone()),
            grad(v.clone()),
        );
        let res_form = Form::new(integrand, Measure::Dx);

        // Derivative with respect to u in direction delta_u (trial function 0):
        let jac_form = res_form.derivative(0, 0);

        // Expected derivative:
        // inner( (2*u*du)*grad(u) + (1 + u^2)*grad(du), grad(v) )
        // Let's verify that the differentiated tree contains both terms!
        match &jac_form.integrand {
            Expr::Inner(lhs, rhs) => {
                assert_eq!(**rhs, grad(v));
                // lhs must be a sum of two terms: one with grad(u) and one with grad(du)
                match &**lhs {
                    Expr::Add(term1, term2) => {
                        let repr = format!("{:?} + {:?}", term1, term2);
                        assert!(repr.contains("TrialFunction"));
                        assert!(repr.contains("Grad"));
                    }
                    _ => panic!("Expected Add in LHS of Inner product, got {:?}", lhs),
                }
            }
            _ => panic!("Expected Inner, got {:?}", jac_form.integrand),
        }
    }

    #[test]
    fn test_linear_poisson_derivative() {
        // Linear Poisson residual: inner(grad(u), grad(v)) - f * v
        let u = coefficient(0);
        let v = test_function(0);
        let f = constant(10.0);

        let residual = inner(grad(u), grad(v.clone())) - f * v.clone();
        let f_form = Form::new(residual, Measure::Dx);

        let jacobian = f_form.derivative(0, 0);
        let du = trial_function(0);

        // Derivative of (inner(grad(u), grad(v)) - f*v) is exactly inner(grad(du), grad(v))!
        let expected = inner(grad(du), grad(v));
        assert_eq!(jacobian.integrand, expected);
    }

    #[test]
    fn test_hyperelastic_kinematics_and_derivative() {
        // Deformation gradient F = I + grad(u)
        let u = coefficient(0);
        let f_def = identity(3) + grad(u);

        // Derivative of F in direction du:
        let d_f = derivative(&f_def, 0, 0);
        let du = trial_function(0);
        assert_eq!(d_f, grad(du));

        // Volume ratio J = det(F)
        let j_det = det(f_def);
        let d_j = derivative(&j_det, 0, 0);
        // dJ = det(F) * tr(inv(F) * grad(du))
        let repr = format!("{:?}", d_j);
        assert!(repr.contains("Det"));
        assert!(repr.contains("Trace"));
        assert!(repr.contains("Inv"));
        assert!(repr.contains("TrialFunction"));
    }
}

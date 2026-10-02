//! Declarative macro DSL for compile-time variational form definitions.

/// Variational form expression macro inspired by FEniCS UFL.
///
/// Syntax examples:
/// ```rust
/// use fenics_form::{var_form, FormKernel};
///
/// // Poisson equation stiffness: \int_\Omega \nabla u \cdot \nabla v dx
/// let a = var_form!(inner(grad(u), grad(v)) * dx);
/// assert_eq!(a.kernel, FormKernel::GradGrad);
///
/// // Mass matrix: \int_\Omega u v dx
/// let m = var_form!(u * v * dx);
/// assert_eq!(m.kernel, FormKernel::Mass);
///
/// // Helmholtz acoustics: \int_\Omega (\nabla u \cdot \nabla v - k^2 u v) dx
/// let k = 5.0;
/// let h = var_form!((inner(grad(u), grad(v)) - k * k * u * v) * dx);
/// ```
#[macro_export]
macro_rules! var_form {
    // 1. Poisson / Laplace stiffness: inner(grad(u), grad(v)) * dx
    (inner(grad($u:ident), grad($v:ident)) * dx) => {
        $crate::BilinearForm::new($crate::FormKernel::GradGrad)
    };

    // 2. Mass matrix: u * v * dx
    ($u:ident * $v:ident * dx) => {
        $crate::BilinearForm::new($crate::FormKernel::Mass)
    };

    // 3. Helmholtz: (inner(grad(u), grad(v)) - $k:expr * $k2:expr * u * v) * dx
    ((inner(grad($u:ident), grad($v:ident)) - $k:ident * $k2:ident * $u2:ident * $v2:ident) * dx) => {
        $crate::BilinearForm::new($crate::FormKernel::Helmholtz { k: $k as f64 })
    };

    // 4. Reaction-Diffusion: (kappa * inner(grad(u), grad(v)) + c * u * v) * dx
    (($kappa:tt * inner(grad($u:ident), grad($v:ident)) + $c:tt * $u2:ident * $v2:ident) * dx) => {
        $crate::BilinearForm::new($crate::FormKernel::ReactionDiffusion {
            kappa: $kappa as f64,
            c: $c as f64,
        })
    };
}

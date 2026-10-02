//! Transient Heat Conduction and Thermal Transport solver.
//!
//! Solves the time-dependent heat equation:
//! rho * c_p * \partial T / \partial t - \nabla \cdot (k \nabla T) = Q(x, t)
//! using implicit Crank-Nicolson or Backward Euler time integration.

use crate::transient::{TimeSteppingScheme, TransientHeatSolver};
use faer::sparse::SparseColMat;
use fenics_assembly::mass::assemble_mass_2d;
use fenics_assembly::poisson::{assemble_stiffness_2d, DirichletBC};
use fenics_mesh::TriangleMesh;

/// Solution record for transient heat conduction.
#[derive(Clone, Debug)]
pub struct TransientHeatSolution {
    pub time_steps: Vec<f64>,
    pub temperature_history: Vec<Vec<f64>>,
    pub final_temperature: Vec<f64>,
}

/// Solve 2D transient heat equation on a triangle mesh.
#[allow(clippy::too_many_arguments)]
pub fn solve_transient_heat_2d<F>(
    mesh: &TriangleMesh,
    diffusivity: f64,
    u_initial: &[f64],
    bcs: &[DirichletBC],
    dt: f64,
    num_steps: usize,
    scheme: TimeSteppingScheme,
    mut heat_source: F,
) -> Result<TransientHeatSolution, String>
where
    F: FnMut(f64, &[f64; 2]) -> f64,
{
    let n = mesh.num_vertices();
    if u_initial.len() != n {
        return Err(format!(
            "Initial temperature size {} != mesh vertices {}",
            u_initial.len(),
            n
        ));
    }

    let mass = assemble_mass_2d(mesh);
    let mut stiff = assemble_stiffness_2d(mesh);

    // Scale stiffness by thermal diffusivity alpha = k / (rho * c_p)
    let mut stiff_triplets = Vec::new();
    for col in 0..n {
        let r_inds = stiff.as_ref().row_indices_of_col(col);
        let vals = stiff.as_ref().values_of_col(col);
        for (r, &val) in r_inds.zip(vals) {
            use faer::Unbind;
            stiff_triplets.push((r.unbound(), col, diffusivity * val));
        }
    }
    stiff = SparseColMat::try_new_from_triplets(n, n, &stiff_triplets).unwrap();

    let stepper = TransientHeatSolver::new(scheme, dt);
    let mut u = u_initial.to_vec();

    let mut time_steps = Vec::with_capacity(num_steps + 1);
    let mut history = Vec::with_capacity(num_steps + 1);
    time_steps.push(0.0);
    history.push(u.clone());

    for step in 1..=num_steps {
        let t_curr = (step - 1) as f64 * dt;
        let t_next = step as f64 * dt;

        // Compute heat source RHS at t_curr and t_next
        let mut f_curr = vec![0.0; n];
        let mut f_next = vec![0.0; n];
        for i in 0..n {
            let pt = &mesh.vertices[i];
            f_curr[i] = heat_source(t_curr, pt);
            f_next[i] = heat_source(t_next, pt);
        }

        // Apply Dirichlet boundary conditions to current state
        for bc in bcs {
            for (&dof, &val) in bc.dofs.iter().zip(bc.values.iter()) {
                u[dof] = val;
            }
        }

        // Advance one time step
        let mut u_next = stepper.step(&mass, &stiff, &u, &f_curr, &f_next)?;

        // Enforce Dirichlet BCs on updated state
        for bc in bcs {
            for (&dof, &val) in bc.dofs.iter().zip(bc.values.iter()) {
                u_next[dof] = val;
            }
        }

        u = u_next;
        time_steps.push(t_next);
        history.push(u.clone());
    }

    Ok(TransientHeatSolution {
        time_steps,
        temperature_history: history,
        final_temperature: u,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use fenics_mesh::generators::unit_square;

    #[test]
    fn test_transient_heat_conduction_convergence_to_steady_state() {
        // Heated left wall at T=100.0, Cold right wall at T=0.0.
        // As t -> inf, temperature profile approaches linear T(x) = 100 * (1 - x).
        let mesh = unit_square(8, 8);
        let n = mesh.num_vertices();

        let mut left_dofs = Vec::new();
        let mut right_dofs = Vec::new();

        for i in 0..n {
            let x = mesh.vertices[i][0];
            if x.abs() < 1e-6 {
                left_dofs.push(i);
            } else if (x - 1.0).abs() < 1e-6 {
                right_dofs.push(i);
            }
        }

        let bcs = vec![
            DirichletBC {
                dofs: left_dofs.clone(),
                values: vec![100.0; left_dofs.len()],
            },
            DirichletBC {
                dofs: right_dofs.clone(),
                values: vec![0.0; right_dofs.len()],
            },
        ];

        let u0 = vec![0.0; n];
        let diffusivity = 1.0;
        let dt = 0.05;
        let num_steps = 40; // Total time t = 2.0 (thermal diffusion time scale L^2 / alpha = 1.0)

        let sol = solve_transient_heat_2d(
            &mesh,
            diffusivity,
            &u0,
            &bcs,
            dt,
            num_steps,
            TimeSteppingScheme::CrankNicolson,
            |_t, _pt| 0.0,
        )
        .unwrap();

        // At center x = 0.5, steady state temperature is exactly 50.0
        for i in 0..n {
            let x = mesh.vertices[i][0];
            if (x - 0.5).abs() < 1e-6 {
                let t_val = sol.final_temperature[i];
                assert_relative_eq!(t_val, 50.0, epsilon = 1.5); // Within 1.5% of steady-state
            }
        }
    }
}

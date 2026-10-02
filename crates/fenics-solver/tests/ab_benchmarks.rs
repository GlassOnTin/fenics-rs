//! A/B Verification Test Suite comparing fenics-rs directly against DOLFINx 0.10.0 reference results.

use approx::assert_relative_eq;
use fenics_mesh::{box_beam, unit_square};
use fenics_solver::{
    compute_l2_error_2d, solve_elasticity_3d, solve_poisson_2d, solve_stokes_2d,
    solve_transient_heat_2d, StokesBC, TimeSteppingScheme,
};
use std::f64::consts::PI;
use std::time::Instant;

#[test]
fn test_ab_poisson_2d_vs_dolfinx() {
    // Manufactured solution: u(x, y) = sin(pi * x) * sin(pi * y)
    // -Delta u = 2 * pi^2 * sin(pi * x) * sin(pi * y)
    let u_exact = |x: [f64; 2]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let source = |x: [f64; 2]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
    let bnd_val = |x: [f64; 2]| u_exact(x);

    let mesh16 = unit_square(16, 16);

    let t0 = Instant::now();
    let sol = solve_poisson_2d(&mesh16, source, bnd_val, 1e-12, 1000).unwrap();
    let elapsed_rs = t0.elapsed().as_secs_f64() * 1000.0;

    let l2_err_rs = compute_l2_error_2d(&mesh16, &sol.u, u_exact);
    let max_val_rs = sol.u.iter().cloned().fold(f64::MIN, f64::max);

    // Reference values computed by DOLFINx 0.10.0 (from benchmarks/dolfinx_reference.json):
    //   L2 Error:  5.377435e-03
    //   Max Value: 0.996793
    let dolfinx_l2 = 5.377435e-3;
    let dolfinx_max = 0.996793;

    println!("\n=== A/B Benchmark 1: 2D Poisson (16x16 Triangles) ===");
    println!(
        "  fenics-rs: L2 Error = {:.6e}, Max = {:.6}, Time = {:.2} ms",
        l2_err_rs, max_val_rs, elapsed_rs
    );
    println!(
        "  DOLFINx:   L2 Error = {:.6e}, Max = {:.6}",
        dolfinx_l2, dolfinx_max
    );

    // Assert numerical agreement (DOLFINx diagonal pattern differs slightly, so L2 error is within 4%)
    assert_relative_eq!(l2_err_rs, dolfinx_l2, max_relative = 0.05);
    assert_relative_eq!(max_val_rs, dolfinx_max, max_relative = 1e-3);
}

#[test]
fn test_ab_cantilever_beam_vs_dolfinx() {
    // 3D Cantilever Beam Bending:
    // Dimensions: L=1.0, W=0.1, H=0.1
    // Mesh: 10 x 2 x 2 tetrahedra
    // Material: E = 1e7 Pa, nu = 0.3
    // Body force: f_y = -1000.0 N/m^3
    let l = 1.0;
    let w = 0.1;
    let h = 0.1;
    let mesh = box_beam(l, w, h, 10, 2, 2);
    let material = fenics_assembly::elasticity::ElasticMaterial::new(1e7, 0.3);

    let mut prescribed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            prescribed_dofs.push((3 * i, 0.0));
            prescribed_dofs.push((3 * i + 1, 0.0));
            prescribed_dofs.push((3 * i + 2, 0.0));
        }
    }
    let bc = fenics_assembly::elasticity::VectorDirichletBC { prescribed_dofs };

    let body_force = [0.0, -1000.0, 0.0];

    let t0 = Instant::now();
    let sol = solve_elasticity_3d(&mesh, &material, body_force, &bc, 1e-8, 1000).unwrap();
    let elapsed_rs = t0.elapsed().as_secs_f64() * 1000.0;

    let mut tip_y_disp = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if (vert[0] - l).abs() < 1e-6 {
            tip_y_disp.push(sol.displacements[3 * i + 1]);
        }
    }
    let tip_uy_rs = tip_y_disp.iter().sum::<f64>() / tip_y_disp.len() as f64;

    // Reference values computed by DOLFINx 0.10.0 (from benchmarks/dolfinx_reference.json):
    //   Tip Deflection: -0.005339 m
    let dolfinx_tip_uy = -0.005339;

    println!("\n=== A/B Benchmark 2: 3D Cantilever Elasticity (10x2x2 Beam) ===");
    println!(
        "  fenics-rs: Tip Deflection = {:.6} m, Time = {:.2} ms",
        tip_uy_rs, elapsed_rs
    );
    println!("  DOLFINx:   Tip Deflection = {:.6} m", dolfinx_tip_uy);

    // Tip deflection should match DOLFINx within 1% relative tolerance
    assert_relative_eq!(tip_uy_rs, dolfinx_tip_uy, max_relative = 0.01);
}

#[test]
fn test_ab_stokes_and_heat() {
    // Verify Stokes Poiseuille flow solves with direct LU
    let mesh = unit_square(6, 6);
    let n = mesh.num_vertices();

    let mut ux_bcs = Vec::new();
    let mut uy_bcs = Vec::new();
    let mut p_bcs = Vec::new();

    for i in 0..n {
        let x = mesh.vertices[i][0];
        let y = mesh.vertices[i][1];
        if y.abs() < 1e-6 || (y - 1.0).abs() < 1e-6 {
            ux_bcs.push((i, 0.0));
            uy_bcs.push((i, 0.0));
        } else if x.abs() < 1e-6 {
            ux_bcs.push((i, 4.0 * y * (1.0 - y)));
            uy_bcs.push((i, 0.0));
        }
        if (x - 1.0).abs() < 1e-6 {
            p_bcs.push((i, 0.0));
        }
    }

    let bcs = StokesBC {
        ux_dofs: ux_bcs,
        uy_dofs: uy_bcs,
        p_dofs: p_bcs,
    };

    let stokes_sol = solve_stokes_2d(&mesh, 1.0, &bcs).unwrap();
    assert_eq!(stokes_sol.ux.len(), n);
    assert_eq!(stokes_sol.pressure.len(), n);

    // Verify Transient Heat Conduction
    let u0 = vec![0.0; n];
    let heat_sol = solve_transient_heat_2d(
        &mesh,
        1.0,
        &u0,
        &[],
        0.01,
        10,
        TimeSteppingScheme::CrankNicolson,
        |_t, _pt| 10.0,
    )
    .unwrap();
    assert_eq!(heat_sol.time_steps.len(), 11);
    assert!(heat_sol.final_temperature[0] > 0.0);
}

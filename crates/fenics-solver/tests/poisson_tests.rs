use fenics_mesh::{unit_cube, unit_square};
use fenics_solver::{compute_l2_error_2d, compute_l2_error_3d, solve_poisson_2d, solve_poisson_3d};
use std::f64::consts::PI;

#[test]
fn test_poisson_2d_manufactured_solution() {
    // Exact solution: u(x, y) = sin(pi * x) * sin(pi * y)
    // -Delta u = 2 * pi^2 * sin(pi * x) * sin(pi * y)
    let u_exact = |x: [f64; 2]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let source = |x: [f64; 2]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
    let bnd_val = |x: [f64; 2]| u_exact(x);

    // Solve on N=8 mesh
    let mesh8 = unit_square(8, 8);
    let sol8 = solve_poisson_2d(&mesh8, source, bnd_val, 1e-10, 500).unwrap();
    let err8 = compute_l2_error_2d(&mesh8, &sol8.u, u_exact);

    // Solve on N=16 mesh
    let mesh16 = unit_square(16, 16);
    let sol16 = solve_poisson_2d(&mesh16, source, bnd_val, 1e-10, 500).unwrap();
    let err16 = compute_l2_error_2d(&mesh16, &sol16.u, u_exact);

    // Solve on N=32 mesh
    let mesh32 = unit_square(32, 32);
    let sol32 = solve_poisson_2d(&mesh32, source, bnd_val, 1e-10, 500).unwrap();
    let err32 = compute_l2_error_2d(&mesh32, &sol32.u, u_exact);

    println!("2D Poisson L2 Errors: N=8: {:.5e}, N=16: {:.5e}, N=32: {:.5e}", err8, err16, err32);

    // Check O(h^2) convergence rate: err(h) / err(h/2) should be approximately 4.0
    let rate1 = err8 / err16;
    let rate2 = err16 / err32;
    println!("Convergence rates: N8->N16: {:.2}, N16->N32: {:.2}", rate1, rate2);

    // Rates for linear P1 elements should be between 3.8 and 4.2
    assert!(rate1 > 3.8 && rate1 < 4.2, "Expected quadratic convergence rate, got {}", rate1);
    assert!(rate2 > 3.8 && rate2 < 4.2, "Expected quadratic convergence rate, got {}", rate2);
}

#[test]
fn test_poisson_3d_manufactured_solution() {
    // Exact solution: u(x, y, z) = sin(pi * x) * sin(pi * y) * sin(pi * z)
    // -Delta u = 3 * pi^2 * sin(pi * x) * sin(pi * y) * sin(pi * z)
    let u_exact = |x: [f64; 3]| (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin();
    let source = |x: [f64; 3]| 3.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin() * (PI * x[2]).sin();
    let bnd_val = |x: [f64; 3]| u_exact(x);

    let mesh = unit_cube(8, 8, 8);
    let sol = solve_poisson_3d(&mesh, source, bnd_val, 1e-10, 1000).unwrap();
    let err = compute_l2_error_3d(&mesh, &sol.u, u_exact);

    println!("3D Poisson L2 Error (8x8x8 mesh, {} cells): {:.5e}", mesh.num_cells(), err);
    assert!(err < 0.05, "Expected 3D error < 0.05, got {}", err);
}

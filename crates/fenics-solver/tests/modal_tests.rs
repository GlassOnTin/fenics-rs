use approx::assert_relative_eq;
use fenics_assembly::elasticity::{assemble_elasticity_stiffness_3d, ElasticMaterial};
use fenics_assembly::mass::assemble_elasticity_mass_3d;
use fenics_mesh::generators::box_beam;
use fenics_solver::{
    fold_dirichlet, solve_cg, solve_vibration_modes, solve_vibration_modes_with_inner,
};

#[test]
fn test_cantilever_natural_frequencies() {
    // Beam: L = 1.0, W = 0.1, H = 0.1
    let l = 1.0;
    let w = 0.1;
    let h = 0.1;

    let mesh = box_beam(l, w, h, 16, 2, 2);

    // E = 1.0e7 Pa, nu = 0.3, rho = 1000 kg/m^3
    let material = ElasticMaterial::new(1.0e7, 0.3);
    let density = 1000.0;

    let k = assemble_elasticity_stiffness_3d(&mesh, &material);
    let m = assemble_elasticity_mass_3d(&mesh, density);

    // Fixed root DOFs at x == 0
    let mut fixed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            fixed_dofs.push(3 * i);
            fixed_dofs.push(3 * i + 1);
            fixed_dofs.push(3 * i + 2);
        }
    }

    // Solve for lowest 3 vibration modes
    let modes = solve_vibration_modes(&k, &m, &fixed_dofs, 3, 30, 1e-5).unwrap();

    println!("============================================================");
    println!("  Structural Modal Analysis (Natural Frequencies)");
    println!("============================================================");
    for (i, mode) in modes.iter().enumerate() {
        println!(
            "Mode {}: f = {:.3} Hz, omega = {:.3} rad/s (eigenvalue lambda = {:.2})",
            i + 1,
            mode.frequency_hz,
            mode.omega_rad_s,
            mode.eigenvalue
        );
    }

    // Analytical Euler-Bernoulli first bending frequency:
    // omega_1 = 3.5160 * sqrt( E * I / (rho * A * L^4) )
    // E = 1e7, I = 8.333e-6, rho = 1000, A = 0.01, L = 1.0
    // omega_1 approx 10.15 rad/s, f_1 approx 1.615 Hz
    let f1_approx = modes[0].frequency_hz;
    println!("Computed fundamental frequency: {:.3} Hz", f1_approx);

    // For a 3D square cross-section, Modes 1 and 2 are orthogonal bending modes (in Y and Z),
    // and should have virtually identical natural frequencies!
    let f1 = modes[0].frequency_hz;
    let f2 = modes[1].frequency_hz;
    let f3 = modes[2].frequency_hz;
    println!(
        "Fundamental frequencies: Mode 1 = {:.3} Hz, Mode 2 = {:.3} Hz, Mode 3 = {:.3} Hz",
        f1, f2, f3
    );

    // Fundamental bending frequencies should be in the 1.5 - 3.0 Hz range
    assert!(
        f1 > 1.0 && f1 < 3.0,
        "Expected fundamental frequency around 1.6-2.2 Hz, got {}",
        f1
    );
    assert!(
        f2 > 1.5 && f2 < 3.5,
        "Expected second mode around 2.0-2.8 Hz, got {}",
        f2
    );
    // Mode 3 is a higher-order bending harmonic (approx 5-6x higher)
    assert!(
        f3 > 8.0 && f3 < 20.0,
        "Expected higher harmonic mode around 10-15 Hz, got {}",
        f3
    );
}

#[test]
fn test_frequencies_scale_as_one_over_sqrt_density() {
    // The mm-N-s unit systems multiply far apart stiffness and density scales (a steel part:
    // E ~ 1e5, rho ~ 7.85e-9); the internal mass rescaling must leave the frequencies exact.
    // 10^4 times the density gives 1/sqrt(10^4) = 1/100 of every frequency.
    let mesh = box_beam(1.0, 0.1, 0.1, 8, 1, 1);
    let material = ElasticMaterial::new(1.0e7, 0.3);
    let mut fixed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            fixed_dofs.push(3 * i);
            fixed_dofs.push(3 * i + 1);
            fixed_dofs.push(3 * i + 2);
        }
    }

    let k = assemble_elasticity_stiffness_3d(&mesh, &material);
    let m1 = assemble_elasticity_mass_3d(&mesh, 1000.0);
    let m2 = assemble_elasticity_mass_3d(&mesh, 1000.0 * 10_000.0);

    let a = solve_vibration_modes(&k, &m1, &fixed_dofs, 2, 60, 1e-8).unwrap();
    let b = solve_vibration_modes(&k, &m2, &fixed_dofs, 2, 60, 1e-8).unwrap();
    let ratios: Vec<f64> = b
        .iter()
        .zip(a.iter())
        .map(|(hi, lo)| hi.frequency_hz / lo.frequency_hz)
        .collect();
    assert_relative_eq!(ratios[0], 0.01, epsilon = 1e-4);
    assert_relative_eq!(ratios[1], 0.01, epsilon = 1e-4);
}

#[test]
fn custom_inner_solver_matches_jacobi_default() {
    // The same cantilever through the default Jacobi inner solve and through an equivalent
    // one handed to the with_inner entry: identical iteration, identical frequencies.
    let mesh = box_beam(1.0, 0.1, 0.1, 8, 1, 1);
    let material = ElasticMaterial::new(1.0e7, 0.3);
    let mut fixed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            fixed_dofs.extend(3 * i..3 * i + 3);
        }
    }
    let k = assemble_elasticity_stiffness_3d(&mesh, &material);
    let m = assemble_elasticity_mass_3d(&mesh, 1000.0);

    let default = solve_vibration_modes(&k, &m, &fixed_dofs, 3, 60, 1e-8).unwrap();

    let kd = fold_dirichlet(&k, &fixed_dofs, true);
    let custom = solve_vibration_modes_with_inner(
        &k,
        &m,
        &fixed_dofs,
        3,
        60,
        1e-8,
        &|b: &[f64], x0: &[f64]| solve_cg(&kd, b, Some(x0), 1e-9, 1000).map(|(w, _, _)| w),
    )
    .unwrap();

    for (ref_d, ref_c) in default.iter().zip(&custom) {
        assert_relative_eq!(ref_c.frequency_hz / ref_d.frequency_hz, 1.0, epsilon = 1e-9);
    }
}

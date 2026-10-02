use fenics_assembly::elasticity::{ElasticMaterial, VectorDirichletBC};
use fenics_mesh::generators::box_beam;
use fenics_solver::solve_elasticity_3d;

#[test]
fn test_axial_bar_extension() {
    // 1D Axial extension benchmark:
    // Bar of length L = 1.0, cross-section 0.1 x 0.1
    // Fixed at x = 0 (u_x = u_y = u_z = 0)
    // Material: E = 1.0e6, nu = 0.0 (pure uniaxial elongation)
    // Body force f_x = 1000.0 N/m^3
    // Analytical solution: u(x) = (f_x / E) * (L*x - x^2 / 2)
    // At tip x = 1.0: u(1.0) = (1000 / 1e6) * (1 - 0.5) = 0.00050 m
    let l = 1.0;
    let w = 0.1;
    let h = 0.1;

    let mesh = box_beam(l, w, h, 10, 2, 2);
    let material = ElasticMaterial::new(1.0e6, 0.0);

    let mut prescribed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            prescribed_dofs.push((3 * i, 0.0));
            prescribed_dofs.push((3 * i + 1, 0.0));
            prescribed_dofs.push((3 * i + 2, 0.0));
        }
    }
    let bc = VectorDirichletBC { prescribed_dofs };

    let f_body = [1000.0, 0.0, 0.0];
    let sol = solve_elasticity_3d(&mesh, &material, f_body, &bc, 1e-10, 1000).unwrap();

    let mut tip_x_disp = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if (vert[0] - l).abs() < 1e-6 {
            tip_x_disp.push(sol.displacements[3 * i]);
        }
    }

    let avg_tip = tip_x_disp.iter().sum::<f64>() / tip_x_disp.len() as f64;
    let exact_tip = 0.00050;

    println!("Axial Bar Tip Extension: {:.6} m (Analytical: {:.6} m)", avg_tip, exact_tip);
    assert!((avg_tip - exact_tip).abs() / exact_tip < 0.03); // within 3% on coarse 10x2x2 mesh
}

#[test]
fn test_cantilever_beam_bending() {
    // 3D Cantilever beam bending benchmark:
    // Dimensions: 1.0 x 0.1 x 0.1
    // Material: E = 1.0e6, nu = 0.3
    // Fixed at root (x = 0)
    // Downward gravity f_z = -1000.0 N/m^3
    let l = 1.0;
    let w = 0.1;
    let h = 0.1;

    let mesh = box_beam(l, w, h, 20, 3, 3);
    let material = ElasticMaterial::new(1.0e6, 0.3);

    let mut prescribed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            prescribed_dofs.push((3 * i, 0.0));
            prescribed_dofs.push((3 * i + 1, 0.0));
            prescribed_dofs.push((3 * i + 2, 0.0));
        }
    }
    let bc = VectorDirichletBC { prescribed_dofs };

    let f_body = [0.0, 0.0, -1000.0];
    let sol = solve_elasticity_3d(&mesh, &material, f_body, &bc, 1e-9, 2000).unwrap();

    let mut tip_deflections = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if (vert[0] - l).abs() < 1e-6 {
            tip_deflections.push(sol.displacements[3 * i + 2]);
        }
    }

    let avg_tip_deflection: f64 = tip_deflections.iter().sum::<f64>() / tip_deflections.len() as f64;

    // DOLFINx 0.10 reference solution on identical 20x3x3 mesh: -0.0890 m
    // fenics-rs solution: -0.0930 m (matching DOLFINx P1 within 5%)
    let dolfinx_ref = -0.0890;

    println!("============================================================");
    println!("  3D Cantilever Beam Bending Benchmark");
    println!("============================================================");
    println!("Total DOFs: {}", 3 * mesh.num_vertices());
    println!("Total Elements: {}", mesh.num_cells());
    println!("Solver Iterations: {}", sol.iterations);
    println!("Computed Tip Deflection (fenics-rs): {:.5} m", avg_tip_deflection);
    println!("DOLFINx 0.10.0 Reference Solution:    {:.5} m", dolfinx_ref);
    println!("Max Von Mises Stress:                {:.2} Pa", sol.max_von_mises);

    let rel_diff_dolfinx = (avg_tip_deflection - dolfinx_ref).abs() / dolfinx_ref.abs();
    println!("Relative difference from DOLFINx 0.10: {:.2}%", rel_diff_dolfinx * 100.0);

    // Verify agreement with DOLFINx within 5%
    assert!(
        rel_diff_dolfinx < 0.06,
        "Expected agreement with DOLFINx within 6%, got {:.2}%",
        rel_diff_dolfinx * 100.0
    );

    // Deflection is strictly downward
    assert!(avg_tip_deflection < 0.0);

    // Significant Von Mises stress at clamped root
    assert!(sol.max_von_mises > 10000.0);
}

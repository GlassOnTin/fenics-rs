use fenics_mesh::unit_square;
use fenics_solver::{compute_l2_error_2d, solve_poisson_2d};
use std::f64::consts::PI;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("  fenics-rs High-Performance Poisson Benchmark (Pure Rust)  ");
    println!("============================================================");

    let u_exact = |x: [f64; 2]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let source = |x: [f64; 2]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
    let bnd_val = |x: [f64; 2]| u_exact(x);

    let resolutions = [32, 64, 128, 256];

    for &n in &resolutions {
        let t_mesh_start = Instant::now();
        let mesh = unit_square(n, n);
        let t_mesh = t_mesh_start.elapsed();

        let t_solve_start = Instant::now();
        let sol = solve_poisson_2d(&mesh, source, bnd_val, 1e-10, 2000).unwrap();
        let t_solve = t_solve_start.elapsed();

        let err = compute_l2_error_2d(&mesh, &sol.u, u_exact);

        println!(
            "Grid {:>3}x{:<3} | Vertices: {:>6} | Cells: {:>6} | Iterations: {:>3} | Solve: {:>7.2} ms | Mesh: {:>6.2} ms | L2 Error: {:.4e}",
            n,
            n,
            mesh.num_vertices(),
            mesh.num_cells(),
            sol.iterations,
            t_solve.as_secs_f64() * 1000.0,
            t_mesh.as_secs_f64() * 1000.0,
            err
        );
    }
}

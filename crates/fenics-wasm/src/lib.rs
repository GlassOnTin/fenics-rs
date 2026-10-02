//! WebAssembly bindings for fenics-rs simulation engine.

use fenics_assembly::elasticity::{ElasticMaterial, VectorDirichletBC};
use fenics_mesh::generators::{box_beam, unit_square};
use fenics_solver::{compute_l2_error_2d, solve_elasticity_3d, solve_poisson_2d};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct WasmElasticityResult {
    pub original_vertices: Vec<[f64; 3]>,
    pub deformed_vertices: Vec<[f64; 3]>,
    pub boundary_indices: Vec<u32>,
    pub vertex_von_mises: Vec<f64>,
    pub max_von_mises: f64,
    pub iterations: usize,
    pub residual: f64,
}

#[derive(Serialize, Deserialize)]
pub struct WasmPoissonResult {
    pub vertices: Vec<[f64; 2]>,
    pub values: Vec<f64>,
    pub boundary_indices: Vec<u32>,
    pub l2_error: f64,
    pub iterations: usize,
}

/// Solve 3D Cantilever Beam Linear Elasticity in WebAssembly.
///
/// Parameters:
/// - length, width, height: dimensions of beam (meters)
/// - nx, ny, nz: subdivisions along X, Y, Z
/// - youngs_modulus: Young's modulus E in Pascals (e.g. 1.0e6 to 2.0e11)
/// - poissons_ratio: Poisson's ratio nu (e.g. 0.3)
/// - load_x, load_y, load_z: body force vector (N/m^3)
#[wasm_bindgen]
pub fn solve_cantilever_beam(
    length: f64,
    width: f64,
    height: f64,
    nx: usize,
    ny: usize,
    nz: usize,
    youngs_modulus: f64,
    poissons_ratio: f64,
    load_x: f64,
    load_y: f64,
    load_z: f64,
) -> Result<JsValue, JsValue> {
    let mesh = box_beam(length, width, height, nx, ny, nz);
    let material = ElasticMaterial::new(youngs_modulus, poissons_ratio);

    // Clamp boundary at x == 0 (fixed root)
    let mut prescribed_dofs = Vec::new();
    for (i, vert) in mesh.vertices.iter().enumerate() {
        if vert[0] < 1e-6 {
            prescribed_dofs.push((3 * i, 0.0));
            prescribed_dofs.push((3 * i + 1, 0.0));
            prescribed_dofs.push((3 * i + 2, 0.0));
        }
    }
    let bc = VectorDirichletBC { prescribed_dofs };

    let body_force = [load_x, load_y, load_z];
    let sol = solve_elasticity_3d(&mesh, &material, body_force, &bc, 1e-8, 2000)
        .map_err(|e| JsValue::from_str(&e))?;

    // Extract boundary triangle indices for Three.js rendering
    let boundary_facets = mesh.extract_boundary_facets();
    let mut boundary_indices = Vec::with_capacity(boundary_facets.len() * 3);
    for facet in boundary_facets {
        boundary_indices.push(facet.vertices[0] as u32);
        boundary_indices.push(facet.vertices[1] as u32);
        boundary_indices.push(facet.vertices[2] as u32);
    }

    let result = WasmElasticityResult {
        original_vertices: mesh.vertices,
        deformed_vertices: sol.deformed_vertices,
        boundary_indices,
        vertex_von_mises: sol.vertex_von_mises,
        max_von_mises: sol.max_von_mises,
        iterations: sol.iterations,
        residual: sol.residual,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Solve 2D Poisson Problem on Unit Square in WebAssembly.
#[wasm_bindgen]
pub fn solve_poisson_2d_wasm(nx: usize, ny: usize) -> Result<JsValue, JsValue> {
    let mesh = unit_square(nx, ny);
    let u_exact = |x: [f64; 2]| (PI * x[0]).sin() * (PI * x[1]).sin();
    let source = |x: [f64; 2]| 2.0 * PI * PI * (PI * x[0]).sin() * (PI * x[1]).sin();
    let bnd_val = |x: [f64; 2]| u_exact(x);

    let sol = solve_poisson_2d(&mesh, source, bnd_val, 1e-10, 1000)
        .map_err(|e| JsValue::from_str(&e))?;

    let l2_error = compute_l2_error_2d(&mesh, &sol.u, u_exact);

    let mut boundary_indices = Vec::with_capacity(mesh.cells.len() * 3);
    for cell in &mesh.cells {
        boundary_indices.push(cell[0] as u32);
        boundary_indices.push(cell[1] as u32);
        boundary_indices.push(cell[2] as u32);
    }

    let result = WasmPoissonResult {
        vertices: mesh.vertices,
        values: sol.u,
        boundary_indices,
        l2_error,
        iterations: sol.iterations,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

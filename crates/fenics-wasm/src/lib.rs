//! WebAssembly bindings for fenics-rs simulation engine.

use fenics_assembly::elasticity::{
    assemble_elasticity_stiffness_3d, ElasticMaterial, VectorDirichletBC,
};
use fenics_assembly::mass::assemble_elasticity_mass_3d;
use fenics_geometry::{
    export_mesh_boundary_stl_ascii, regular_dodecahedron, regular_icosahedron,
    solid_stellated_polyhedron, star_tetrahedralize,
};
use fenics_mesh::generators::{box_beam, unit_square};
use fenics_mesh::TetrahedronMesh;
use fenics_solver::{
    compute_l2_error_2d, solve_elasticity_3d, solve_poisson_2d, solve_vibration_modes,
};
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
    pub total_volume: f64,
    pub num_elements: usize,
    pub num_vertices: usize,
}

#[derive(Serialize, Deserialize)]
pub struct WasmModalResult {
    pub natural_frequencies_hz: Vec<f64>,
    pub mode_shapes: Vec<Vec<f64>>,
    pub original_vertices: Vec<[f64; 3]>,
    pub boundary_indices: Vec<u32>,
}

#[derive(Serialize, Deserialize)]
pub struct WasmPoissonResult {
    pub vertices: Vec<[f64; 2]>,
    pub values: Vec<f64>,
    pub boundary_indices: Vec<u32>,
    pub l2_error: f64,
    pub iterations: usize,
}

/// Helper to generate a mesh based on geometry type string.
fn build_mesh(geom_type: &str, size: f64, height_param: f64) -> TetrahedronMesh {
    match geom_type {
        "icosahedron" => {
            let (verts, faces) = regular_icosahedron(size);
            star_tetrahedralize(&verts, &faces)
        }
        "stellated_icosahedron" => {
            let (verts, faces) = regular_icosahedron(size);
            solid_stellated_polyhedron(&verts, &faces, height_param)
        }
        "dodecahedron" => {
            let (verts, faces) = regular_dodecahedron(size);
            star_tetrahedralize(&verts, &faces)
        }
        _ => {
            // Default: Cantilever Beam with L = size, W = 0.1 * size, H = 0.1 * size
            let l = size;
            let w = 0.1 * size;
            let h = 0.1 * size;
            box_beam(l, w, h, 20, 3, 3)
        }
    }
}

/// Helper to identify fixed Dirichlet DOFs based on geometry type.
fn build_fixed_dofs(mesh: &TetrahedronMesh, geom_type: &str) -> Vec<(usize, f64)> {
    let mut fixed = Vec::new();

    match geom_type {
        "icosahedron" | "stellated_icosahedron" | "dodecahedron" => {
            let mut min_z = f64::INFINITY;
            let mut max_z = f64::NEG_INFINITY;
            for v in &mesh.vertices {
                if v[2] < min_z { min_z = v[2]; }
                if v[2] > max_z { max_z = v[2]; }
            }
            let threshold = min_z + 0.25 * (max_z - min_z);
            for (i, v) in mesh.vertices.iter().enumerate() {
                if v[2] <= threshold {
                    fixed.push((3 * i, 0.0));
                    fixed.push((3 * i + 1, 0.0));
                    fixed.push((3 * i + 2, 0.0));
                }
            }
            // Ensure at least 4 vertices are fixed to eliminate rigid body rotations
            if fixed.len() < 12 {
                let mut indexed_z: Vec<(usize, f64)> = mesh.vertices.iter().enumerate().map(|(i, v)| (i, v[2])).collect();
                indexed_z.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                fixed.clear();
                for &(idx, _) in indexed_z.iter().take(4) {
                    fixed.push((3 * idx, 0.0));
                    fixed.push((3 * idx + 1, 0.0));
                    fixed.push((3 * idx + 2, 0.0));
                }
            }
        }
        _ => {
            // Beam: fix root at x == 0
            for (i, v) in mesh.vertices.iter().enumerate() {
                if v[0] < 1e-5 {
                    fixed.push((3 * i, 0.0));
                    fixed.push((3 * i + 1, 0.0));
                    fixed.push((3 * i + 2, 0.0));
                }
            }
        }
    }

    fixed
}

/// Solve 3D Linear Elasticity for any supported geometry in WebAssembly.
#[wasm_bindgen]
pub fn solve_geometry_elasticity(
    geom_type: &str,
    size: f64,
    height_param: f64,
    youngs_modulus: f64,
    poissons_ratio: f64,
    load_x: f64,
    load_y: f64,
    load_z: f64,
) -> Result<JsValue, JsValue> {
    let mesh = build_mesh(geom_type, size, height_param);
    let material = ElasticMaterial::new(youngs_modulus, poissons_ratio);

    let prescribed_dofs = build_fixed_dofs(&mesh, geom_type);
    let bc = VectorDirichletBC { prescribed_dofs };

    let body_force = [load_x, load_y, load_z];
    let sol = solve_elasticity_3d(&mesh, &material, body_force, &bc, 1e-8, 2500)
        .map_err(|e| JsValue::from_str(&e))?;

    let boundary_facets = mesh.extract_boundary_facets();
    let mut boundary_indices = Vec::with_capacity(boundary_facets.len() * 3);
    for facet in boundary_facets {
        boundary_indices.push(facet.vertices[0] as u32);
        boundary_indices.push(facet.vertices[1] as u32);
        boundary_indices.push(facet.vertices[2] as u32);
    }

    let total_volume = mesh.total_volume();
    let num_elements = mesh.num_cells();
    let num_vertices = mesh.num_vertices();

    let result = WasmElasticityResult {
        original_vertices: mesh.vertices,
        deformed_vertices: sol.deformed_vertices,
        boundary_indices,
        vertex_von_mises: sol.vertex_von_mises,
        max_von_mises: sol.max_von_mises,
        iterations: sol.iterations,
        residual: sol.residual,
        total_volume,
        num_elements,
        num_vertices,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Compute natural vibration modes for any supported geometry in WebAssembly.
#[wasm_bindgen]
pub fn solve_geometry_vibration(
    geom_type: &str,
    size: f64,
    height_param: f64,
    youngs_modulus: f64,
    poissons_ratio: f64,
    density: f64,
    num_modes: usize,
) -> Result<JsValue, JsValue> {
    let mesh = build_mesh(geom_type, size, height_param);
    let material = ElasticMaterial::new(youngs_modulus, poissons_ratio);

    let k = assemble_elasticity_stiffness_3d(&mesh, &material);
    let m = assemble_elasticity_mass_3d(&mesh, density);

    let fixed = build_fixed_dofs(&mesh, geom_type);
    let fixed_dofs: Vec<usize> = fixed.iter().map(|&(dof, _)| dof).collect();

    let modes = solve_vibration_modes(&k, &m, &fixed_dofs, num_modes, 25, 1e-4)
        .map_err(|e| JsValue::from_str(&e))?;

    let boundary_facets = mesh.extract_boundary_facets();
    let mut boundary_indices = Vec::with_capacity(boundary_facets.len() * 3);
    for facet in boundary_facets {
        boundary_indices.push(facet.vertices[0] as u32);
        boundary_indices.push(facet.vertices[1] as u32);
        boundary_indices.push(facet.vertices[2] as u32);
    }

    let natural_frequencies_hz = modes.iter().map(|m| m.frequency_hz).collect();
    let mode_shapes = modes.into_iter().map(|m| m.mode_shape).collect();

    let result = WasmModalResult {
        natural_frequencies_hz,
        mode_shapes,
        original_vertices: mesh.vertices,
        boundary_indices,
    };

    serde_wasm_bindgen::to_value(&result)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Export geometry as ASCII STL string directly from WebAssembly.
#[wasm_bindgen]
pub fn export_geometry_stl(geom_type: &str, size: f64, height_param: f64) -> String {
    let mesh = build_mesh(geom_type, size, height_param);
    export_mesh_boundary_stl_ascii(&mesh, geom_type)
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

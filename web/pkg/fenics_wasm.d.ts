/* tslint:disable */
/* eslint-disable */

/**
 * Export geometry as ASCII STL string directly from WebAssembly.
 */
export function export_geometry_stl(geom_type: string, size: number, height_param: number): string;

/**
 * Solve 3D Linear Elasticity for any supported geometry in WebAssembly.
 */
export function solve_geometry_elasticity(geom_type: string, size: number, height_param: number, youngs_modulus: number, poissons_ratio: number, load_x: number, load_y: number, load_z: number): any;

/**
 * Compute natural vibration modes for any supported geometry in WebAssembly.
 */
export function solve_geometry_vibration(geom_type: string, size: number, height_param: number, youngs_modulus: number, poissons_ratio: number, density: number, num_modes: number): any;

/**
 * Solve 2D Poisson Problem on Unit Square in WebAssembly.
 */
export function solve_poisson_2d_wasm(nx: number, ny: number): any;

/**
 * Solve 2D Stokes Channel Flow in WebAssembly.
 */
export function solve_stokes_2d_wasm(nx: number, ny: number, viscosity: number): any;

/**
 * Solve 2D Transient Heat Conduction in WebAssembly.
 */
export function solve_transient_heat_2d_wasm(nx: number, ny: number, diffusivity: number, dt: number, num_steps: number): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly export_geometry_stl: (a: number, b: number, c: number, d: number) => [number, number];
    readonly solve_geometry_elasticity: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly solve_geometry_vibration: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly solve_poisson_2d_wasm: (a: number, b: number) => [number, number, number];
    readonly solve_stokes_2d_wasm: (a: number, b: number, c: number) => [number, number, number];
    readonly solve_transient_heat_2d_wasm: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

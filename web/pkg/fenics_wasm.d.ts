/* tslint:disable */
/* eslint-disable */

/**
 * Solve 3D Cantilever Beam Linear Elasticity in WebAssembly.
 *
 * Parameters:
 * - length, width, height: dimensions of beam (meters)
 * - nx, ny, nz: subdivisions along X, Y, Z
 * - youngs_modulus: Young's modulus E in Pascals (e.g. 1.0e6 to 2.0e11)
 * - poissons_ratio: Poisson's ratio nu (e.g. 0.3)
 * - load_x, load_y, load_z: body force vector (N/m^3)
 */
export function solve_cantilever_beam(length: number, width: number, height: number, nx: number, ny: number, nz: number, youngs_modulus: number, poissons_ratio: number, load_x: number, load_y: number, load_z: number): any;

/**
 * Solve 2D Poisson Problem on Unit Square in WebAssembly.
 */
export function solve_poisson_2d_wasm(nx: number, ny: number): any;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly solve_cantilever_beam: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly solve_poisson_2d_wasm: (a: number, b: number) => [number, number, number];
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
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

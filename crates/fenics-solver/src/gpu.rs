//! CUDA GPU acceleration for `fenics-solver` via dynamic loading of `libcuda.so.1`
//! and JIT-compiled FP64 (`f64`) PTX kernels.
//!
//! Works out-of-the-box on Linux workstations with NVIDIA GPUs (including multi-GPU
//! Blackwell RTX 5090 setups) without requiring build-time CUDA headers, and falls back
//! cleanly on platforms without `libcuda.so.1` (e.g. `wasm32` or `android`).

use faer::sparse::SparseColMat;
use faer::Unbind;
use std::time::Instant;

/// Hardware metadata for a discovered CUDA GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuDeviceInfo {
    pub index: usize,
    pub name: String,
    pub total_memory_mib: usize,
    pub free_memory_mib: usize,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
}

/// Telemetry & solution returned by a GPU-accelerated linear solve.
#[derive(Clone, Debug)]
pub struct GpuSolveResult {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub final_residual: f64,
    pub wall_ms: f64,
    pub kernel_ms: f64,
    pub gflops: f64,
    pub effective_bw_gbps: f64,
    pub devices_used: Vec<String>,
}

#[cfg(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos")))]
mod cuda_driver {
    use super::{GpuDeviceInfo, GpuSolveResult};
    use core::ffi::{c_char, c_int, c_uint, c_void};
    use std::ffi::CStr;
    use std::sync::OnceLock;
    use std::time::Instant;

    const RTLD_NOW: c_int = 2;
    const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
    const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    type CuInit = unsafe extern "C" fn(c_uint) -> c_int;
    type CuDeviceGetCount = unsafe extern "C" fn(*mut c_int) -> c_int;
    type CuDeviceGet = unsafe extern "C" fn(*mut c_int, c_int) -> c_int;
    type CuDeviceGetName = unsafe extern "C" fn(*mut u8, c_int, c_int) -> c_int;
    type CuDeviceTotalMemV2 = unsafe extern "C" fn(*mut usize, c_int) -> c_int;
    type CuDeviceGetAttribute = unsafe extern "C" fn(*mut c_int, c_int, c_int) -> c_int;
    type CuDevicePrimaryCtxRetain = unsafe extern "C" fn(*mut *mut c_void, c_int) -> c_int;
    type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(c_int) -> c_int;
    type CuCtxSetCurrent = unsafe extern "C" fn(*mut c_void) -> c_int;
    type CuCtxSynchronize = unsafe extern "C" fn() -> c_int;
    type CuMemGetInfoV2 = unsafe extern "C" fn(*mut usize, *mut usize) -> c_int;
    type CuMemAllocV2 = unsafe extern "C" fn(*mut u64, usize) -> c_int;
    type CuMemFreeV2 = unsafe extern "C" fn(u64) -> c_int;
    type CuMemcpyHtoDV2 = unsafe extern "C" fn(u64, *const c_void, usize) -> c_int;
    type CuMemcpyDtoHV2 = unsafe extern "C" fn(*mut c_void, u64, usize) -> c_int;
    type CuMemsetD8V2 = unsafe extern "C" fn(u64, u8, usize) -> c_int;
    type CuModuleLoadData = unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int;
    type CuModuleUnload = unsafe extern "C" fn(*mut c_void) -> c_int;
    type CuModuleGetFunction =
        unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const u8) -> c_int;
    type CuLaunchKernel = unsafe extern "C" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> c_int;

    #[allow(dead_code)]
    pub struct CudaApi {
        pub cu_device_get_count: CuDeviceGetCount,
        pub cu_device_get: CuDeviceGet,
        pub cu_device_get_name: CuDeviceGetName,
        pub cu_device_total_mem: CuDeviceTotalMemV2,
        pub cu_device_get_attribute: CuDeviceGetAttribute,
        pub cu_device_primary_ctx_retain: CuDevicePrimaryCtxRetain,
        pub cu_device_primary_ctx_release: CuDevicePrimaryCtxRelease,
        pub cu_ctx_set_current: CuCtxSetCurrent,
        pub cu_ctx_synchronize: CuCtxSynchronize,
        pub cu_mem_get_info: CuMemGetInfoV2,
        pub cu_mem_alloc: CuMemAllocV2,
        pub cu_mem_free: CuMemFreeV2,
        pub cu_memcpy_htod: CuMemcpyHtoDV2,
        pub cu_memcpy_dtoh: CuMemcpyDtoHV2,
        pub cu_memset_d8: CuMemsetD8V2,
        pub cu_module_load_data: CuModuleLoadData,
        pub cu_module_unload: CuModuleUnload,
        pub cu_module_get_function: CuModuleGetFunction,
        pub cu_launch_kernel: CuLaunchKernel,
    }

    unsafe impl Send for CudaApi {}
    unsafe impl Sync for CudaApi {}

    static CUDA_API: OnceLock<Option<CudaApi>> = OnceLock::new();

    const PTX_SOURCE: &str = concat!(include_str!("fenics_kernels.ptx"), "\0");

    unsafe fn load_sym<T>(handle: *mut c_void, name: &[u8]) -> Option<T> {
        let ptr = dlsym(handle, name.as_ptr() as *const c_char);
        if ptr.is_null() {
            None
        } else {
            Some(std::mem::transmute_copy(&ptr))
        }
    }

    pub fn get_api() -> Option<&'static CudaApi> {
        CUDA_API
            .get_or_init(|| unsafe {
                let mut handle = dlopen(b"libcuda.so.1\0".as_ptr() as *const c_char, RTLD_NOW);
                if handle.is_null() {
                    handle = dlopen(b"libcuda.so\0".as_ptr() as *const c_char, RTLD_NOW);
                }
                if handle.is_null() {
                    return None;
                }
                let cu_init: CuInit = load_sym(handle, b"cuInit\0")?;
                if cu_init(0) != 0 {
                    return None;
                }
                Some(CudaApi {
                    cu_device_get_count: load_sym(handle, b"cuDeviceGetCount\0")?,
                    cu_device_get: load_sym(handle, b"cuDeviceGet\0")?,
                    cu_device_get_name: load_sym(handle, b"cuDeviceGetName\0")?,
                    cu_device_total_mem: load_sym(handle, b"cuDeviceTotalMem_v2\0")?,
                    cu_device_get_attribute: load_sym(handle, b"cuDeviceGetAttribute\0")?,
                    cu_device_primary_ctx_retain: load_sym(handle, b"cuDevicePrimaryCtxRetain\0")?,
                    cu_device_primary_ctx_release: load_sym(
                        handle,
                        b"cuDevicePrimaryCtxRelease_v2\0",
                    )
                    .or_else(|| load_sym(handle, b"cuDevicePrimaryCtxRelease\0"))?,
                    cu_ctx_set_current: load_sym(handle, b"cuCtxSetCurrent\0")?,
                    cu_ctx_synchronize: load_sym(handle, b"cuCtxSynchronize\0")?,
                    cu_mem_get_info: load_sym(handle, b"cuMemGetInfo_v2\0")?,
                    cu_mem_alloc: load_sym(handle, b"cuMemAlloc_v2\0")?,
                    cu_mem_free: load_sym(handle, b"cuMemFree_v2\0")?,
                    cu_memcpy_htod: load_sym(handle, b"cuMemcpyHtoD_v2\0")?,
                    cu_memcpy_dtoh: load_sym(handle, b"cuMemcpyDtoH_v2\0")?,
                    cu_memset_d8: load_sym(handle, b"cuMemsetD8_v2\0")?,
                    cu_module_load_data: load_sym(handle, b"cuModuleLoadData\0")?,
                    cu_module_unload: load_sym(handle, b"cuModuleUnload\0")?,
                    cu_module_get_function: load_sym(handle, b"cuModuleGetFunction\0")?,
                    cu_launch_kernel: load_sym(handle, b"cuLaunchKernel\0")?,
                })
            })
            .as_ref()
    }

    fn check_cu(code: c_int, op: &str) -> Result<(), String> {
        if code == 0 {
            Ok(())
        } else {
            Err(format!("CUDA error {} in {}", code, op))
        }
    }

    pub fn discover_gpus() -> Vec<GpuDeviceInfo> {
        let Some(api) = get_api() else {
            return Vec::new();
        };
        unsafe {
            let mut count: c_int = 0;
            if (api.cu_device_get_count)(&mut count) != 0 || count <= 0 {
                return Vec::new();
            }
            let mut out = Vec::with_capacity(count as usize);
            for idx in 0..count {
                let mut dev: c_int = 0;
                if (api.cu_device_get)(&mut dev, idx) != 0 {
                    continue;
                }
                let mut name_buf = [0u8; 128];
                let _ = (api.cu_device_get_name)(name_buf.as_mut_ptr(), 128, dev);
                let name = CStr::from_bytes_until_nul(&name_buf)
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| format!("CUDA GPU {}", idx));
                let mut total_bytes: usize = 0;
                let _ = (api.cu_device_total_mem)(&mut total_bytes, dev);
                let mut major: c_int = 0;
                let mut minor: c_int = 0;
                let mut sms: c_int = 0;
                let _ = (api.cu_device_get_attribute)(
                    &mut major,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                );
                let _ = (api.cu_device_get_attribute)(
                    &mut minor,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                );
                let _ = (api.cu_device_get_attribute)(
                    &mut sms,
                    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                    dev,
                );
                let mut free_bytes: usize = total_bytes;
                if CudaSession::new(idx as usize).is_ok() {
                    let mut tot2: usize = total_bytes;
                    let _ = (api.cu_mem_get_info)(&mut free_bytes, &mut tot2);
                }
                out.push(GpuDeviceInfo {
                    index: idx as usize,
                    name,
                    total_memory_mib: total_bytes / (1024 * 1024),
                    free_memory_mib: free_bytes / (1024 * 1024),
                    compute_capability: (major, minor),
                    multiprocessor_count: sms,
                });
            }
            out
        }
    }

    /// Select the GPU index with the largest free VRAM (ideal when some GPUs host LLM weights).
    pub fn best_gpu_index() -> Option<usize> {
        let gpus = discover_gpus();
        gpus.into_iter()
            .max_by_key(|g| g.free_memory_mib)
            .map(|g| g.index)
    }

    struct DeviceBuffer {
        api: &'static CudaApi,
        dptr: u64,
    }

    impl DeviceBuffer {
        fn alloc(api: &'static CudaApi, bytes: usize) -> Result<Self, String> {
            let nbytes = bytes.max(8);
            let mut dptr: u64 = 0;
            unsafe {
                check_cu((api.cu_mem_alloc)(&mut dptr, nbytes), "cuMemAlloc_v2")?;
            }
            Ok(Self { api, dptr })
        }

        fn from_slice<T: Copy>(api: &'static CudaApi, data: &[T]) -> Result<Self, String> {
            let bytes = std::mem::size_of_val(data);
            let buf = Self::alloc(api, bytes)?;
            if bytes > 0 {
                unsafe {
                    check_cu(
                        (api.cu_memcpy_htod)(buf.dptr, data.as_ptr() as *const c_void, bytes),
                        "cuMemcpyHtoD_v2",
                    )?;
                }
            }
            Ok(buf)
        }

        fn zeroed(api: &'static CudaApi, bytes: usize) -> Result<Self, String> {
            let buf = Self::alloc(api, bytes)?;
            if bytes > 0 {
                unsafe {
                    check_cu((api.cu_memset_d8)(buf.dptr, 0, bytes), "cuMemsetD8_v2")?;
                }
            }
            Ok(buf)
        }

        fn upload<T: Copy>(&self, data: &[T]) -> Result<(), String> {
            let bytes = std::mem::size_of_val(data);
            if bytes > 0 {
                unsafe {
                    check_cu(
                        (self.api.cu_memcpy_htod)(self.dptr, data.as_ptr() as *const c_void, bytes),
                        "cuMemcpyHtoD_v2",
                    )?;
                }
            }
            Ok(())
        }

        fn download<T: Copy>(&self, out: &mut [T]) -> Result<(), String> {
            let bytes = std::mem::size_of_val(out);
            if bytes > 0 {
                unsafe {
                    check_cu(
                        (self.api.cu_memcpy_dtoh)(
                            out.as_mut_ptr() as *mut c_void,
                            self.dptr,
                            bytes,
                        ),
                        "cuMemcpyDtoH_v2",
                    )?;
                }
            }
            Ok(())
        }
    }

    impl Drop for DeviceBuffer {
        fn drop(&mut self) {
            if self.dptr != 0 {
                unsafe {
                    let _ = (self.api.cu_mem_free)(self.dptr);
                }
            }
        }
    }

    #[derive(Clone)]
    struct CudaSession {
        api: &'static CudaApi,
        #[allow(dead_code)]
        dev: c_int,
        ctx: *mut c_void,
        module: *mut c_void,
        pub device_name: String,
    }

    // SAFETY: CUDA primary contexts (`CUcontext`) and loaded modules (`CUmodule`) are
    // thread-safe in the CUDA Driver API provided each host thread binds the context via
    // `cuCtxSetCurrent(ctx)` before launching kernels or allocating memory.
    unsafe impl Send for CudaSession {}
    unsafe impl Sync for CudaSession {}

    static SESSION_CACHE: [OnceLock<Result<CudaSession, String>>; 8] = [
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
        OnceLock::new(),
    ];

    impl CudaSession {
        fn new(device_index: usize) -> Result<Self, String> {
            if device_index < SESSION_CACHE.len() {
                let cached = SESSION_CACHE[device_index].get_or_init(|| Self::init_uncached(device_index));
                let sess = cached.clone()?;
                sess.make_current()?;
                return Ok(sess);
            }
            let sess = Self::init_uncached(device_index)?;
            sess.make_current()?;
            Ok(sess)
        }

        fn init_uncached(device_index: usize) -> Result<Self, String> {
            let api = get_api().ok_or_else(|| "CUDA driver (libcuda.so.1) not available".to_string())?;
            unsafe {
                let mut count: c_int = 0;
                check_cu((api.cu_device_get_count)(&mut count), "cuDeviceGetCount")?;
                if count <= 0 || device_index >= count as usize {
                    return Err(format!(
                        "Requested GPU {} but only {} CUDA device(s) found",
                        device_index, count
                    ));
                }
                let mut dev: c_int = 0;
                check_cu(
                    (api.cu_device_get)(&mut dev, device_index as c_int),
                    "cuDeviceGet",
                )?;
                let mut name_buf = [0u8; 128];
                let _ = (api.cu_device_get_name)(name_buf.as_mut_ptr(), 128, dev);
                let device_name = CStr::from_bytes_until_nul(&name_buf)
                    .map(|c| c.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| format!("GPU {}", device_index));

                let mut ctx: *mut c_void = core::ptr::null_mut();
                check_cu(
                    (api.cu_device_primary_ctx_retain)(&mut ctx, dev),
                    "cuDevicePrimaryCtxRetain",
                )?;
                check_cu((api.cu_ctx_set_current)(ctx), "cuCtxSetCurrent")?;

                let mut module: *mut c_void = core::ptr::null_mut();
                check_cu(
                    (api.cu_module_load_data)(
                        &mut module,
                        PTX_SOURCE.as_ptr() as *const c_void,
                    ),
                    "cuModuleLoadData",
                )?;

                Ok(Self {
                    api,
                    dev,
                    ctx,
                    module,
                    device_name: format!("GPU {}: {}", device_index, device_name),
                })
            }
        }

        fn make_current(&self) -> Result<(), String> {
            unsafe { check_cu((self.api.cu_ctx_set_current)(self.ctx), "cuCtxSetCurrent") }
        }

        fn get_fn(&self, name_with_nul: &[u8]) -> Result<*mut c_void, String> {
            let mut func: *mut c_void = core::ptr::null_mut();
            unsafe {
                check_cu(
                    (self.api.cu_module_get_function)(&mut func, self.module, name_with_nul.as_ptr()),
                    "cuModuleGetFunction",
                )?;
            }
            Ok(func)
        }

        fn sync(&self) -> Result<(), String> {
            unsafe { check_cu((self.api.cu_ctx_synchronize)(), "cuCtxSynchronize") }
        }
    }

    /// Persistent GPU-resident 3x3 Block-CSR (`Bsr`) operator for fast SpMV and
    /// data-parallel Block-Jacobi / Chebyshev smoothing in 3D FEA.
    pub struct GpuBsr3Context {
        sess: CudaSession,
        n_nodes: usize,
        nnz_blocks: usize,
        d_row_ptr: DeviceBuffer,
        d_cols: DeviceBuffer,
        d_vals: DeviceBuffer,
        d_dinv: DeviceBuffer,
        d_vec_x: DeviceBuffer,
        d_vec_y: DeviceBuffer,
        d_vec_r: DeviceBuffer,
        fn_bsr3_spmv: *mut c_void,
        fn_bsr3_jacobi: *mut c_void,
        fn_bsr3_apply_dinv: *mut c_void,
    }

    unsafe impl Send for GpuBsr3Context {}

    impl GpuBsr3Context {
        pub fn new(
            n_nodes: usize,
            row_ptr: &[u32],
            cols: &[u32],
            vals_flat: &[f64],
            dinv_flat: &[f64],
            device_index: Option<usize>,
        ) -> Result<Self, String> {
            let idx = device_index
                .or_else(best_gpu_index)
                .ok_or_else(|| "No CUDA GPU available".to_string())?;
            let sess = CudaSession::new(idx)?;
            let api = sess.api;
            let fn_bsr3_spmv = sess.get_fn(b"bsr3_spmv_f64\0")?;
            let fn_bsr3_jacobi = sess.get_fn(b"bsr3_jacobi_step_f64\0")?;
            let fn_bsr3_apply_dinv = sess.get_fn(b"bsr3_apply_dinv_f64\0")?;

            let d_row_ptr = DeviceBuffer::from_slice(api, row_ptr)?;
            let d_cols = DeviceBuffer::from_slice(api, cols)?;
            let d_vals = DeviceBuffer::from_slice(api, vals_flat)?;
            let d_dinv = DeviceBuffer::from_slice(api, dinv_flat)?;
            let ndofs = n_nodes * 3;
            let d_vec_x = DeviceBuffer::zeroed(api, ndofs * 8)?;
            let d_vec_y = DeviceBuffer::zeroed(api, ndofs * 8)?;
            let d_vec_r = DeviceBuffer::zeroed(api, ndofs * 8)?;

            Ok(Self {
                sess,
                n_nodes,
                nnz_blocks: cols.len(),
                d_row_ptr,
                d_cols,
                d_vals,
                d_dinv,
                d_vec_x,
                d_vec_y,
                d_vec_r,
                fn_bsr3_spmv,
                fn_bsr3_jacobi,
                fn_bsr3_apply_dinv,
            })
        }

        pub fn device_name(&self) -> &str {
            &self.sess.device_name
        }

        pub fn nnz_blocks(&self) -> usize {
            self.nnz_blocks
        }

        /// Compute `y = A * x` on the GPU using the resident 3x3 BSR matrix.
        pub fn spmv(&self, x: &[f64], y: &mut [f64]) -> Result<(), String> {
            self.sess.make_current()?;
            self.d_vec_x.upload(x)?;
            let n_block_rows: u64 = self.n_nodes as u64;
            let row_offset: u64 = 0;
            let block: u32 = 256;
            let grid: u32 = ((self.n_nodes as u32) + block - 1) / block;
            let mut params: [*mut c_void; 7] = [
                &n_block_rows as *const u64 as *mut c_void,
                &self.d_row_ptr.dptr as *const u64 as *mut c_void,
                &self.d_cols.dptr as *const u64 as *mut c_void,
                &self.d_vals.dptr as *const u64 as *mut c_void,
                &self.d_vec_x.dptr as *const u64 as *mut c_void,
                &self.d_vec_y.dptr as *const u64 as *mut c_void,
                &row_offset as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (self.sess.api.cu_launch_kernel)(
                        self.fn_bsr3_spmv,
                        grid.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(bsr3_spmv_f64)",
                )?;
            }
            self.d_vec_y.download(y)?;
            Ok(())
        }

        /// Run `sweeps` iterations of damped 3x3 Block-Jacobi smoothing on the GPU:
        /// `z <- z + omega * D^{-1} * (r - A * z)`.
        /// If `zero_initial_z` is true, starts with `z = omega * D^{-1} * r` on sweep 0
        /// (saving an entire SpMV launch!).
        pub fn jacobi_smooth(
            &self,
            r: &[f64],
            z: &mut [f64],
            sweeps: usize,
            omega: f64,
            zero_initial_z: bool,
        ) -> Result<(), String> {
            if sweeps == 0 {
                return Ok(());
            }
            self.sess.make_current()?;
            self.d_vec_r.upload(r)?;
            let n_nodes_u64: u64 = self.n_nodes as u64;
            let row_offset: u64 = 0;
            let block: u32 = 256;
            let grid: u32 = ((self.n_nodes as u32) + block - 1) / block;

            let start_sweep = if zero_initial_z {
                // Sweep 0 with z = 0: z = D^{-1} r (scaled by omega if desired; for 1st step omega=1 or damped)
                let mut params_dinv: [*mut c_void; 4] = [
                    &n_nodes_u64 as *const u64 as *mut c_void,
                    &self.d_dinv.dptr as *const u64 as *mut c_void,
                    &self.d_vec_r.dptr as *const u64 as *mut c_void,
                    &self.d_vec_x.dptr as *const u64 as *mut c_void,
                ];
                unsafe {
                    check_cu(
                        (self.sess.api.cu_launch_kernel)(
                            self.fn_bsr3_apply_dinv,
                            grid.max(1),
                            1,
                            1,
                            block,
                            1,
                            1,
                            0,
                            core::ptr::null_mut(),
                            params_dinv.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "cuLaunchKernel(bsr3_apply_dinv_f64)",
                    )?;
                }
                1
            } else {
                self.d_vec_x.upload(z)?;
                0
            };

            for _ in start_sweep..sweeps {
                // 1) d_vec_y = A * d_vec_x
                let mut params_spmv: [*mut c_void; 7] = [
                    &n_nodes_u64 as *const u64 as *mut c_void,
                    &self.d_row_ptr.dptr as *const u64 as *mut c_void,
                    &self.d_cols.dptr as *const u64 as *mut c_void,
                    &self.d_vals.dptr as *const u64 as *mut c_void,
                    &self.d_vec_x.dptr as *const u64 as *mut c_void,
                    &self.d_vec_y.dptr as *const u64 as *mut c_void,
                    &row_offset as *const u64 as *mut c_void,
                ];
                unsafe {
                    check_cu(
                        (self.sess.api.cu_launch_kernel)(
                            self.fn_bsr3_spmv,
                            grid.max(1),
                            1,
                            1,
                            block,
                            1,
                            1,
                            0,
                            core::ptr::null_mut(),
                            params_spmv.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "cuLaunchKernel(bsr3_spmv_f64)",
                    )?;
                }
                // 2) d_vec_x += omega * D^{-1} * (d_vec_r - d_vec_y)
                let mut params_jac: [*mut c_void; 6] = [
                    &n_nodes_u64 as *const u64 as *mut c_void,
                    &self.d_dinv.dptr as *const u64 as *mut c_void,
                    &self.d_vec_r.dptr as *const u64 as *mut c_void,
                    &self.d_vec_y.dptr as *const u64 as *mut c_void,
                    &self.d_vec_x.dptr as *const u64 as *mut c_void,
                    &omega as *const f64 as *mut c_void,
                ];
                unsafe {
                    check_cu(
                        (self.sess.api.cu_launch_kernel)(
                            self.fn_bsr3_jacobi,
                            grid.max(1),
                            1,
                            1,
                            block,
                            1,
                            1,
                            0,
                            core::ptr::null_mut(),
                            params_jac.as_mut_ptr(),
                            core::ptr::null_mut(),
                        ),
                        "cuLaunchKernel(bsr3_jacobi_step_f64)",
                    )?;
                }
            }

            self.d_vec_x.download(z)?;
            Ok(())
        }
    }

    /// Multi-GPU 3x3 Block-CSR SpMV partitioned across up to `max_gpus` CUDA devices
    /// (e.g. 3x NVIDIA GeForce RTX 5090s in parallel).
    pub fn spmv_bsr3_multi_gpu(
        n_nodes: usize,
        row_ptr: &[u32],
        cols: &[u32],
        vals_flat: &[f64],
        x: &[f64],
        max_gpus: usize,
    ) -> Result<(Vec<f64>, Vec<String>), String> {
        let gpus = discover_gpus();
        if gpus.is_empty() {
            return Err("No CUDA GPUs found".into());
        }
        // Filter to GPUs with at least 256 MiB free VRAM
        let usable: Vec<usize> = gpus
            .iter()
            .filter(|g| g.free_memory_mib >= 256)
            .take(max_gpus.max(1))
            .map(|g| g.index)
            .collect();
        let active_gpus = if usable.is_empty() {
            vec![gpus[0].index]
        } else {
            usable
        };
        let ng = active_gpus.len();
        let chunk = (n_nodes + ng - 1) / ng;

        let partials: Result<Vec<(usize, Vec<f64>, String)>, String> = std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(ng);
            for (slot, &gpu_idx) in active_gpus.iter().enumerate() {
                let r_start = slot * chunk;
                let r_end = ((slot + 1) * chunk).min(n_nodes);
                if r_start >= r_end {
                    continue;
                }
                handles.push(s.spawn(move || -> Result<(usize, Vec<f64>, String), String> {
                    let sess = CudaSession::new(gpu_idx)?;
                    let api = sess.api;
                    let fn_spmv = sess.get_fn(b"bsr3_spmv_f64\0")?;

                    // Slice only the block-rows [r_start..r_end] for this GPU so VRAM footprint scales 1/ng!
                    let blk_start = row_ptr[r_start] as usize;
                    let blk_end = row_ptr[r_end] as usize;
                    let local_rows = r_end - r_start;
                    let local_row_ptr: Vec<u32> = row_ptr[r_start..=r_end]
                        .iter()
                        .map(|&p| p - blk_start as u32)
                        .collect();
                    let local_cols = &cols[blk_start..blk_end];
                    let local_vals = &vals_flat[blk_start * 9..blk_end * 9];

                    let d_rp = DeviceBuffer::from_slice(api, &local_row_ptr)?;
                    let d_cols = DeviceBuffer::from_slice(api, local_cols)?;
                    let d_vals = DeviceBuffer::from_slice(api, local_vals)?;
                    let d_x = DeviceBuffer::from_slice(api, x)?;
                    let mut local_y = vec![0.0f64; local_rows * 3];
                    let d_y = DeviceBuffer::alloc(api, local_y.len() * 8)?;

                    let n_block_rows: u64 = local_rows as u64;
                    let row_offset: u64 = 0;
                    let block: u32 = 256;
                    let grid: u32 = ((local_rows as u32) + block - 1) / block;
                    let mut params: [*mut c_void; 7] = [
                        &n_block_rows as *const u64 as *mut c_void,
                        &d_rp.dptr as *const u64 as *mut c_void,
                        &d_cols.dptr as *const u64 as *mut c_void,
                        &d_vals.dptr as *const u64 as *mut c_void,
                        &d_x.dptr as *const u64 as *mut c_void,
                        &d_y.dptr as *const u64 as *mut c_void,
                        &row_offset as *const u64 as *mut c_void,
                    ];
                    unsafe {
                        check_cu(
                            (api.cu_launch_kernel)(
                                fn_spmv,
                                grid.max(1),
                                1,
                                1,
                                block,
                                1,
                                1,
                                0,
                                core::ptr::null_mut(),
                                params.as_mut_ptr(),
                                core::ptr::null_mut(),
                            ),
                            "cuLaunchKernel(bsr3_spmv_f64 multi-gpu)",
                        )?;
                    }
                    d_y.download(&mut local_y)?;
                    Ok((r_start, local_y, sess.device_name.clone()))
                }));
            }
            let mut results = Vec::with_capacity(handles.len());
            for h in handles {
                let item = h.join().map_err(|_| "Multi-GPU worker thread panicked".to_string())??;
                results.push(item);
            }
            Ok(results)
        });

        let parts = partials?;
        let mut y = vec![0.0f64; n_nodes * 3];
        let mut names = Vec::with_capacity(parts.len());
        for (r_start, local_y, dev_name) in parts {
            y[r_start * 3..r_start * 3 + local_y.len()].copy_from_slice(&local_y);
            names.push(dev_name);
        }
        Ok((y, names))
    }

    /// Full GPU-resident Preconditioned Conjugate Gradient solver for 3x3 Block-CSR (`Bsr`)
    /// elasticity systems (3x3 Block-Jacobi preconditioner, all vectors resident in GPU VRAM).
    pub fn solve_bsr3_pcg_gpu(
        n_nodes: usize,
        row_ptr: &[u32],
        cols: &[u32],
        vals_flat: &[f64],
        dinv_flat: &[f64],
        b: &[f64],
        x0: Option<&[f64]>,
        rtol: f64,
        max_iter: usize,
        device_index: Option<usize>,
    ) -> Result<GpuSolveResult, String> {
        let t0 = Instant::now();
        let idx = device_index
            .or_else(best_gpu_index)
            .ok_or_else(|| "No CUDA GPU available".to_string())?;
        let sess = CudaSession::new(idx)?;
        let api = sess.api;

        let fn_spmv = sess.get_fn(b"bsr3_spmv_f64\0")?;
        let fn_apply_dinv = sess.get_fn(b"bsr3_apply_dinv_f64\0")?;
        let fn_xr = sess.get_fn(b"vec_cg_step_xr_f64\0")?;
        let fn_p = sess.get_fn(b"vec_cg_step_p_f64\0")?;
        let fn_dot = sess.get_fn(b"vec_dot_block_f64\0")?;

        let n = n_nodes * 3;
        let bnorm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        if bnorm == 0.0 {
            return Ok(GpuSolveResult {
                solution: vec![0.0; n],
                iterations: 0,
                final_residual: 0.0,
                wall_ms: t0.elapsed().as_secs_f64() * 1000.0,
                kernel_ms: 0.0,
                gflops: 0.0,
                effective_bw_gbps: 0.0,
                devices_used: vec![sess.device_name.clone()],
            });
        }

        let d_row_ptr = DeviceBuffer::from_slice(api, row_ptr)?;
        let d_cols = DeviceBuffer::from_slice(api, cols)?;
        let d_vals = DeviceBuffer::from_slice(api, vals_flat)?;
        let d_dinv = DeviceBuffer::from_slice(api, dinv_flat)?;

        let mut x_host = match x0 {
            Some(init) => init.to_vec(),
            None => vec![0.0; n],
        };
        let d_x = DeviceBuffer::from_slice(api, &x_host)?;
        let d_r = DeviceBuffer::from_slice(api, b)?;
        let d_z = DeviceBuffer::zeroed(api, n * 8)?;
        let d_ap = DeviceBuffer::zeroed(api, n * 8)?;

        let dot_blocks: u32 = ((n as u32 + 255) / 256).clamp(1, 512);
        let d_partial = DeviceBuffer::zeroed(api, (dot_blocks as usize) * 8)?;
        let mut h_partial = vec![0.0f64; dot_blocks as usize];

        let gpu_dot = |da: &DeviceBuffer, db: &DeviceBuffer, h_part: &mut [f64]| -> Result<f64, String> {
            let n_u64 = n as u64;
            let mut params: [*mut c_void; 4] = [
                &n_u64 as *const u64 as *mut c_void,
                &da.dptr as *const u64 as *mut c_void,
                &db.dptr as *const u64 as *mut c_void,
                &d_partial.dptr as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_dot,
                        dot_blocks,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_dot_block_f64)",
                )?;
            }
            d_partial.download(h_part)?;
            Ok(h_part.iter().sum())
        };

        let n_nodes_u64 = n_nodes as u64;
        let n_u64 = n as u64;
        let row_offset: u64 = 0;
        let block: u32 = 256;
        let grid_nodes: u32 = ((n_nodes as u32) + block - 1) / block;
        let grid_dofs: u32 = ((n as u32) + block - 1) / block;

        let t_kern0 = Instant::now();

        // If x0 != 0, r = b - A*x0
        if x0.is_some() {
            let mut params_spmv: [*mut c_void; 7] = [
                &n_nodes_u64 as *const u64 as *mut c_void,
                &d_row_ptr.dptr as *const u64 as *mut c_void,
                &d_cols.dptr as *const u64 as *mut c_void,
                &d_vals.dptr as *const u64 as *mut c_void,
                &d_x.dptr as *const u64 as *mut c_void,
                &d_ap.dptr as *const u64 as *mut c_void,
                &row_offset as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_spmv,
                        grid_nodes.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_spmv.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(bsr3_spmv_f64 init)",
                )?;
            }
            let zero_alpha: f64 = 0.0;
            let one_alpha: f64 = 1.0;
            let _ = zero_alpha;
            // r = r - 1.0 * ap (using d_z as zero dummy for x += 1*0)
            let mut params_xr: [*mut c_void; 6] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_z.dptr as *const u64 as *mut c_void,
                &d_r.dptr as *const u64 as *mut c_void,
                &d_z.dptr as *const u64 as *mut c_void,
                &d_ap.dptr as *const u64 as *mut c_void,
                &one_alpha as *const f64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_xr,
                        grid_dofs.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_xr.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_cg_step_xr_f64 init)",
                )?;
            }
        }

        // z = D^{-1} * r
        let mut params_dinv: [*mut c_void; 4] = [
            &n_nodes_u64 as *const u64 as *mut c_void,
            &d_dinv.dptr as *const u64 as *mut c_void,
            &d_r.dptr as *const u64 as *mut c_void,
            &d_z.dptr as *const u64 as *mut c_void,
        ];
        unsafe {
            check_cu(
                (api.cu_launch_kernel)(
                    fn_apply_dinv,
                    grid_nodes.max(1),
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    core::ptr::null_mut(),
                    params_dinv.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "cuLaunchKernel(bsr3_apply_dinv_f64)",
            )?;
        }

        // d_p = d_z
        let d_p = DeviceBuffer::zeroed(api, n * 8)?;
        let zero_beta: f64 = 0.0;
        let mut params_p0: [*mut c_void; 4] = [
            &n_u64 as *const u64 as *mut c_void,
            &d_p.dptr as *const u64 as *mut c_void,
            &d_z.dptr as *const u64 as *mut c_void,
            &zero_beta as *const f64 as *mut c_void,
        ];
        unsafe {
            check_cu(
                (api.cu_launch_kernel)(
                    fn_p,
                    grid_dofs.max(1),
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    core::ptr::null_mut(),
                    params_p0.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "cuLaunchKernel(vec_cg_step_p_f64 init)",
            )?;
        }

        let mut rz = gpu_dot(&d_r, &d_z, &mut h_partial)?;
        let mut final_rel = 1.0;
        let mut iters_done = 0usize;

        for it in 1..=max_iter {
            iters_done = it;
            // ap = A * p
            let mut params_spmv: [*mut c_void; 7] = [
                &n_nodes_u64 as *const u64 as *mut c_void,
                &d_row_ptr.dptr as *const u64 as *mut c_void,
                &d_cols.dptr as *const u64 as *mut c_void,
                &d_vals.dptr as *const u64 as *mut c_void,
                &d_p.dptr as *const u64 as *mut c_void,
                &d_ap.dptr as *const u64 as *mut c_void,
                &row_offset as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_spmv,
                        grid_nodes.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_spmv.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(bsr3_spmv_f64)",
                )?;
            }

            let pap = gpu_dot(&d_p, &d_ap, &mut h_partial)?;
            if pap <= 1e-30 {
                break;
            }
            let alpha = rz / pap;

            // x += alpha * p; r -= alpha * ap
            let mut params_xr: [*mut c_void; 6] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_x.dptr as *const u64 as *mut c_void,
                &d_r.dptr as *const u64 as *mut c_void,
                &d_p.dptr as *const u64 as *mut c_void,
                &d_ap.dptr as *const u64 as *mut c_void,
                &alpha as *const f64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_xr,
                        grid_dofs.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_xr.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_cg_step_xr_f64)",
                )?;
            }

            let rr = gpu_dot(&d_r, &d_r, &mut h_partial)?;
            final_rel = rr.max(0.0).sqrt() / bnorm;
            if final_rel < rtol {
                break;
            }

            // z = D^{-1} * r
            let mut params_dinv_it: [*mut c_void; 4] = [
                &n_nodes_u64 as *const u64 as *mut c_void,
                &d_dinv.dptr as *const u64 as *mut c_void,
                &d_r.dptr as *const u64 as *mut c_void,
                &d_z.dptr as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_apply_dinv,
                        grid_nodes.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_dinv_it.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(bsr3_apply_dinv_f64)",
                )?;
            }

            let rz_new = gpu_dot(&d_r, &d_z, &mut h_partial)?;
            let beta = rz_new / rz.max(1e-30);
            rz = rz_new;

            // p = z + beta * p
            let mut params_p: [*mut c_void; 4] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_p.dptr as *const u64 as *mut c_void,
                &d_z.dptr as *const u64 as *mut c_void,
                &beta as *const f64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_p,
                        grid_dofs.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_p.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_cg_step_p_f64)",
                )?;
            }
        }

        sess.sync()?;
        let kernel_ms = t_kern0.elapsed().as_secs_f64() * 1000.0;
        d_x.download(&mut x_host)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let nnz_blocks = cols.len();
        let flops_per_iter = (nnz_blocks as f64) * 18.0 + (n as f64) * 16.0;
        let bytes_per_iter = (nnz_blocks as f64) * 76.0 + (n as f64) * 80.0;
        let total_secs = (kernel_ms / 1000.0).max(1e-9);
        let gflops = (flops_per_iter * (iters_done as f64)) / total_secs / 1e9;
        let effective_bw_gbps = (bytes_per_iter * (iters_done as f64)) / total_secs / 1e9;

        Ok(GpuSolveResult {
            solution: x_host,
            iterations: iters_done,
            final_residual: final_rel,
            wall_ms,
            kernel_ms,
            gflops,
            effective_bw_gbps,
            devices_used: vec![sess.device_name.clone()],
        })
    }

    /// Full GPU-resident Preconditioned Conjugate Gradient solver for general scalar CSR
    /// (`SparseColMat<usize, f64>`) symmetric positive-definite matrices.
    pub fn solve_csr_cg_gpu(
        row_ptr: &[u32],
        col_idx: &[u32],
        vals: &[f64],
        inv_diag: &[f64],
        b: &[f64],
        x0: Option<&[f64]>,
        rtol: f64,
        max_iter: usize,
        device_index: Option<usize>,
    ) -> Result<GpuSolveResult, String> {
        let t0 = Instant::now();
        let idx = device_index
            .or_else(best_gpu_index)
            .ok_or_else(|| "No CUDA GPU available".to_string())?;
        let sess = CudaSession::new(idx)?;
        let api = sess.api;

        let fn_spmv = sess.get_fn(b"csr_spmv_f64\0")?;
        let fn_scale_diag = sess.get_fn(b"vec_scale_diag_f64\0")?;
        let fn_xr = sess.get_fn(b"vec_cg_step_xr_f64\0")?;
        let fn_p = sess.get_fn(b"vec_cg_step_p_f64\0")?;
        let fn_dot = sess.get_fn(b"vec_dot_block_f64\0")?;

        let n = b.len();
        let bnorm: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
        if bnorm == 0.0 {
            return Ok(GpuSolveResult {
                solution: vec![0.0; n],
                iterations: 0,
                final_residual: 0.0,
                wall_ms: 0.0,
                kernel_ms: 0.0,
                gflops: 0.0,
                effective_bw_gbps: 0.0,
                devices_used: vec![sess.device_name.clone()],
            });
        }

        let d_row_ptr = DeviceBuffer::from_slice(api, row_ptr)?;
        let d_col_idx = DeviceBuffer::from_slice(api, col_idx)?;
        let d_vals = DeviceBuffer::from_slice(api, vals)?;
        let d_inv_diag = DeviceBuffer::from_slice(api, inv_diag)?;

        let mut x_host = match x0 {
            Some(init) => init.to_vec(),
            None => vec![0.0; n],
        };
        let d_x = DeviceBuffer::from_slice(api, &x_host)?;
        let d_r = DeviceBuffer::from_slice(api, b)?;
        let d_z = DeviceBuffer::zeroed(api, n * 8)?;
        let d_ap = DeviceBuffer::zeroed(api, n * 8)?;

        let dot_blocks: u32 = ((n as u32 + 255) / 256).clamp(1, 512);
        let d_partial = DeviceBuffer::zeroed(api, (dot_blocks as usize) * 8)?;
        let mut h_partial = vec![0.0f64; dot_blocks as usize];

        let gpu_dot = |da: &DeviceBuffer, db: &DeviceBuffer, h_part: &mut [f64]| -> Result<f64, String> {
            let n_u64 = n as u64;
            let mut params: [*mut c_void; 4] = [
                &n_u64 as *const u64 as *mut c_void,
                &da.dptr as *const u64 as *mut c_void,
                &db.dptr as *const u64 as *mut c_void,
                &d_partial.dptr as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_dot,
                        dot_blocks,
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_dot_block_f64)",
                )?;
            }
            d_partial.download(h_part)?;
            Ok(h_part.iter().sum())
        };

        let n_u64 = n as u64;
        let row_offset: u64 = 0;
        let block: u32 = 256;
        let grid: u32 = ((n as u32) + block - 1) / block;

        let t_kern0 = Instant::now();

        // z = M^{-1} r
        let mut params_diag: [*mut c_void; 4] = [
            &n_u64 as *const u64 as *mut c_void,
            &d_r.dptr as *const u64 as *mut c_void,
            &d_inv_diag.dptr as *const u64 as *mut c_void,
            &d_z.dptr as *const u64 as *mut c_void,
        ];
        unsafe {
            check_cu(
                (api.cu_launch_kernel)(
                    fn_scale_diag,
                    grid.max(1),
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    core::ptr::null_mut(),
                    params_diag.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "cuLaunchKernel(vec_scale_diag_f64)",
            )?;
        }

        let d_p = DeviceBuffer::zeroed(api, n * 8)?;
        let zero_beta: f64 = 0.0;
        let mut params_p0: [*mut c_void; 4] = [
            &n_u64 as *const u64 as *mut c_void,
            &d_p.dptr as *const u64 as *mut c_void,
            &d_z.dptr as *const u64 as *mut c_void,
            &zero_beta as *const f64 as *mut c_void,
        ];
        unsafe {
            check_cu(
                (api.cu_launch_kernel)(
                    fn_p,
                    grid.max(1),
                    1,
                    1,
                    block,
                    1,
                    1,
                    0,
                    core::ptr::null_mut(),
                    params_p0.as_mut_ptr(),
                    core::ptr::null_mut(),
                ),
                "cuLaunchKernel(vec_cg_step_p_f64 init)",
            )?;
        }

        let mut rz = gpu_dot(&d_r, &d_z, &mut h_partial)?;
        let mut final_rel = 1.0;
        let mut iters_done = 0usize;

        for it in 1..=max_iter {
            iters_done = it;
            let mut params_spmv: [*mut c_void; 7] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_row_ptr.dptr as *const u64 as *mut c_void,
                &d_col_idx.dptr as *const u64 as *mut c_void,
                &d_vals.dptr as *const u64 as *mut c_void,
                &d_p.dptr as *const u64 as *mut c_void,
                &d_ap.dptr as *const u64 as *mut c_void,
                &row_offset as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_spmv,
                        grid.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_spmv.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(csr_spmv_f64)",
                )?;
            }

            let pap = gpu_dot(&d_p, &d_ap, &mut h_partial)?;
            if pap <= 1e-30 {
                break;
            }
            let alpha = rz / pap;

            let mut params_xr: [*mut c_void; 6] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_x.dptr as *const u64 as *mut c_void,
                &d_r.dptr as *const u64 as *mut c_void,
                &d_p.dptr as *const u64 as *mut c_void,
                &d_ap.dptr as *const u64 as *mut c_void,
                &alpha as *const f64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_xr,
                        grid.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_xr.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_cg_step_xr_f64)",
                )?;
            }

            let rr = gpu_dot(&d_r, &d_r, &mut h_partial)?;
            final_rel = rr.max(0.0).sqrt() / bnorm;
            if final_rel < rtol {
                break;
            }

            let mut params_diag_it: [*mut c_void; 4] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_r.dptr as *const u64 as *mut c_void,
                &d_inv_diag.dptr as *const u64 as *mut c_void,
                &d_z.dptr as *const u64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_scale_diag,
                        grid.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_diag_it.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_scale_diag_f64)",
                )?;
            }

            let rz_new = gpu_dot(&d_r, &d_z, &mut h_partial)?;
            let beta = rz_new / rz.max(1e-30);
            rz = rz_new;

            let mut params_p: [*mut c_void; 4] = [
                &n_u64 as *const u64 as *mut c_void,
                &d_p.dptr as *const u64 as *mut c_void,
                &d_z.dptr as *const u64 as *mut c_void,
                &beta as *const f64 as *mut c_void,
            ];
            unsafe {
                check_cu(
                    (api.cu_launch_kernel)(
                        fn_p,
                        grid.max(1),
                        1,
                        1,
                        block,
                        1,
                        1,
                        0,
                        core::ptr::null_mut(),
                        params_p.as_mut_ptr(),
                        core::ptr::null_mut(),
                    ),
                    "cuLaunchKernel(vec_cg_step_p_f64)",
                )?;
            }
        }

        sess.sync()?;
        let kernel_ms = t_kern0.elapsed().as_secs_f64() * 1000.0;
        d_x.download(&mut x_host)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let nnz = col_idx.len();
        let flops_per_iter = (nnz as f64) * 2.0 + (n as f64) * 12.0;
        let bytes_per_iter = (nnz as f64) * 12.0 + (n as f64) * 72.0;
        let total_secs = (kernel_ms / 1000.0).max(1e-9);
        let gflops = (flops_per_iter * (iters_done as f64)) / total_secs / 1e9;
        let effective_bw_gbps = (bytes_per_iter * (iters_done as f64)) / total_secs / 1e9;

        Ok(GpuSolveResult {
            solution: x_host,
            iterations: iters_done,
            final_residual: final_rel,
            wall_ms,
            kernel_ms,
            gflops,
            effective_bw_gbps,
            devices_used: vec![sess.device_name.clone()],
        })
    }
}

#[cfg(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos")))]
pub use cuda_driver::{
    best_gpu_index, discover_gpus, solve_bsr3_pcg_gpu, solve_csr_cg_gpu, spmv_bsr3_multi_gpu,
    GpuBsr3Context,
};

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
pub fn discover_gpus() -> Vec<GpuDeviceInfo> {
    Vec::new()
}

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
pub fn best_gpu_index() -> Option<usize> {
    None
}

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
pub struct GpuBsr3Context;

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
impl GpuBsr3Context {
    pub fn new(
        _n_nodes: usize,
        _row_ptr: &[u32],
        _cols: &[u32],
        _vals_flat: &[f64],
        _dinv_flat: &[f64],
        _device_index: Option<usize>,
    ) -> Result<Self, String> {
        Err("CUDA GPU acceleration not available on this target".into())
    }
    pub fn device_name(&self) -> &str {
        "none"
    }
    pub fn nnz_blocks(&self) -> usize {
        0
    }
    pub fn spmv(&self, _x: &[f64], _y: &mut [f64]) -> Result<(), String> {
        Err("CUDA GPU acceleration not available on this target".into())
    }
    pub fn jacobi_smooth(
        &self,
        _r: &[f64],
        _z: &mut [f64],
        _sweeps: usize,
        _omega: f64,
        _zero_initial_z: bool,
    ) -> Result<(), String> {
        Err("CUDA GPU acceleration not available on this target".into())
    }
}

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
pub fn spmv_bsr3_multi_gpu(
    _n_nodes: usize,
    _row_ptr: &[u32],
    _cols: &[u32],
    _vals_flat: &[f64],
    _x: &[f64],
    _max_gpus: usize,
) -> Result<(Vec<f64>, Vec<String>), String> {
    Err("CUDA GPU acceleration not available on this target".into())
}

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
pub fn solve_bsr3_pcg_gpu(
    _n_nodes: usize,
    _row_ptr: &[u32],
    _cols: &[u32],
    _vals_flat: &[f64],
    _dinv_flat: &[f64],
    _b: &[f64],
    _x0: Option<&[f64]>,
    _rtol: f64,
    _max_iter: usize,
    _device_index: Option<usize>,
) -> Result<GpuSolveResult, String> {
    Err("CUDA GPU acceleration not available on this target".into())
}

#[cfg(not(all(target_os = "linux", not(target_env = "musl"), not(target_env = "ohos"))))]
pub fn solve_csr_cg_gpu(
    _row_ptr: &[u32],
    _col_idx: &[u32],
    _vals: &[f64],
    _inv_diag: &[f64],
    _b: &[f64],
    _x0: Option<&[f64]>,
    _rtol: f64,
    _max_iter: usize,
    _device_index: Option<usize>,
) -> Result<GpuSolveResult, String> {
    Err("CUDA GPU acceleration not available on this target".into())
}

/// Solve `A * x = b` for a `faer::sparse::SparseColMat<usize, f64>` on the GPU
/// (or fall back to CPU `solve_cg` if no CUDA GPU is available).
pub fn solve_cg_gpu_or_cpu(
    mat: &SparseColMat<usize, f64>,
    b: &[f64],
    x0: Option<&[f64]>,
    tol: f64,
    max_iter: usize,
) -> Result<GpuSolveResult, String> {
    let n = mat.nrows();
    if !discover_gpus().is_empty() {
        // Convert symmetric CSC to CSR (for symmetric matrices CSC == CSR)
        let mut row_counts = vec![0u32; n];
        let mut inv_diag = vec![1.0f64; n];
        for col in 0..n {
            let r_inds = mat.as_ref().row_indices_of_col(col);
            let vals = mat.as_ref().values_of_col(col);
            for (r, &v) in r_inds.zip(vals) {
                let ri = r.unbound();
                row_counts[ri] += 1;
                if ri == col && v.abs() > 1e-15 {
                    inv_diag[ri] = 1.0 / v;
                }
            }
        }
        let mut row_ptr = vec![0u32; n + 1];
        for i in 0..n {
            row_ptr[i + 1] = row_ptr[i] + row_counts[i];
        }
        let nnz = row_ptr[n] as usize;
        let mut col_idx = vec![0u32; nnz];
        let mut csr_vals = vec![0.0f64; nnz];
        let mut cursor = row_ptr[..n].to_vec();
        for col in 0..n {
            let r_inds = mat.as_ref().row_indices_of_col(col);
            let vals = mat.as_ref().values_of_col(col);
            for (r, &v) in r_inds.zip(vals) {
                let ri = r.unbound();
                let pos = cursor[ri] as usize;
                col_idx[pos] = col as u32;
                csr_vals[pos] = v;
                cursor[ri] += 1;
            }
        }
        if let Ok(res) = solve_csr_cg_gpu(&row_ptr, &col_idx, &csr_vals, &inv_diag, b, x0, tol, max_iter, None) {
            return Ok(res);
        }
    }

    let t0 = Instant::now();
    let (sol, iters, res) = crate::cg::solve_cg(mat, b, x0, tol, max_iter)?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    Ok(GpuSolveResult {
        solution: sol,
        iterations: iters,
        final_residual: res,
        wall_ms,
        kernel_ms: wall_ms,
        gflops: 0.0,
        effective_bw_gbps: 0.0,
        devices_used: vec!["CPU (fallback)".into()],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_discover_local_rtx_gpus_and_solve_csr() {
        let gpus = discover_gpus();
        println!("Discovered {} CUDA GPUs: {:?}", gpus.len(), gpus);

        // Build a 3D 7-point Poisson finite-difference SPD matrix (20x20x20 = 8,000 DOFs)
        let nx = 20usize;
        let n = nx * nx * nx;
        let mut triplets = Vec::with_capacity(n * 7);
        for z in 0..nx {
            for y in 0..nx {
                for x in 0..nx {
                    let i = (z * nx + y) * nx + x;
                    triplets.push((i, i, 6.0));
                    if x > 0 {
                        triplets.push((i, i - 1, -1.0));
                    }
                    if x + 1 < nx {
                        triplets.push((i, i + 1, -1.0));
                    }
                    if y > 0 {
                        triplets.push((i, i - nx, -1.0));
                    }
                    if y + 1 < nx {
                        triplets.push((i, i + nx, -1.0));
                    }
                    if z > 0 {
                        triplets.push((i, i - nx * nx, -1.0));
                    }
                    if z + 1 < nx {
                        triplets.push((i, i + nx * nx, -1.0));
                    }
                }
            }
        }
        let mat = SparseColMat::try_new_from_triplets(n, n, &triplets).unwrap();
        let b = vec![1.0; n];

        let (x_cpu, it_cpu, _) = crate::cg::solve_cg(&mat, &b, None, 1e-9, 1000).unwrap();
        let gpu_res = solve_cg_gpu_or_cpu(&mat, &b, None, 1e-9, 1000).unwrap();

        println!(
            "CSR CG 8,000 DOFs -> CPU iters={}, GPU iters={}, wall={:.2}ms, kern={:.2}ms, {:.2} GFLOP/s, {:.2} GB/s on {:?}",
            it_cpu,
            gpu_res.iterations,
            gpu_res.wall_ms,
            gpu_res.kernel_ms,
            gpu_res.gflops,
            gpu_res.effective_bw_gbps,
            gpu_res.devices_used
        );
        for i in (0..n).step_by(137) {
            assert_relative_eq!(gpu_res.solution[i], x_cpu[i], epsilon = 1e-5, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_bsr3_multi_gpu_and_pcg_on_rtx_5090s() {
        let gpus = discover_gpus();
        if gpus.is_empty() {
            return;
        }
        // Build a synthetic 3D 3x3 block-CSR SPD matrix (15x15x15 = 3,375 nodes = 10,125 DOFs)
        let nx = 15usize;
        let n_nodes = nx * nx * nx;
        let mut row_ptr = Vec::with_capacity(n_nodes + 1);
        let mut cols = Vec::new();
        let mut vals_flat = Vec::new();
        let mut dinv_flat = Vec::with_capacity(n_nodes * 9);
        row_ptr.push(0u32);

        let diag_blk = [
            18.0, 1.0, 0.5, //
            1.0, 18.0, 0.5, //
            0.5, 0.5, 18.0,
        ];
        // Inverse of diag_blk
        let det = 18.0 * (18.0 * 18.0 - 0.25) - 1.0 * (18.0 - 0.25) + 0.5 * (0.5 - 9.0);
        let inv_diag_blk = [
            (18.0 * 18.0 - 0.25) / det,
            -(18.0 - 0.25) / det,
            (0.5 - 9.0) / det,
            -(18.0 - 0.25) / det,
            (18.0 * 18.0 - 0.25) / det,
            -(9.0 - 0.5) / det,
            (0.5 - 9.0) / det,
            -(9.0 - 0.5) / det,
            (18.0 * 18.0 - 1.0) / det,
        ];
        let off_blk = [
            -2.0, 0.0, 0.0, //
            0.0, -2.0, 0.0, //
            0.0, 0.0, -2.0,
        ];

        for z in 0..nx {
            for y in 0..nx {
                for x in 0..nx {
                    let i = (z * nx + y) * nx + x;
                    let mut nbrs = vec![i];
                    if x > 0 {
                        nbrs.push(i - 1);
                    }
                    if x + 1 < nx {
                        nbrs.push(i + 1);
                    }
                    if y > 0 {
                        nbrs.push(i - nx);
                    }
                    if y + 1 < nx {
                        nbrs.push(i + nx);
                    }
                    if z > 0 {
                        nbrs.push(i - nx * nx);
                    }
                    if z + 1 < nx {
                        nbrs.push(i + nx * nx);
                    }
                    nbrs.sort_unstable();
                    for &j in &nbrs {
                        cols.push(j as u32);
                        if j == i {
                            vals_flat.extend_from_slice(&diag_blk);
                        } else {
                            vals_flat.extend_from_slice(&off_blk);
                        }
                    }
                    row_ptr.push(cols.len() as u32);
                    dinv_flat.extend_from_slice(&inv_diag_blk);
                }
            }
        }

        let x_in: Vec<f64> = (0..n_nodes * 3).map(|k| ((k % 17) as f64) * 0.1).collect();
        let ctx = GpuBsr3Context::new(n_nodes, &row_ptr, &cols, &vals_flat, &dinv_flat, None).unwrap();
        let mut y_single = vec![0.0; n_nodes * 3];
        ctx.spmv(&x_in, &mut y_single).unwrap();

        let (y_multi, used_devs) =
            spmv_bsr3_multi_gpu(n_nodes, &row_ptr, &cols, &vals_flat, &x_in, 3).unwrap();
        println!("Multi-GPU BSR3 SpMV partitioned across {} GPUs: {:?}", used_devs.len(), used_devs);
        for k in 0..y_single.len() {
            assert_relative_eq!(y_single[k], y_multi[k], epsilon = 1e-12);
        }

        let b = vec![1.0; n_nodes * 3];
        let pcg_res =
            solve_bsr3_pcg_gpu(n_nodes, &row_ptr, &cols, &vals_flat, &dinv_flat, &b, None, 1e-9, 500, None)
                .unwrap();
        println!(
            "BSR3 GPU PCG 10,125 DOFs -> iters={}, res={:.2e}, wall={:.2}ms, kern={:.2}ms, {:.2} GFLOP/s, {:.2} GB/s on {:?}",
            pcg_res.iterations,
            pcg_res.final_residual,
            pcg_res.wall_ms,
            pcg_res.kernel_ms,
            pcg_res.gflops,
            pcg_res.effective_bw_gbps,
            pcg_res.devices_used
        );
        assert!(pcg_res.final_residual < 1e-9);
    }
}

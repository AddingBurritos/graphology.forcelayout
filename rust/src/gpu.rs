use wasm_bindgen::prelude::*;
use wgpu::util::DeviceExt;
use bytemuck;

use crate::simulation::{Body, Spring, SimulationSettings, GpuBody, GpuSpring, GpuParams};

/// WGSL compute shader for naive O(N²) n-body force calculation
const FORCE_SHADER: &str = r#"
struct Body {
    pos: vec4<f32>,  // x, y, z, mass
}

struct Spring {
    from: u32,
    to: u32,
    length: f32,
    coefficient: f32,
}

struct Params {
    body_count: u32,
    spring_count: u32,
    gravity: f32,
    spring_length: f32,
    spring_coefficient: f32,
    _padding: vec3<f32>,
}

@group(0) @binding(0) var<storage, read> bodies: array<Body>;
@group(0) @binding(1) var<storage, read> springs: array<Spring>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> forces: array<vec4<f32>>;

// Compute repulsive forces between all body pairs (naive O(N²))
@compute @workgroup_size(64)
fn compute_repulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.body_count) {
        return;
    }

    let body_i = bodies[i];
    var force = vec3<f32>(0.0, 0.0, 0.0);

    // Iterate over all other bodies
    for (var j = 0u; j < params.body_count; j = j + 1u) {
        if (i == j) {
            continue;
        }

        let body_j = bodies[j];

        // Calculate distance vector
        var d = body_j.pos.xyz - body_i.pos.xyz;
        var r = length(d);

        // Prevent division by zero
        if (r < 0.0001) {
            // Add small random offset based on indices
            let offset = 0.01 * (f32(i) - f32(j));
            d = vec3<f32>(offset, offset * 0.5, offset * 0.25);
            r = length(d);
        }

        // Gravitational/coulomb force: F = G * m1 * m2 / r²
        // We divide by r³ to normalize the direction vector
        let mass_i = body_i.pos.w;
        let mass_j = body_j.pos.w;
        let f = params.gravity * mass_i * mass_j / (r * r * r);

        force = force + f * d;
    }

    forces[i] = vec4<f32>(force, 0.0);
}

// Compute spring forces (attraction between connected nodes)
@compute @workgroup_size(64)
fn compute_springs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let spring_idx = global_id.x;
    if (spring_idx >= params.spring_count) {
        return;
    }

    let spring = springs[spring_idx];
    let body_from = bodies[spring.from];
    let body_to = bodies[spring.to];

    // Calculate distance vector
    var d = body_to.pos.xyz - body_from.pos.xyz;
    var r = length(d);

    // Prevent division by zero
    if (r < 0.0001) {
        return;
    }

    // Spring force: F = k * (r - rest_length)
    let rest_length = select(params.spring_length, spring.length, spring.length > 0.0);
    let k = select(params.spring_coefficient, spring.coefficient, spring.coefficient > 0.0);
    let displacement = r - rest_length;
    let f = k * displacement / r;

    let force = f * d;

    // Atomically add force to both bodies (opposite directions)
    // Note: WGSL doesn't have atomic float operations, so we use a workaround
    // by running this in a separate pass and accumulating on CPU
    // For now, we store spring forces in the forces array offset by body_count

    // Actually, let's compute spring contribution per-body in a separate approach
    // Store the spring force magnitude and we'll accumulate on CPU
    forces[params.body_count + spring_idx] = vec4<f32>(force, f32(spring.from));
}

// Combined kernel that computes all forces for a single body
@compute @workgroup_size(64)
fn compute_all_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.body_count) {
        return;
    }

    let body_i = bodies[i];
    let mass_i = body_i.pos.w;
    var force = vec3<f32>(0.0, 0.0, 0.0);

    // === Repulsive forces (O(N) per thread, O(N²) total) ===
    for (var j = 0u; j < params.body_count; j = j + 1u) {
        if (i == j) {
            continue;
        }

        let body_j = bodies[j];
        var d = body_j.pos.xyz - body_i.pos.xyz;
        var r = length(d);

        if (r < 0.0001) {
            let offset = 0.01 * (f32(i) - f32(j));
            d = vec3<f32>(offset, offset * 0.5, offset * 0.25);
            r = length(d);
        }

        let mass_j = body_j.pos.w;
        let f = params.gravity * mass_i * mass_j / (r * r * r);
        force = force + f * d;
    }

    // === Spring forces ===
    for (var s = 0u; s < params.spring_count; s = s + 1u) {
        let spring = springs[s];

        // Check if this body is connected by this spring
        var other_idx: u32;
        var direction: f32;

        if (spring.from == i) {
            other_idx = spring.to;
            direction = 1.0;
        } else if (spring.to == i) {
            other_idx = spring.from;
            direction = 1.0;
        } else {
            continue;
        }

        let other_body = bodies[other_idx];
        var d = other_body.pos.xyz - body_i.pos.xyz;
        var r = length(d);

        if (r < 0.0001) {
            continue;
        }

        let rest_length = select(params.spring_length, spring.length, spring.length > 0.0);
        let k = select(params.spring_coefficient, spring.coefficient, spring.coefficient > 0.0);
        let displacement = r - rest_length;
        let f = k * displacement / r;

        force = force + direction * f * d;
    }

    forces[i] = vec4<f32>(force, 0.0);
}
"#;

/// GPU-accelerated force simulator using wgpu
pub struct GpuSimulator {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dimensions: usize,
}

impl GpuSimulator {
    /// Initialize the GPU simulator
    pub async fn new(dimensions: usize) -> Result<Self, JsValue> {
        // Get GPU adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| JsValue::from_str("Failed to get GPU adapter"))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Force Layout Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            }, None)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to get GPU device: {:?}", e)))?;

        // Create shader module
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Force Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(FORCE_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Force Bind Group Layout"),
            entries: &[
                // Bodies buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Springs buffer (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Params buffer (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Forces buffer (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Force Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Force Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_all_forces"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Ok(GpuSimulator {
            device,
            queue,
            pipeline,
            bind_group_layout,
            dimensions,
        })
    }

    /// Compute forces for all bodies using GPU
    pub async fn compute_forces(
        &self,
        bodies: &[Body],
        springs: &[Spring],
        settings: &SimulationSettings,
    ) -> Result<Vec<f32>, JsValue> {
        let body_count = bodies.len();
        if body_count == 0 {
            return Ok(vec![]);
        }

        // Convert bodies to GPU format
        let gpu_bodies: Vec<GpuBody> = bodies.iter().map(|b| b.to_gpu()).collect();

        // Convert springs to GPU format (or create dummy if empty)
        let gpu_springs: Vec<GpuSpring> = if springs.is_empty() {
            vec![GpuSpring { from: 0, to: 0, length: 0.0, coefficient: 0.0 }]
        } else {
            springs.iter().map(|s| s.to_gpu()).collect()
        };

        // Create params
        let params = GpuParams {
            body_count: body_count as u32,
            spring_count: springs.len() as u32,
            gravity: settings.gravity,
            spring_length: settings.spring_length,
            spring_coefficient: settings.spring_coefficient,
            _padding: [0.0; 3],
        };

        // Create GPU buffers
        let bodies_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bodies Buffer"),
            contents: bytemuck::cast_slice(&gpu_bodies),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let springs_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Springs Buffer"),
            contents: bytemuck::cast_slice(&gpu_springs),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Output forces buffer (vec4 per body)
        let forces_size = (body_count * std::mem::size_of::<[f32; 4]>()) as u64;
        let forces_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Forces Buffer"),
            size: forces_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Staging buffer for reading back results
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: forces_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Force Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bodies_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: springs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: forces_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder and dispatch compute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Force Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Force Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch with workgroup size of 64
            let workgroups = (body_count as u32 + 63) / 64;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Copy results to staging buffer
        encoder.copy_buffer_to_buffer(&forces_buffer, 0, &staging_buffer, 0, forces_size);

        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));

        // Read back results
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|e| JsValue::from_str(&format!("Failed to receive map result: {:?}", e)))?
            .map_err(|e| JsValue::from_str(&format!("Failed to map buffer: {:?}", e)))?;

        // Extract force data
        let data = buffer_slice.get_mapped_range();
        let force_vec4s: &[[f32; 4]] = bytemuck::cast_slice(&data);

        // Convert to flat array of forces per dimension
        let mut forces = Vec::with_capacity(body_count * self.dimensions);
        for force in force_vec4s.iter().take(body_count) {
            for d in 0..self.dimensions {
                forces.push(force[d]);
            }
        }

        drop(data);
        staging_buffer.unmap();

        Ok(forces)
    }
}

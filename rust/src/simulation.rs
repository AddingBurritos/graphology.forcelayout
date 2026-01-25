use bytemuck::{Pod, Zeroable};

/// A physical body in the simulation
#[derive(Clone, Debug)]
pub struct Body {
    pub pos: Vec<f32>,
    pub vel: Vec<f32>,
    pub force: Vec<f32>,
    pub mass: f32,
    pub pinned: bool,
}

/// A spring connecting two bodies
#[derive(Clone, Debug)]
pub struct Spring {
    pub from: usize,
    pub to: usize,
    pub length: f32,
    pub coefficient: f32,
}

/// Simulation settings
#[derive(Clone, Debug)]
pub struct SimulationSettings {
    pub gravity: f32,
    pub spring_length: f32,
    pub spring_coefficient: f32,
    pub drag_coefficient: f32,
    pub time_step: f32,
}

impl Default for SimulationSettings {
    fn default() -> Self {
        SimulationSettings {
            gravity: -12.0,
            spring_length: 10.0,
            spring_coefficient: 0.8,
            drag_coefficient: 0.9,
            time_step: 0.5,
        }
    }
}

/// GPU-compatible body data (padded for alignment)
/// Uses vec4 for positions to ensure proper GPU alignment
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuBody {
    pub pos: [f32; 4],  // x, y, z, mass (w component stores mass)
}

/// GPU-compatible spring data
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuSpring {
    pub from: u32,
    pub to: u32,
    pub length: f32,
    pub coefficient: f32,
}

/// GPU simulation parameters
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuParams {
    pub body_count: u32,
    pub spring_count: u32,
    pub gravity: f32,
    pub spring_length: f32,
    pub spring_coefficient: f32,
    pub _padding: [f32; 3], // Pad to 32 bytes for alignment
}

impl Body {
    /// Convert to GPU-compatible format
    pub fn to_gpu(&self) -> GpuBody {
        GpuBody {
            pos: [
                self.pos.get(0).copied().unwrap_or(0.0),
                self.pos.get(1).copied().unwrap_or(0.0),
                self.pos.get(2).copied().unwrap_or(0.0),
                self.mass,
            ],
        }
    }
}

impl Spring {
    /// Convert to GPU-compatible format
    pub fn to_gpu(&self) -> GpuSpring {
        GpuSpring {
            from: self.from as u32,
            to: self.to as u32,
            length: self.length,
            coefficient: self.coefficient,
        }
    }
}

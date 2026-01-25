use wasm_bindgen::prelude::*;
use std::collections::HashMap;

mod gpu;
mod simulation;

pub use simulation::{Body, Spring, SimulationSettings};
pub use gpu::GpuSimulator;

#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Main force layout simulator accessible from JavaScript
#[wasm_bindgen]
pub struct ForceLayout {
    bodies: Vec<Body>,
    springs: Vec<Spring>,
    node_index: HashMap<String, usize>,
    settings: SimulationSettings,
    gpu: Option<GpuSimulator>,
    dimensions: usize,
}

#[wasm_bindgen]
impl ForceLayout {
    /// Create a new force layout with the given settings
    #[wasm_bindgen(constructor)]
    pub fn new(dimensions: u32) -> ForceLayout {
        ForceLayout {
            bodies: Vec::new(),
            springs: Vec::new(),
            node_index: HashMap::new(),
            settings: SimulationSettings::default(),
            gpu: None,
            dimensions: dimensions as usize,
        }
    }

    /// Initialize the GPU context - must be called before step()
    #[wasm_bindgen]
    pub async fn init_gpu(&mut self) -> Result<(), JsValue> {
        let gpu = GpuSimulator::new(self.dimensions).await?;
        self.gpu = Some(gpu);
        Ok(())
    }

    /// Check if GPU is initialized
    #[wasm_bindgen]
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// Add a body (node) to the simulation
    #[wasm_bindgen]
    pub fn add_body(&mut self, id: &str, x: f32, y: f32, z: f32, mass: f32) {
        let idx = self.bodies.len();
        self.node_index.insert(id.to_string(), idx);

        let mut pos = vec![0.0f32; self.dimensions.max(3)];
        pos[0] = x;
        pos[1] = y;
        if self.dimensions > 2 {
            pos[2] = z;
        }

        self.bodies.push(Body {
            pos,
            vel: vec![0.0; self.dimensions],
            force: vec![0.0; self.dimensions],
            mass,
            pinned: false,
        });
    }

    /// Add a spring (edge) between two nodes
    #[wasm_bindgen]
    pub fn add_spring(&mut self, from_id: &str, to_id: &str, length: f32, coefficient: f32) -> bool {
        let from_idx = match self.node_index.get(from_id) {
            Some(&idx) => idx,
            None => return false,
        };
        let to_idx = match self.node_index.get(to_id) {
            Some(&idx) => idx,
            None => return false,
        };

        self.springs.push(Spring {
            from: from_idx,
            to: to_idx,
            length: if length < 0.0 { self.settings.spring_length } else { length },
            coefficient: if coefficient < 0.0 { self.settings.spring_coefficient } else { coefficient },
        });
        true
    }

    /// Remove a body from the simulation
    #[wasm_bindgen]
    pub fn remove_body(&mut self, id: &str) -> bool {
        let idx = match self.node_index.remove(id) {
            Some(idx) => idx,
            None => return false,
        };

        self.bodies.remove(idx);

        // Update indices for bodies after the removed one
        for (_, body_idx) in self.node_index.iter_mut() {
            if *body_idx > idx {
                *body_idx -= 1;
            }
        }

        // Remove springs connected to this body and update indices
        self.springs.retain_mut(|spring| {
            if spring.from == idx || spring.to == idx {
                return false;
            }
            if spring.from > idx {
                spring.from -= 1;
            }
            if spring.to > idx {
                spring.to -= 1;
            }
            true
        });

        true
    }

    /// Set node position
    #[wasm_bindgen]
    pub fn set_position(&mut self, id: &str, x: f32, y: f32, z: f32) -> bool {
        if let Some(&idx) = self.node_index.get(id) {
            self.bodies[idx].pos[0] = x;
            self.bodies[idx].pos[1] = y;
            if self.dimensions > 2 {
                self.bodies[idx].pos[2] = z;
            }
            true
        } else {
            false
        }
    }

    /// Get node position as [x, y, z]
    #[wasm_bindgen]
    pub fn get_position(&self, id: &str) -> Option<Vec<f32>> {
        self.node_index.get(id).map(|&idx| {
            let body = &self.bodies[idx];
            vec![
                body.pos[0],
                body.pos[1],
                if self.dimensions > 2 { body.pos[2] } else { 0.0 },
            ]
        })
    }

    /// Pin a node (prevent it from moving)
    #[wasm_bindgen]
    pub fn pin_node(&mut self, id: &str, pinned: bool) -> bool {
        if let Some(&idx) = self.node_index.get(id) {
            self.bodies[idx].pinned = pinned;
            true
        } else {
            false
        }
    }

    /// Check if a node is pinned
    #[wasm_bindgen]
    pub fn is_pinned(&self, id: &str) -> bool {
        self.node_index.get(id)
            .map(|&idx| self.bodies[idx].pinned)
            .unwrap_or(false)
    }

    /// Set simulation settings
    #[wasm_bindgen]
    pub fn set_settings(
        &mut self,
        gravity: f32,
        spring_length: f32,
        spring_coefficient: f32,
        drag_coefficient: f32,
        time_step: f32,
    ) {
        self.settings = SimulationSettings {
            gravity,
            spring_length,
            spring_coefficient,
            drag_coefficient,
            time_step,
        };
    }

    /// Get the number of bodies
    #[wasm_bindgen]
    pub fn body_count(&self) -> usize {
        self.bodies.len()
    }

    /// Get the number of springs
    #[wasm_bindgen]
    pub fn spring_count(&self) -> usize {
        self.springs.len()
    }

    /// Perform one simulation step, returns total movement
    #[wasm_bindgen]
    pub async fn step(&mut self) -> Result<f32, JsValue> {
        if self.bodies.is_empty() {
            return Ok(0.0);
        }

        let gpu = self.gpu.as_mut()
            .ok_or_else(|| JsValue::from_str("GPU not initialized. Call init_gpu() first."))?;

        // Run GPU force calculation
        let forces = gpu.compute_forces(&self.bodies, &self.springs, &self.settings).await?;

        // Apply forces and integrate
        let mut total_movement = 0.0f32;

        for (i, body) in self.bodies.iter_mut().enumerate() {
            if body.pinned {
                continue;
            }

            let force_offset = i * self.dimensions;

            for d in 0..self.dimensions {
                // Update velocity
                let force = forces[force_offset + d];
                body.vel[d] = (body.vel[d] + force * self.settings.time_step) * self.settings.drag_coefficient;

                // Clamp velocity
                if body.vel[d] > 1.0 {
                    body.vel[d] = 1.0;
                } else if body.vel[d] < -1.0 {
                    body.vel[d] = -1.0;
                }

                // Update position
                let dx = body.vel[d] * self.settings.time_step;
                body.pos[d] += dx;
                total_movement += dx.abs();
            }
        }

        Ok(total_movement)
    }

    /// Get all positions as a flat Float32Array [x0, y0, z0, x1, y1, z1, ...]
    #[wasm_bindgen]
    pub fn get_all_positions(&self) -> Vec<f32> {
        let mut positions = Vec::with_capacity(self.bodies.len() * 3);
        for body in &self.bodies {
            positions.push(body.pos[0]);
            positions.push(body.pos[1]);
            positions.push(if self.dimensions > 2 { body.pos[2] } else { 0.0 });
        }
        positions
    }

    /// Get bounding box as [min_x, min_y, min_z, max_x, max_y, max_z]
    #[wasm_bindgen]
    pub fn get_bounding_box(&self) -> Vec<f32> {
        if self.bodies.is_empty() {
            return vec![0.0; 6];
        }

        let mut min = vec![f32::INFINITY; 3];
        let mut max = vec![f32::NEG_INFINITY; 3];

        for body in &self.bodies {
            for d in 0..3.min(self.dimensions) {
                if body.pos[d] < min[d] {
                    min[d] = body.pos[d];
                }
                if body.pos[d] > max[d] {
                    max[d] = body.pos[d];
                }
            }
        }

        vec![min[0], min[1], min[2], max[0], max[1], max[2]]
    }
}

# graphology-forcelayout-gpu

GPU-accelerated force-directed graph layout using Rust + WebGPU (wgpu).

## Overview

This is a WebAssembly implementation of the force-directed layout algorithm that uses GPU compute shaders for the n-body force calculation. It implements the naive O(N²) algorithm which, despite its higher complexity, runs faster on GPU than the Barnes-Hut O(N log N) algorithm due to:

- Massive parallelism (thousands of GPU cores vs 8-16 CPU cores)
- No branch divergence from tree traversal
- Predictable memory access patterns
- Higher memory bandwidth (~900 GB/s vs ~50 GB/s)

## Algorithm

The GPU compute shader calculates:

1. **Repulsive forces** (Coulomb's law): Each body repels all other bodies with force proportional to `G * m1 * m2 / r²`

2. **Spring forces** (Hooke's law): Connected bodies attract/repel to reach ideal spring length with force `k * (r - rest_length)`

3. **Integration**: Euler integration with velocity clamping and drag

## Building

```bash
# Install wasm-pack if needed
cargo install wasm-pack

# Build the WASM package
wasm-pack build --target web
```

## Usage

```javascript
import { createLayout, initWasm } from './rust/js/index.js';
import Graph from 'graphology';

// Initialize WASM (call once)
await initWasm();

// Create a graph
const graph = new Graph();
graph.addNode('a', { x: 0, y: 0 });
graph.addNode('b', { x: 100, y: 0 });
graph.addEdge('a', 'b');

// Create GPU-accelerated layout
const layout = await createLayout(graph, {
    dimensions: 2,
    gravity: -12,
    springLength: 10,
    springCoefficient: 0.8,
    dragCoefficient: 0.9,
    timeStep: 0.5,
});

// Run simulation steps
for (let i = 0; i < 100; i++) {
    const movement = await layout.step();
    if (movement < 0.001) break; // Stable
}

// Positions are automatically synced to graph node attributes
console.log(graph.getNodeAttributes('a')); // { x: ..., y: ... }

// Clean up
layout.dispose();
```

## API

### `initWasm()`
Initialize the WASM module. Call once before creating layouts.

### `createLayout(graph, settings)`
Create a new GPU-accelerated force layout.

**Settings:**
- `dimensions` (default: 2) - Number of spatial dimensions (2 or 3)
- `gravity` (default: -12) - Repulsion strength (negative = repel)
- `springLength` (default: 10) - Ideal spring rest length
- `springCoefficient` (default: 0.8) - Spring stiffness
- `dragCoefficient` (default: 0.9) - Velocity damping (0-1)
- `timeStep` (default: 0.5) - Integration time step

**Returns:** Layout object with methods:
- `step()` - Run one simulation step, returns total movement
- `getNodePosition(id)` - Get `{x, y, z}` position
- `setNodePosition(id, x, y, z)` - Set position
- `pinNode(id, pinned)` - Pin/unpin a node
- `isNodePinned(id)` - Check if pinned
- `getGraphRect()` - Get bounding box
- `dispose()` - Clean up resources

## Requirements

- Browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly)
- For unsupported browsers, falls back will need to be implemented

## Complexity

| Operation | Time | Space |
|-----------|------|-------|
| Force calculation (GPU) | O(N²/P) | O(N) |
| Integration (CPU) | O(N) | O(N) |
| Total per step | O(N²/P + N) | O(N) |

Where P = number of GPU cores (typically 1000-10000).

For a 10,000 node graph with 4096 GPU cores:
- GPU: 10,000² / 4096 ≈ 24,000 operations per core
- This runs in parallel, so effective time is similar to O(N)

## License

BSD-3-Clause

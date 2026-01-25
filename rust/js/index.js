/**
 * GPU-accelerated force layout for graphology using WebGPU
 *
 * This module provides a drop-in replacement for the CPU-based force layout,
 * using Rust + WebGPU for O(N²) naive n-body force calculation on the GPU.
 */

import init, { ForceLayout } from '../pkg/graphology_forcelayout_gpu.js';

let wasmInitialized = false;
let initPromise = null;

/**
 * Initialize the WASM module (call once before creating layouts)
 */
export async function initWasm() {
    if (wasmInitialized) return;
    if (initPromise) return initPromise;

    initPromise = init();
    await initPromise;
    wasmInitialized = true;
}

/**
 * Default settings for the force layout
 */
const defaultSettings = {
    dimensions: 2,
    gravity: -12,
    springLength: 10,
    springCoefficient: 0.8,
    dragCoefficient: 0.9,
    timeStep: 0.5,
};

/**
 * Create a GPU-accelerated force layout for a graphology graph
 *
 * @param {Graph} graph - A graphology graph instance
 * @param {Object} settings - Layout settings
 * @returns {Promise<Object>} - Layout API object
 */
export async function createLayout(graph, settings = {}) {
    await initWasm();

    const config = { ...defaultSettings, ...settings };
    const layout = new ForceLayout(config.dimensions);

    // Initialize GPU
    await layout.init_gpu();

    // Apply settings
    layout.set_settings(
        config.gravity,
        config.springLength,
        config.springCoefficient,
        config.dragCoefficient,
        config.timeStep
    );

    // Node ID to internal tracking
    const nodeIds = [];

    // Initialize bodies from existing nodes
    graph.forEachNode((nodeId, attributes) => {
        const pos = attributes.position || attributes;
        const x = pos.x ?? Math.random() * 100 - 50;
        const y = pos.y ?? Math.random() * 100 - 50;
        const z = pos.z ?? (config.dimensions > 2 ? Math.random() * 100 - 50 : 0);
        const mass = attributes.mass ?? 1;

        layout.add_body(nodeId, x, y, z, mass);
        nodeIds.push(nodeId);

        if (attributes.pinned || attributes.fixed) {
            layout.pin_node(nodeId, true);
        }
    });

    // Initialize springs from existing edges
    graph.forEachEdge((edgeId, attributes, source, target) => {
        const length = attributes.length ?? -1;
        const coefficient = attributes.springCoefficient ?? -1;
        layout.add_spring(source, target, length, coefficient);
    });

    // Event handlers for graph modifications
    const onNodeAdded = (payload) => {
        const { key, attributes } = payload;
        const pos = attributes.position || attributes;
        const x = pos.x ?? Math.random() * 100 - 50;
        const y = pos.y ?? Math.random() * 100 - 50;
        const z = pos.z ?? 0;
        const mass = attributes.mass ?? 1;

        layout.add_body(key, x, y, z, mass);
        nodeIds.push(key);

        if (attributes.pinned || attributes.fixed) {
            layout.pin_node(key, true);
        }
    };

    const onNodeDropped = (payload) => {
        const { key } = payload;
        layout.remove_body(key);
        const idx = nodeIds.indexOf(key);
        if (idx !== -1) nodeIds.splice(idx, 1);
    };

    const onEdgeAdded = (payload) => {
        const { attributes, source, target } = payload;
        const length = attributes?.length ?? -1;
        const coefficient = attributes?.springCoefficient ?? -1;
        layout.add_spring(source, target, length, coefficient);
    };

    // Subscribe to graph events
    graph.on('nodeAdded', onNodeAdded);
    graph.on('nodeDropped', onNodeDropped);
    graph.on('edgeAdded', onEdgeAdded);

    // Track if disposed
    let disposed = false;

    // Public API (mirrors the original library)
    return {
        /**
         * Perform one simulation step
         * @returns {Promise<number>} Total movement (use to detect stability)
         */
        async step() {
            if (disposed) return 0;

            const movement = await layout.step();

            // Sync positions back to graph
            const positions = layout.get_all_positions();
            for (let i = 0; i < nodeIds.length; i++) {
                const nodeId = nodeIds[i];
                if (graph.hasNode(nodeId)) {
                    graph.setNodeAttribute(nodeId, 'x', positions[i * 3]);
                    graph.setNodeAttribute(nodeId, 'y', positions[i * 3 + 1]);
                    if (config.dimensions > 2) {
                        graph.setNodeAttribute(nodeId, 'z', positions[i * 3 + 2]);
                    }
                }
            }

            return movement;
        },

        /**
         * Get position of a node
         * @param {string} nodeId
         * @returns {{ x: number, y: number, z: number } | null}
         */
        getNodePosition(nodeId) {
            const pos = layout.get_position(nodeId);
            if (!pos) return null;
            return { x: pos[0], y: pos[1], z: pos[2] };
        },

        /**
         * Set position of a node
         * @param {string} nodeId
         * @param {number} x
         * @param {number} y
         * @param {number} z
         */
        setNodePosition(nodeId, x, y, z = 0) {
            layout.set_position(nodeId, x, y, z);
        },

        /**
         * Pin a node (prevent it from moving)
         * @param {string} nodeId
         * @param {boolean} pinned
         */
        pinNode(nodeId, pinned = true) {
            layout.pin_node(nodeId, pinned);
        },

        /**
         * Check if a node is pinned
         * @param {string} nodeId
         * @returns {boolean}
         */
        isNodePinned(nodeId) {
            return layout.is_pinned(nodeId);
        },

        /**
         * Get the bounding box of all nodes
         * @returns {{ min_x, min_y, min_z, max_x, max_y, max_z }}
         */
        getGraphRect() {
            const bbox = layout.get_bounding_box();
            return {
                min_x: bbox[0],
                min_y: bbox[1],
                min_z: bbox[2],
                max_x: bbox[3],
                max_y: bbox[4],
                max_z: bbox[5],
            };
        },

        /**
         * Get number of bodies in simulation
         * @returns {number}
         */
        getBodyCount() {
            return layout.body_count();
        },

        /**
         * Get number of springs in simulation
         * @returns {number}
         */
        getSpringCount() {
            return layout.spring_count();
        },

        /**
         * Check if GPU is available
         * @returns {boolean}
         */
        hasGpu() {
            return layout.has_gpu();
        },

        /**
         * Get dimensions
         * @returns {number}
         */
        getDimensions() {
            return config.dimensions;
        },

        /**
         * Dispose of the layout and clean up resources
         */
        dispose() {
            if (disposed) return;
            disposed = true;

            graph.off('nodeAdded', onNodeAdded);
            graph.off('nodeDropped', onNodeDropped);
            graph.off('edgeAdded', onEdgeAdded);

            // Free WASM memory
            layout.free();
        },
    };
}

export { ForceLayout };
export default createLayout;

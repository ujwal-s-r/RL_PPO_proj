/**
 * GPU Cluster Architect — Interactive Topology Builder Controller (Stages 1 & 2)
 */

// Application State
let selectedNodeCount = 3;
let nodeGpus = []; // Array of arrays: nodeGpus[nodeIdx] = [ { type, vram, name, badge } ]
let draggedGpuData = null;

// Available GPU catalog
const GPU_CATALOG = {
    "NVIDIA-H100-80GB": { type: "NVIDIA-H100-80GB", vram: 80.0, name: "NVIDIA H100", badge: "badge-hopper", badgeText: "Hopper" },
    "A100-SXM4-80GB": { type: "A100-SXM4-80GB", vram: 80.0, name: "A100 SXM4", badge: "badge-ampere", badgeText: "Ampere" },
    "A100-PCIE-40GB": { type: "A100-PCIE-40GB", vram: 40.0, name: "A100 PCIe", badge: "badge-pcie", badgeText: "PCIe" },
    "NVIDIA-A10-24GB": { type: "NVIDIA-A10-24GB", vram: 24.0, name: "NVIDIA A10", badge: "badge-a10", badgeText: "Inference" },
};

// DOM Elements
const stage1 = document.getElementById("stage-1");
const stage2 = document.getElementById("stage-2");
const stepNav1 = document.getElementById("step-nav-1");
const stepNav2 = document.getElementById("step-nav-2");

const selectedNodesBadge = document.getElementById("selected-nodes-badge");
const numbersTrack = document.getElementById("numbers-track");
const nodesStackGrid = document.getElementById("nodes-stack-grid");
const btnGotoStage2 = document.getElementById("btn-goto-stage-2");

const btnBackToStage1 = document.getElementById("btn-back-to-stage-1");
const btnFillBalanced = document.getElementById("btn-fill-balanced");
const btnFillTraining = document.getElementById("btn-fill-training");
const btnFillRandom = document.getElementById("btn-fill-random");
const btnClearAllGpus = document.getElementById("btn-clear-all-gpus");

const hierarchyNodesRow = document.getElementById("hierarchy-nodes-row");
const hierarchySvgLines = document.getElementById("hierarchy-svg-lines");
const clusterSpecsTally = document.getElementById("cluster-specs-tally");

// ================= STAGE 1: NODE COUNT SELECTION =================

function initStage1() {
    renderNumberButtons();
    renderStage1NodesStack();
}

function setNodeCount(count) {
    selectedNodeCount = count;
    selectedNodesBadge.textContent = `${selectedNodeCount} Node${selectedNodeCount > 1 ? 's' : ''} Selected`;
    
    // Update active button
    document.querySelectorAll(".num-btn").forEach(btn => {
        btn.classList.toggle("active", parseInt(btn.dataset.val, 10) === selectedNodeCount);
    });

    renderStage1NodesStack();
    syncNodeGpusArray();
}

function renderNumberButtons() {
    numbersTrack.innerHTML = "";
    for (let i = 1; i <= 10; i++) {
        const btn = document.createElement("button");
        btn.className = `num-btn ${i === selectedNodeCount ? 'active' : ''}`;
        btn.dataset.val = i;
        btn.textContent = i;
        btn.addEventListener("click", () => setNodeCount(i));
        numbersTrack.appendChild(btn);
    }
}

function renderStage1NodesStack() {
    nodesStackGrid.innerHTML = "";
    for (let i = 0; i < selectedNodeCount; i++) {
        const card = document.createElement("div");
        card.className = "stack-node-card";
        card.innerHTML = `
            <div class="node-icon-chassis">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="2" y="2" width="20" height="8" rx="2"></rect>
                    <rect x="2" y="14" width="20" height="8" rx="2"></rect>
                    <line x1="6" y1="6" x2="6.01" y2="6"></line>
                    <line x1="6" y1="18" x2="6.01" y2="18"></line>
                </svg>
            </div>
            <div class="stack-node-title">Node ${i}</div>
            <span class="stack-node-status">Online Slot</span>
        `;
        nodesStackGrid.appendChild(card);
    }
}

// Synchronize nodeGpus array to match selectedNodeCount
function syncNodeGpusArray() {
    while (nodeGpus.length < selectedNodeCount) {
        nodeGpus.push([]);
    }
    if (nodeGpus.length > selectedNodeCount) {
        nodeGpus = nodeGpus.slice(0, selectedNodeCount);
    }
}

// ================= STAGE 2: GPU ASSIGNMENT & HIERARCHY =================

function initStage2() {
    syncNodeGpusArray();
    if (nodeGpus.every(arr => arr.length === 0)) {
        applyPreset("balanced");
    } else {
        renderHierarchy();
    }
    initPaletteDragAndClick();
}

function initPaletteDragAndClick() {
    document.querySelectorAll(".gpu-source-card").forEach(card => {
        card.addEventListener("dragstart", (e) => {
            draggedGpuData = {
                type: card.dataset.type,
                vram: parseFloat(card.dataset.vram),
                name: card.dataset.name,
                badge: card.dataset.badge,
            };
            e.dataTransfer.setData("text/plain", card.dataset.type);
        });

        // Click-to-Add to first available node
        card.addEventListener("click", () => {
            const gpuInfo = {
                type: card.dataset.type,
                vram: parseFloat(card.dataset.vram),
                name: card.dataset.name,
                badge: card.dataset.badge,
            };
            // Find first node with < 8 GPUs
            let targetNode = nodeGpus.findIndex(gpus => gpus.length < 8);
            if (targetNode === -1) targetNode = 0;
            addGpuToNode(targetNode, gpuInfo);
        });
    });
}

function addGpuToNode(nodeIdx, gpuInfo) {
    if (!nodeGpus[nodeIdx]) nodeGpus[nodeIdx] = [];
    if (nodeGpus[nodeIdx].length >= 8) {
        alert(`Node ${nodeIdx} has reached maximum capacity of 8 GPUs.`);
        return;
    }
    nodeGpus[nodeIdx].push(gpuInfo);
    renderHierarchy();
}

function removeGpuFromNode(nodeIdx, gpuIndex) {
    if (nodeGpus[nodeIdx]) {
        nodeGpus[nodeIdx].splice(gpuIndex, 1);
        renderHierarchy();
    }
}

function renderHierarchy() {
    hierarchyNodesRow.innerHTML = "";

    nodeGpus.forEach((gpus, nodeIdx) => {
        const col = document.createElement("div");
        col.className = "hierarchy-node-column";
        col.dataset.nodeIndex = nodeIdx;

        const primaryType = gpus.length > 0 ? gpus[0].name : "Unassigned";
        const capCount = gpus.length;
        const isFull = capCount >= 8;

        // 1. Node Chassis Rectangle (Top of Column)
        const chassis = document.createElement("div");
        chassis.className = "node-chassis-rect";
        chassis.id = `node-rect-${nodeIdx}`;
        chassis.innerHTML = `
            <div class="node-rect-info">
                <div class="node-rect-title">Node ${nodeIdx}</div>
                <div class="node-rect-gputype">${primaryType}</div>
            </div>
            <div class="node-edge-capacity">
                <span class="capacity-label">CAPACITY</span>
                <span class="capacity-count">${capCount} / 8 GPUs</span>
                <div class="capacity-meter-bar">
                    <div class="capacity-meter-fill ${isFull ? 'full' : ''}" style="width: ${(capCount / 8) * 100}%;"></div>
                </div>
            </div>
        `;

        // Setup Drop Target on Chassis
        chassis.addEventListener("dragover", (e) => {
            e.preventDefault();
            chassis.classList.add("drag-over");
        });

        chassis.addEventListener("dragleave", () => {
            chassis.classList.remove("drag-over");
        });

        chassis.addEventListener("drop", (e) => {
            e.preventDefault();
            chassis.classList.remove("drag-over");
            if (draggedGpuData) {
                addGpuToNode(nodeIdx, draggedGpuData);
                draggedGpuData = null;
            }
        });

        // 2. Child GPUs Array (Connected Below Node)
        const gpusGrid = document.createElement("div");
        gpusGrid.className = "node-child-gpus-grid";
        gpusGrid.id = `gpus-grid-${nodeIdx}`;

        if (gpus.length === 0) {
            gpusGrid.innerHTML = `<div class="empty-gpu-dropzone-hint">+ Drop GPUs here or click cards above</div>`;
        } else {
            gpus.forEach((gpu, gpuIdx) => {
                const chip = document.createElement("div");
                chip.className = "gpu-assigned-chip";
                chip.id = `gpu-chip-${nodeIdx}-${gpuIdx}`;
                chip.innerHTML = `
                    <div class="assigned-chip-left">
                        <span class="gpu-badge ${GPU_CATALOG[gpu.type]?.badge || 'badge-ampere'}">${GPU_CATALOG[gpu.type]?.badgeText || 'GPU'}</span>
                        <span class="assigned-gpu-name">${gpu.name}</span>
                    </div>
                    <div class="assigned-chip-right">
                        <span class="text-muted small">${gpu.vram.toFixed(0)} GB</span>
                        <button class="btn-remove-gpu" title="Remove GPU" onclick="removeGpuFromNode(${nodeIdx}, ${gpuIdx})">✕</button>
                    </div>
                `;
                gpusGrid.appendChild(chip);
            });
        }

        col.appendChild(chassis);
        col.appendChild(gpusGrid);
        hierarchyNodesRow.appendChild(col);
    });

    updateClusterTally();
    
    // Draw SVG Dotted Animated Connecting Lines after DOM layout settles
    requestAnimationFrame(() => {
        setTimeout(drawConnectingLines, 50);
    });
}

function updateClusterTally() {
    const totalGpus = nodeGpus.reduce((sum, gpus) => sum + gpus.length, 0);
    const totalVram = nodeGpus.reduce((sum, gpus) => sum + gpus.reduce((s, g) => s + g.vram, 0), 0);
    clusterSpecsTally.innerHTML = `Cluster Summary: <strong>${selectedNodeCount} Nodes</strong> · <strong>${totalGpus} Total GPUs</strong> · <strong>${totalVram.toFixed(0)} GB VRAM</strong>`;
}

// ================= SVG DOTTED ANIMATED CONNECTING LINES =================

function drawConnectingLines() {
    hierarchySvgLines.innerHTML = "";
    const viewportRect = document.querySelector(".hierarchy-diagram-viewport").getBoundingClientRect();

    nodeGpus.forEach((gpus, nodeIdx) => {
        const chassisEl = document.getElementById(`node-rect-${nodeIdx}`);
        if (!chassisEl) return;

        const chassisRect = chassisEl.getBoundingClientRect();
        const startX = chassisRect.left + (chassisRect.width / 2) - viewportRect.left;
        const startY = chassisRect.bottom - viewportRect.top;

        gpus.forEach((gpu, gpuIdx) => {
            const chipEl = document.getElementById(`gpu-chip-${nodeIdx}-${gpuIdx}`);
            if (!chipEl) return;

            const chipRect = chipEl.getBoundingClientRect();
            const endX = chipRect.left + (chipRect.width / 2) - viewportRect.left;
            const endY = chipRect.top - viewportRect.top;

            // Draw smooth Bezier curve from Node Chassis to Child GPU Chip
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            const midY = (startY + endY) / 2;
            const d = `M ${startX} ${startY} C ${startX} ${midY}, ${endX} ${midY}, ${endX} ${endY}`;
            
            path.setAttribute("d", d);
            path.setAttribute("class", "dotted-connector-line");
            hierarchySvgLines.appendChild(path);
        });
    });
}

window.addEventListener("resize", drawConnectingLines);

// ================= PRESET TEMPLATES =================

function applyPreset(presetType) {
    nodeGpus = [];
    for (let i = 0; i < selectedNodeCount; i++) {
        if (presetType === "balanced") {
            if (i % 3 === 0) {
                nodeGpus.push([GPU_CATALOG["A100-SXM4-80GB"], GPU_CATALOG["A100-SXM4-80GB"]]);
            } else if (i % 3 === 1) {
                nodeGpus.push([GPU_CATALOG["A100-PCIE-40GB"], GPU_CATALOG["A100-PCIE-40GB"], GPU_CATALOG["A100-PCIE-40GB"], GPU_CATALOG["A100-PCIE-40GB"]]);
            } else {
                nodeGpus.push([GPU_CATALOG["NVIDIA-A10-24GB"], GPU_CATALOG["NVIDIA-A10-24GB"]]);
            }
        } else if (presetType === "training") {
            nodeGpus.push([
                GPU_CATALOG["NVIDIA-H100-80GB"], GPU_CATALOG["NVIDIA-H100-80GB"],
                GPU_CATALOG["NVIDIA-H100-80GB"], GPU_CATALOG["NVIDIA-H100-80GB"],
                GPU_CATALOG["NVIDIA-H100-80GB"], GPU_CATALOG["NVIDIA-H100-80GB"],
                GPU_CATALOG["NVIDIA-H100-80GB"], GPU_CATALOG["NVIDIA-H100-80GB"],
            ]);
        } else if (presetType === "random") {
            const keys = Object.keys(GPU_CATALOG);
            const count = Math.floor(Math.random() * 4) + 1; // 1 to 4 GPUs
            const chosenKey = keys[Math.floor(Math.random() * keys.length)];
            const list = [];
            for (let k = 0; k < count; k++) {
                list.push(GPU_CATALOG[chosenKey]);
            }
            nodeGpus.push(list);
        }
    }
    renderHierarchy();
}

// ================= EVENT LISTENERS & NAVIGATION =================

btnGotoStage2.addEventListener("click", () => {
    stage1.classList.add("hidden");
    stage2.classList.remove("hidden");
    stepNav1.classList.remove("active");
    stepNav2.classList.add("active");
    initStage2();
});

btnBackToStage1.addEventListener("click", () => {
    stage2.classList.add("hidden");
    stage1.classList.remove("hidden");
    stepNav2.classList.remove("active");
    stepNav1.classList.add("active");
});

btnFillBalanced.addEventListener("click", () => applyPreset("balanced"));
btnFillTraining.addEventListener("click", () => applyPreset("training"));
btnFillRandom.addEventListener("click", () => applyPreset("random"));
btnClearAllGpus.addEventListener("click", () => {
    nodeGpus = nodeGpus.map(() => []);
    renderHierarchy();
});

// Initialize on load
initStage1();

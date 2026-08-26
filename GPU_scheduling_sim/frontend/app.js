/**
 * GPU Cluster Visualizer — Diagrammatic Frontend Controller
 */

const API_BASE = window.location.origin;

let autoPlayInterval = null;
let isAutoPlaying = false;
let lastScheduledNodeId = null;

// DOM Elements
const simTimeEl = document.getElementById("sim-time");
const clusterSummaryChip = document.getElementById("cluster-summary-chip");
const policySelect = document.getElementById("policy-select");
const scenarioSelect = document.getElementById("scenario-select");
const btnStep = document.getElementById("btn-step");
const btnAuto = document.getElementById("btn-auto");
const autoText = document.getElementById("auto-text");
const btnReset = document.getElementById("btn-reset");

const statGpuUtil = document.getElementById("stat-gpu-util");
const statThroughput = document.getElementById("stat-throughput");
const statCompleted = document.getElementById("stat-completed");

const nodesContainer = document.getElementById("nodes-container");
const queueCountEl = document.getElementById("queue-count");
const queueMeanWait = document.getElementById("queue-mean-wait");
const queueContainer = document.getElementById("queue-container");
const latestDecisionText = document.getElementById("latest-decision-text");
const jobForm = document.getElementById("job-form");

// --- API Calls ---

async function fetchStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/cluster/status`);
        if (!res.ok) throw new Error("Failed to fetch cluster status");
        const data = await res.json();
        renderDashboard(data);
    } catch (err) {
        console.error("Fetch error:", err);
    }
}

async function stepSimulation() {
    const policy = policySelect.value;
    try {
        btnStep.disabled = true;
        const res = await fetch(`${API_BASE}/api/schedule/step`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ policy: policy, auto_advance: true }),
        });
        const result = await res.json();
        if (result.action_taken && result.node_id !== undefined) {
            lastScheduledNodeId = result.node_id;
            latestDecisionText.innerHTML = `<strong>[${policy}]</strong> Scheduled <strong>Job #${result.job_id}</strong> on <strong>Node ${result.node_id}</strong> (Sim Time: ${result.sim_time.toFixed(1)}s)`;
        } else {
            lastScheduledNodeId = null;
            latestDecisionText.textContent = `[${policy}] Advanced time to ${result.sim_time.toFixed(1)}s (no placement needed).`;
        }
        await fetchStatus();
    } catch (err) {
        console.error("Step error:", err);
    } finally {
        btnStep.disabled = false;
    }
}

async function resetCluster() {
    const scenario = scenarioSelect.value;
    try {
        await fetch(`${API_BASE}/api/cluster/reset`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                cluster_config: "configs/cluster_small.yaml",
                scenario: scenario,
                seed: Math.floor(Math.random() * 1000) + 1,
            }),
        });
        lastScheduledNodeId = null;
        latestDecisionText.textContent = `Cluster reset with "${scenario}" scenario.`;
        await fetchStatus();
    } catch (err) {
        console.error("Reset error:", err);
    }
}

async function submitJob(e) {
    e.preventDefault();
    const gpuCount = parseInt(document.getElementById("job-gpu-count").value, 10);
    const vram = parseFloat(document.getElementById("job-vram").value);
    const runtime = parseFloat(document.getElementById("job-runtime").value);
    const workloadType = document.getElementById("job-type").value;

    const payload = {
        gpu_count: gpuCount,
        vram_per_gpu_gb: vram,
        estimated_runtime: runtime,
        priority: 7,
        workload_type: workloadType,
        deadline_slack: 2.0,
    };

    try {
        const res = await fetch(`${API_BASE}/api/job/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        latestDecisionText.textContent = `Injected Custom Job #${data.job_id} (${gpuCount}x GPUs, ${vram}GB).`;
        await fetchStatus();
    } catch (err) {
        console.error("Submit error:", err);
    }
}

function toggleAutoPlay() {
    isAutoPlaying = !isAutoPlaying;
    if (isAutoPlaying) {
        btnAuto.classList.add("active");
        autoText.textContent = "Pause ⏸";
        autoPlayInterval = setInterval(async () => {
            await stepSimulation();
        }, 750);
    } else {
        btnAuto.classList.remove("active");
        autoText.textContent = "Auto Play ▶";
        if (autoPlayInterval) {
            clearInterval(autoPlayInterval);
            autoPlayInterval = null;
        }
    }
}

// --- Render Engine ---

function renderDashboard(data) {
    // Header & Summary
    simTimeEl.textContent = `${data.simulation_time.toFixed(1)}s`;
    const activeGpus = data.total_gpus - data.available_gpus;
    clusterSummaryChip.innerHTML = `<span class="dot-active"></span> ${activeGpus} / ${data.total_gpus} GPUs Active`;

    // Quick Stats
    statGpuUtil.textContent = `${data.cluster_gpu_utilization_pct.toFixed(0)}%`;
    const metrics = data.metrics || {};
    statThroughput.textContent = (metrics.throughput_jobs_per_hour || 0).toFixed(0);
    statCompleted.textContent = metrics.completed_jobs_count || 0;

    // Render Node Chassis Cards
    renderNodesDiagram(data.nodes);

    // Render Queue Cards
    renderQueueCards(data.queue);
}

function renderNodesDiagram(nodes) {
    nodesContainer.innerHTML = "";
    nodes.forEach(node => {
        const chassis = document.createElement("div");
        chassis.className = `node-chassis ${node.node_id === lastScheduledNodeId ? 'flash-highlight' : ''}`;

        // Physical GPU chips
        let chipsHtml = "";
        node.gpus.forEach(gpu => {
            const isBusy = !gpu.is_free;
            const flashChip = node.node_id === lastScheduledNodeId && isBusy;
            chipsHtml += `
                <div class="gpu-chip ${isBusy ? 'busy' : 'free'} ${flashChip ? 'flash-chip' : ''}">
                    <div class="chip-top">
                        <span>GPU #${gpu.gpu_id}</span>
                        <span>${gpu.total_vram_gb.toFixed(0)} GB</span>
                    </div>
                    <div class="chip-state">
                        ${isBusy ? `⚡ Job #${gpu.running_job_id}` : '⚪ Available'}
                    </div>
                </div>
            `;
        });

        // Running Tasks drawer
        let runningTasksHtml = "";
        if (node.running_jobs && node.running_jobs.length > 0) {
            runningTasksHtml = `<div class="running-jobs-drawer">`;
            node.running_jobs.forEach(rj => {
                runningTasksHtml += `
                    <div class="running-task-pill">
                        <div>
                            <strong>Job #${rj.job_id}</strong>
                            <span class="text-muted">(${rj.workload_type} · ${rj.gpu_count}x GPUs)</span>
                        </div>
                        <span class="text-muted font-mono">${rj.elapsed_sec.toFixed(0)}s / ${rj.total_runtime_sec.toFixed(0)}s (${rj.progress_pct.toFixed(0)}%)</span>
                    </div>
                `;
            });
            runningTasksHtml += `</div>`;
        }

        chassis.innerHTML = `
            <div class="chassis-header">
                <div class="node-label">
                    <span class="node-id-badge">Node ${node.node_id}</span>
                    <span class="node-model-tag">${node.gpu_type}</span>
                </div>
                <span class="node-status-online">● Online (${node.available_gpus} / ${node.total_gpus} Free)</span>
            </div>

            <div class="chassis-meters">
                <div class="meter-box">
                    <div class="meter-header">
                        <span>VRAM Allocated</span>
                        <span class="meter-val">${node.allocated_vram_gb.toFixed(0)} / ${node.total_vram_gb.toFixed(0)} GB (${node.vram_utilization_pct.toFixed(0)}%)</span>
                    </div>
                    <div class="meter-bar-bg"><div class="meter-bar-fill fill-purple" style="width: ${node.vram_utilization_pct}%;"></div></div>
                </div>

                <div class="meter-box">
                    <div class="meter-header">
                        <span>GPU Core Load</span>
                        <span class="meter-val">${node.total_gpus - node.available_gpus} / ${node.total_gpus} GPUs (${node.gpu_utilization_pct.toFixed(0)}%)</span>
                    </div>
                    <div class="meter-bar-bg"><div class="meter-bar-fill fill-blue" style="width: ${node.gpu_utilization_pct}%;"></div></div>
                </div>
            </div>

            <div class="gpu-chips-diagram">
                ${chipsHtml}
            </div>

            ${runningTasksHtml}
        `;

        nodesContainer.appendChild(chassis);
    });
}

function renderQueueCards(queue) {
    queueCountEl.textContent = queue.length;
    if (queue.length === 0) {
        queueContainer.innerHTML = `<div class="empty-queue-msg">Queue is empty.</div>`;
        queueMeanWait.textContent = `Avg Wait: 0.0s`;
        return;
    }

    const meanWait = queue.reduce((acc, j) => acc + j.waiting_time, 0) / queue.length;
    queueMeanWait.textContent = `Avg Wait: ${meanWait.toFixed(1)}s`;

    let html = "";
    queue.forEach((job, idx) => {
        html += `
            <div class="job-queue-card ${job.is_urgent ? 'urgent' : ''}">
                <div class="job-card-top">
                    <span class="job-card-id">#${idx + 1} · Job #${job.job_id}</span>
                    <span class="job-type-pill pill-${job.workload_type}">${job.workload_type}</span>
                </div>
                <div class="job-card-specs">
                    <span>⚡ ${job.gpu_count}x GPUs</span>
                    <span>💾 ${job.vram_per_gpu_gb.toFixed(0)} GB</span>
                    <span>⏱️ ${job.estimated_runtime.toFixed(0)}s</span>
                </div>
                <div class="job-card-footer">
                    <span>Waiting: <strong>${job.waiting_time.toFixed(1)}s</strong></span>
                    <span>Priority: <strong>★ ${job.priority}</strong></span>
                </div>
            </div>
        `;
    });
    queueContainer.innerHTML = html;
}

// --- Event Listeners ---

btnStep.addEventListener("click", stepSimulation);
btnAuto.addEventListener("click", toggleAutoPlay);
btnReset.addEventListener("click", resetCluster);
scenarioSelect.addEventListener("change", resetCluster);
jobForm.addEventListener("submit", submitJob);

window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        stepSimulation();
    }
});

// Initial load
fetchStatus();
setInterval(() => {
    if (!isAutoPlaying) {
        fetchStatus();
    }
}, 3000);

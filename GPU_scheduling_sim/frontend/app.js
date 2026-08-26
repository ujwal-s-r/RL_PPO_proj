/**
 * GPU Cluster Scheduler — Real-Time Interactive Frontend Controller
 */

const API_BASE = window.location.origin;

let autoPlayInterval = null;
let isAutoPlaying = false;

// DOM Elements
const simTimeEl = document.getElementById("sim-time");
const policySelect = document.getElementById("policy-select");
const scenarioSelect = document.getElementById("scenario-select");
const btnStep = document.getElementById("btn-step");
const btnAuto = document.getElementById("btn-auto");
const autoText = document.getElementById("auto-text");
const btnReset = document.getElementById("btn-reset");
const btnClearLogs = document.getElementById("btn-clear-logs");

const metricGpuUtil = document.getElementById("metric-gpu-util");
const metricGpuCounts = document.getElementById("metric-gpu-counts");
const barGpuUtil = document.getElementById("bar-gpu-util");

const metricVramUtil = document.getElementById("metric-vram-util");
const metricVramCounts = document.getElementById("metric-vram-counts");
const barVramUtil = document.getElementById("bar-vram-util");

const metricThroughput = document.getElementById("metric-throughput");
const metricCompleted = document.getElementById("metric-completed");
const metricJct = document.getElementById("metric-jct");
const metricDeadlineMiss = document.getElementById("metric-deadline-miss");
const barDeadline = document.getElementById("bar-deadline");

const nodesContainer = document.getElementById("nodes-container");
const clusterSummaryPill = document.getElementById("cluster-summary-pill");

const queueCountEl = document.getElementById("queue-count");
const queueWaitSummary = document.getElementById("queue-wait-summary");
const queueTbody = document.getElementById("queue-tbody");

const consoleLogs = document.getElementById("console-logs");
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
    const priority = parseInt(document.getElementById("job-priority").value, 10);
    const workloadType = document.getElementById("job-type").value;

    const payload = {
        gpu_count: gpuCount,
        vram_per_gpu_gb: vram,
        estimated_runtime: runtime,
        priority: priority,
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
        }, 700);
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
    // 1. Header Info
    simTimeEl.textContent = `${data.simulation_time.toFixed(1)}s`;

    // 2. Metrics HUD
    metricGpuUtil.textContent = `${data.cluster_gpu_utilization_pct.toFixed(1)}%`;
    barGpuUtil.style.width = `${Math.min(100, data.cluster_gpu_utilization_pct)}%`;
    const activeGpus = data.total_gpus - data.available_gpus;
    metricGpuCounts.textContent = `${activeGpus} / ${data.total_gpus} GPUs Active`;

    metricVramUtil.textContent = `${data.cluster_vram_utilization_pct.toFixed(1)}%`;
    barVramUtil.style.width = `${Math.min(100, data.cluster_vram_utilization_pct)}%`;
    const totalVram = data.nodes.reduce((acc, n) => acc + n.total_vram_gb, 0);
    const allocVram = data.nodes.reduce((acc, n) => acc + n.allocated_vram_gb, 0);
    metricVramCounts.textContent = `${allocVram.toFixed(0)} GB / ${totalVram.toFixed(0)} GB`;

    const metrics = data.metrics || {};
    metricThroughput.textContent = (metrics.throughput_jobs_per_hour || 0).toFixed(1);
    metricCompleted.textContent = metrics.completed_jobs_count || 0;
    metricJct.textContent = `Mean JCT: ${(metrics.mean_turnaround_time_seconds || 0).toFixed(1)}s`;

    const missPct = (metrics.deadline_violation_rate || 0) * 100;
    metricDeadlineMiss.textContent = `${missPct.toFixed(1)}%`;
    barDeadline.style.width = `${Math.min(100, missPct)}%`;

    clusterSummaryPill.textContent = `${data.nodes.length} Nodes · ${data.total_gpus} GPUs Total (${data.scenario.toUpperCase()})`;

    // 3. Render Nodes
    renderNodes(data.nodes);

    // 4. Render Queue
    renderQueue(data.queue);

    // 5. Render Logs
    renderLogs(data.recent_logs);
}

function renderNodes(nodes) {
    nodesContainer.innerHTML = "";
    nodes.forEach(node => {
        const nodeCard = document.createElement("div");
        nodeCard.className = "node-card";

        // Cores markup
        let coresHtml = "";
        node.gpus.forEach(gpu => {
            const isBusy = !gpu.is_free;
            coresHtml += `
                <div class="gpu-core ${isBusy ? 'busy' : 'free'}">
                    <div class="gpu-core-top">
                        <span>GPU ${gpu.gpu_id}</span>
                        <span>${gpu.total_vram_gb.toFixed(0)}G</span>
                    </div>
                    <div class="gpu-core-status">
                        ${isBusy ? `⚡ Job #${gpu.running_job_id}` : '● FREE'}
                    </div>
                </div>
            `;
        });

        // Running Jobs markup
        let jobsHtml = "";
        if (node.running_jobs && node.running_jobs.length > 0) {
            jobsHtml = `<div class="node-running-jobs">`;
            node.running_jobs.forEach(rj => {
                jobsHtml += `
                    <div class="running-job-item">
                        <div class="running-job-meta">
                            <strong>Job #${rj.job_id}</strong>
                            <span class="workload-badge workload-${rj.workload_type}">${rj.workload_type}</span>
                            <span class="text-muted">(${rj.gpu_count}x GPUs · ${rj.vram_gb}GB)</span>
                        </div>
                        <span class="text-muted small">${rj.elapsed_sec.toFixed(0)}s / ${rj.total_runtime_sec.toFixed(0)}s (${rj.progress_pct.toFixed(0)}%)</span>
                    </div>
                `;
            });
            jobsHtml += `</div>`;
        }

        nodeCard.innerHTML = `
            <div class="node-header">
                <div class="node-name">
                    <span>Node ${node.node_id}</span>
                    <span class="node-type-badge">${node.gpu_type}</span>
                </div>
                <div class="node-vram-text">
                    ${node.allocated_vram_gb.toFixed(0)} / ${node.total_vram_gb.toFixed(0)} GB VRAM (${node.vram_utilization_pct.toFixed(0)}%)
                </div>
            </div>
            <div class="progress-bar-bg"><div class="progress-bar-fill fill-purple" style="width: ${node.vram_utilization_pct}%;"></div></div>
            <div class="gpu-cores-grid">
                ${coresHtml}
            </div>
            ${jobsHtml}
        `;

        nodesContainer.appendChild(nodeCard);
    });
}

function renderQueue(queue) {
    queueCountEl.textContent = queue.length;
    if (queue.length === 0) {
        queueTbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted" style="padding: 24px;">Queue is currently empty.</td></tr>`;
        queueWaitSummary.textContent = `Mean Wait: 0.0s`;
        return;
    }

    const meanWait = queue.reduce((acc, j) => acc + j.waiting_time, 0) / queue.length;
    queueWaitSummary.textContent = `Mean Wait: ${meanWait.toFixed(1)}s`;

    let html = "";
    queue.forEach(job => {
        const isUrgent = job.is_urgent;
        html += `
            <tr class="${isUrgent ? 'urgent-row' : ''}">
                <td><strong>#${job.slot_index}</strong></td>
                <td><strong>Job #${job.job_id}</strong> ${isUrgent ? '<span title="Urgent Deadline">🔥</span>' : ''}</td>
                <td><span class="workload-badge workload-${job.workload_type}">${job.workload_type}</span></td>
                <td>${job.gpu_count}x</td>
                <td>${job.vram_per_gpu_gb.toFixed(0)} GB</td>
                <td>${job.estimated_runtime.toFixed(0)}s</td>
                <td><strong>★ ${job.priority}</strong></td>
                <td>${job.waiting_time.toFixed(1)}s</td>
            </tr>
        `;
    });
    queueTbody.innerHTML = html;
}

function renderLogs(logs) {
    if (!logs || logs.length === 0) return;
    consoleLogs.innerHTML = "";
    logs.forEach(log => {
        const logEl = document.createElement("div");
        logEl.className = `log-entry ${log.level || 'info'}`;
        logEl.innerHTML = `<span class="log-time">[${log.timestamp.toFixed(1)}s]</span> ${log.message}`;
        consoleLogs.appendChild(logEl);
    });
}

// --- Event Listeners ---

btnStep.addEventListener("click", stepSimulation);
btnAuto.addEventListener("click", toggleAutoPlay);
btnReset.addEventListener("click", resetCluster);
scenarioSelect.addEventListener("change", resetCluster);
jobForm.addEventListener("submit", submitJob);

btnClearLogs.addEventListener("click", () => {
    consoleLogs.innerHTML = `<div class="log-entry system"><span class="log-time">[${simTimeEl.textContent}]</span> Log cleared by user.</div>`;
});

// Keyboard Shortcut: Spacebar steps simulation
window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        stepSimulation();
    }
});

// Initial load & periodic background refresh
fetchStatus();
setInterval(() => {
    if (!isAutoPlaying) {
        fetchStatus();
    }
}, 3000);

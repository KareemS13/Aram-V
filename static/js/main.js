/* ===================================================================
   Armenia CPI Dashboard — main.js
   =================================================================== */

const HISTORY_MONTHS = 48;   // how many months of history to show in chart

// ── BOOT ──────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  initNavigation();
  initRunModal();
  await loadAll();
});

// ── LOAD ALL DATA ──────────────────────────────────────────────────────
async function loadAll() {
  try {
    const [history, forecast, drivers, metrics] = await Promise.all([
      fetchJSON("/api/history"),
      fetchJSON("/api/forecast"),
      fetchJSON("/api/drivers"),
      fetchJSON("/api/metrics"),
    ]);

    if (!forecast || forecast.error) return showNoData();

    renderKPIs(history, forecast, drivers, metrics);
    renderForecastChart(history, forecast);
    renderYoYChart(history, forecast);
    renderForecastTable(forecast);
    renderDriverBars(drivers);
    renderMetricsTable(metrics);
    updateSidebarMeta(forecast);

  } catch (e) {
    console.error("loadAll failed:", e);
    showNoData();
  }
}

async function fetchJSON(url) {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return await r.json();
  } catch { return null; }
}

function showNoData() {
  document.getElementById("sidebarMeta").textContent = "No data — run the pipeline first.";
}

// ── KPI CARDS ──────────────────────────────────────────────────────────
function renderKPIs(history, forecast, drivers, metrics) {
  // Last observed
  if (history && history.length) {
    const last = history[history.length - 1];
    const val  = last.value;
    document.getElementById("kpiLast").textContent = fmt(val) + "%";
    document.getElementById("kpiLast").className   = "kpi-value " + valClass(val);
    document.getElementById("kpiLastDate").textContent =
      new Date(last.date).toLocaleDateString("en-US", { month: "long", year: "numeric" });
  }

  // Forecast range
  if (forecast && forecast.length) {
    const pts  = forecast.map(r => r.point);
    const mn   = Math.min(...pts), mx = Math.max(...pts);
    document.getElementById("kpiFcRange").textContent = `${fmt(mn)}% → ${fmt(mx)}%`;
  }

  // Top driver
  if (drivers && drivers.length) {
    const top = drivers[0];
    document.getElementById("kpiDriver").textContent = top.group;
    const dir = top.direction === "inflationary";
    const dirEl = document.getElementById("kpiDriverDir");
    dirEl.textContent = (dir ? "▲ " : "▼ ") + top.direction + " · |SHAP| " + top.mean_abs_shap.toFixed(2) + " pp";
    dirEl.style.color  = dir ? "var(--red)" : "var(--green)";
  }

  // Model weights — derived from CV metrics (lower MAE = higher weight)
  if (metrics && metrics.length) {
    const sarima = metrics.filter(m => m.model === "SARIMA");
    const gbm    = metrics.filter(m => m.model === "GBM");
    if (sarima.length && gbm.length) {
      const maeS = sarima.reduce((s, r) => s + r.MAE, 0) / sarima.length;
      const maeG = gbm.reduce((s,   r) => s + r.MAE, 0) / gbm.length;
      const wS   = 1/maeS, wG = 1/maeG, tot = wS + wG;
      document.getElementById("kpiWeights").textContent =
        `${(wS/tot*100).toFixed(0)}% / ${(wG/tot*100).toFixed(0)}%`;
    }
  }
}

// ── FORECAST FAN CHART (Plotly) ────────────────────────────────────────
function renderForecastChart(history, forecast) {
  const histSlice = history ? history.slice(-HISTORY_MONTHS) : [];

  const histX = histSlice.map(r => r.date);
  const histY = histSlice.map(r => r.value);

  const fcX     = forecast.map(r => r.date);
  const ptY     = forecast.map(r => r.point);
  const lo95Y   = forecast.map(r => r.lower_95);
  const hi95Y   = forecast.map(r => r.upper_95);
  const lo50Y   = forecast.map(r => r.lower_50);
  const hi50Y   = forecast.map(r => r.upper_50);

  // Vertical forecast-start line
  const fstart  = fcX[0];
  const yMin    = Math.min(...histY, ...lo95Y);
  const yMax    = Math.max(...histY, ...hi95Y);

  const traces = [
    // 95% CI band (upper + lower filled)
    {
      x: [...fcX, ...fcX.slice().reverse()],
      y: [...hi95Y, ...lo95Y.slice().reverse()],
      fill: "toself",
      fillcolor: "rgba(25,118,210,0.10)",
      line: { color: "transparent" },
      name: "95% CI",
      hoverinfo: "skip",
      showlegend: true,
      type: "scatter",
    },
    // 50% CI band
    {
      x: [...fcX, ...fcX.slice().reverse()],
      y: [...hi50Y, ...lo50Y.slice().reverse()],
      fill: "toself",
      fillcolor: "rgba(25,118,210,0.22)",
      line: { color: "transparent" },
      name: "50% CI",
      hoverinfo: "skip",
      showlegend: true,
      type: "scatter",
    },
    // Historical line
    {
      x: histX,
      y: histY,
      mode: "lines",
      line: { color: "#455a64", width: 2 },
      name: "Historical CPI MoM%",
      type: "scatter",
      hovertemplate: "%{x}<br>CPI MoM%: <b>%{y:.2f}%</b><extra></extra>",
    },
    // Forecast line
    {
      x: fcX,
      y: ptY,
      mode: "lines+markers",
      line: { color: "#1565c0", width: 2.5, dash: "dot" },
      marker: { size: 5, color: "#1565c0" },
      name: "Ensemble Forecast",
      type: "scatter",
      hovertemplate: "%{x}<br>Forecast: <b>%{y:.2f}%</b><extra></extra>",
    },
    // Zero reference
    {
      x: [...histX, ...fcX],
      y: Array(histX.length + fcX.length).fill(0),
      mode: "lines",
      line: { color: "rgba(0,0,0,0.25)", width: 1, dash: "dot" },
      hoverinfo: "skip",
      showlegend: false,
      type: "scatter",
    },
  ];

  const layout = {
    margin: { t: 16, r: 20, b: 48, l: 52 },
    paper_bgcolor: "transparent",
    plot_bgcolor:  "transparent",
    font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size: 12 },
    xaxis: {
      showgrid: false,
      showline: true,
      linecolor: "#dce6f0",
      tickformat: "%b %Y",
      tickfont: { size: 11, color: "#5a6e82" },
    },
    yaxis: {
      gridcolor: "#edf2f7",
      gridwidth: 1,
      showline: false,
      ticksuffix: "%",
      tickfont: { size: 11, color: "#5a6e82" },
    },
    legend: {
      orientation: "h",
      y: -0.18,
      x: 0.5,
      xanchor: "center",
      font: { size: 12 },
    },
    shapes: [{
      type: "line",
      x0: fstart, x1: fstart,
      y0: yMin - 0.2, y1: yMax + 0.2,
      line: { color: "#e53935", width: 1.5, dash: "dash" },
    }],
    annotations: [{
      x: fstart,
      y: yMax + 0.1,
      text: "Forecast start",
      showarrow: false,
      font: { size: 11, color: "#e53935" },
      xanchor: "left",
      yanchor: "bottom",
    }],
    hovermode: "x unified",
  };

  Plotly.newPlot("forecastChart", traces, layout, {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  });
}

// ── YoY CHART ──────────────────────────────────────────────────────────
function renderYoYChart(history, forecast) {
  // Build combined MoM% series: enough history to roll 12-month window
  const allHist = history || [];

  // We need at least 11 months of history before the forecast starts
  // to compute YoY for the first forecast month
  // Combine history + forecast into one series for rolling window
  const combined = [
    ...allHist.map(r => ({ date: r.date, mom: r.value })),
    ...forecast.map(r => ({ date: r.date, mom: r.point })),
  ];

  function rollingYoY(series) {
    const out = [];
    for (let i = 11; i < series.length; i++) {
      let product = 1;
      for (let j = i - 11; j <= i; j++) product *= (1 + series[j].mom / 100);
      out.push({ date: series[i].date, yoy: (product - 1) * 100 });
    }
    return out;
  }

  const yoyAll = rollingYoY(combined);

  const fcStart      = forecast[0].date;
  const yoyHist      = yoyAll.filter(r => r.date <  fcStart).slice(-HISTORY_MONTHS);
  const yoyFc        = yoyAll.filter(r => r.date >= fcStart);

  // How many forecast months are already "inside" the 12-month window
  // for the first forecast date — used for the annotation
  const fcIdx0 = combined.findIndex(r => r.date === fcStart);
  const histMonthsInWindow = Math.max(0, 12 - (combined.length - fcIdx0));

  const fstart = yoyFc.length ? yoyFc[0].date : null;
  // Bridge point: last historical value prepended to forecast so lines connect
  const bridge = yoyHist.length ? [yoyHist[yoyHist.length - 1]] : [];

  const allYoY = [...yoyHist.map(r => r.yoy), ...yoyFc.map(r => r.yoy)];
  const yMin   = Math.min(...allYoY) - 0.5;
  const yMax   = Math.max(...allYoY) + 0.5;

  const fcX    = yoyFc.map(r => r.date);

  const traces = [
    // Historical YoY line
    {
      x: yoyHist.map(r => r.date),
      y: yoyHist.map(r => r.yoy),
      mode: "lines",
      line: { color: "#455a64", width: 2 },
      name: "Historical YoY%",
      type: "scatter",
      hovertemplate: "%{x}<br>YoY: <b>%{y:.2f}%</b><extra></extra>",
    },
    // Forecast YoY line — starts from last historical point so lines connect
    {
      x: [...bridge.map(r => r.date), ...fcX],
      y: [...bridge.map(r => r.yoy), ...yoyFc.map(r => r.yoy)],
      mode: "lines+markers",
      line: { color: "#1565c0", width: 2.5, dash: "dot" },
      marker: { size: 5, color: "#1565c0",
                // Hide the bridge point marker — only show forecast markers
                line: { width: 0 } },
      name: "Forecast YoY%",
      type: "scatter",
      hovertemplate: "%{x}<br>Forecast YoY: <b>%{y:.2f}%</b><extra></extra>",
    },
    // Zero reference
    {
      x: [...yoyHist.map(r => r.date), ...fcX],
      y: Array(yoyHist.length + fcX.length).fill(0),
      mode: "lines",
      line: { color: "rgba(0,0,0,0.20)", width: 1, dash: "dot" },
      hoverinfo: "skip",
      showlegend: false,
      type: "scatter",
    },
  ];

  const layout = {
    margin: { t: 16, r: 20, b: 48, l: 52 },
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", size: 12 },
    xaxis: {
      showgrid: false,
      showline: true,
      linecolor: "#dce6f0",
      tickformat: "%b %Y",
      tickfont: { size: 11, color: "#5a6e82" },
    },
    yaxis: {
      gridcolor: "#edf2f7",
      gridwidth: 1,
      showline: false,
      ticksuffix: "%",
      tickfont: { size: 11, color: "#5a6e82" },
    },
    legend: {
      orientation: "h",
      y: -0.18,
      x: 0.5,
      xanchor: "center",
      font: { size: 12 },
    },
    shapes: fstart ? [{
      type: "line",
      x0: fstart, x1: fstart,
      y0: yMin, y1: yMax,
      line: { color: "#e53935", width: 1.5, dash: "dash" },
    }] : [],
    annotations: fstart ? [
      {
        x: fstart,
        y: yMax,
        text: "Forecast start",
        showarrow: false,
        font: { size: 11, color: "#e53935" },
        xanchor: "left",
        yanchor: "bottom",
      },
      {
        x: fcX[Math.floor(fcX.length / 2)] || fstart,
        y: yMin,
        text: "YoY window still includes prior actual months — converges fully by Feb 2027",
        showarrow: false,
        font: { size: 10, color: "#888" },
        xanchor: "center",
        yanchor: "top",
      },
    ] : [],
    hovermode: "x unified",
  };

  Plotly.newPlot("yoyChart", traces, layout, {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
  });
}

// ── FORECAST TABLE ─────────────────────────────────────────────────────
function renderForecastTable(forecast) {
  // Pre-compute YoY for each forecast row using the rolling product
  // yoyByDate is populated by renderYoYChart; here we recompute inline
  // using just the forecast MoM% values (approximation for table display)
  const yoyMap = {};
  for (let i = 0; i < forecast.length; i++) {
    let product = 1;
    // Use as many forecast months as available (up to 12)
    const start = Math.max(0, i - 11);
    for (let j = start; j <= i; j++) product *= (1 + forecast[j].point / 100);
    // Only meaningful once we have 12 months; mark partial with ~
    yoyMap[forecast[i].date] = { val: (product - 1) * 100, partial: (i < 11) };
  }

  const tbody = document.querySelector("#forecastTable tbody");
  tbody.innerHTML = forecast.map(r => {
    const d      = new Date(r.date);
    const mo     = d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
    const cls    = valClass(r.point);
    const yoy    = yoyMap[r.date];
    const yoyStr = yoy ? (yoy.partial ? "~" : "") + fmt(yoy.val) + "%" : "—";
    const yoyCls = yoy ? valClass(yoy.val) : "neutral";
    return `
      <tr>
        <td><strong>${mo}</strong></td>
        <td class="${cls}">${fmt(r.point)}%</td>
        <td class="${yoyCls}">${yoyStr}</td>
        <td class="neutral">${fmt(r.lower_50)}% to ${fmt(r.upper_50)}%</td>
        <td class="neutral">${fmt(r.lower_95)}% to ${fmt(r.upper_95)}%</td>
      </tr>`;
  }).join("");
}

// ── DRIVER BARS ────────────────────────────────────────────────────────
function renderDriverBars(drivers) {
  if (!drivers) return;
  const maxShap = Math.max(...drivers.map(d => d.mean_abs_shap));
  const container = document.getElementById("driverBars");
  container.innerHTML = drivers.map(d => {
    const pct = (d.mean_abs_shap / maxShap * 100).toFixed(1);
    const inf = d.direction === "inflationary";
    const barCls = inf ? "bar-inf" : "bar-dis";
    const tagCls = inf ? "tag-inf" : "tag-dis";
    const tagText = inf ? "Inflationary" : "Disinflationary";
    return `
      <div class="driver-row">
        <div class="driver-name">${d.group}</div>
        <div class="bar-track">
          <div class="bar-fill ${barCls}" style="width:${pct}%"></div>
        </div>
        <div class="driver-shap">${d.mean_abs_shap.toFixed(3)}</div>
        <div><span class="tag ${tagCls}">${tagText}</span></div>
      </div>`;
  }).join("");
}

// ── METRICS TABLE ──────────────────────────────────────────────────────
function renderMetricsTable(metrics) {
  if (!metrics) return;
  const tbody = document.querySelector("#metricsTable tbody");
  tbody.innerHTML = metrics.map(r => {
    const modelCls = r.model === "SARIMA" ? "color:#1565c0;font-weight:700"
                                          : "color:#e65100;font-weight:700";
    return `
      <tr>
        <td style="${modelCls}">${r.model}</td>
        <td>${r.horizon}m</td>
        <td>${r.MAE.toFixed(3)}</td>
        <td>${r.RMSE.toFixed(3)}</td>
        <td>${r.MAPE.toFixed(1)}</td>
      </tr>`;
  }).join("");
}

// ── SIDEBAR META ───────────────────────────────────────────────────────
function updateSidebarMeta(forecast) {
  if (!forecast || !forecast.length) return;
  const last = forecast[forecast.length - 1];
  const d = new Date(last.date);
  const mo = d.toLocaleDateString("en-US", { month: "short", year: "numeric" });
  document.getElementById("sidebarMeta").textContent = `Forecast through ${mo}`;
  document.getElementById("lastUpdated").textContent = "Latest run";
}

// ── NAVIGATION ─────────────────────────────────────────────────────────
function initNavigation() {
  const sections  = document.querySelectorAll(".section");
  const navItems  = document.querySelectorAll(".nav-item");

  // Click navigation
  navItems.forEach(item => {
    item.addEventListener("click", e => {
      e.preventDefault();
      const target = document.getElementById(item.dataset.section);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  // Scroll highlight
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        navItems.forEach(n => n.classList.remove("active"));
        const active = document.querySelector(`[data-section="${entry.target.id}"]`);
        if (active) active.classList.add("active");
      }
    });
  }, { threshold: 0.25 });

  sections.forEach(s => observer.observe(s));
}

// ── RUN MODAL ──────────────────────────────────────────────────────────
function initRunModal() {
  const overlay   = document.getElementById("modalOverlay");
  const btnRun    = document.getElementById("btnRun");
  const btnClose  = document.getElementById("modalClose");
  const btnStart  = document.getElementById("btnStart");
  const btnRefresh= document.getElementById("btnRefresh");
  const modalBody = document.getElementById("modalBody");
  const modalLog  = document.getElementById("modalLog");
  const logOutput = document.getElementById("logOutput");
  const logStatus = document.getElementById("logStatus");
  let pollTimer   = null;

  btnRun.addEventListener("click", () => {
    overlay.classList.add("open");
    modalBody.style.display = "block";
    modalLog.style.display  = "none";
    logOutput.textContent   = "";
    btnRefresh.style.display = "none";
  });

  btnClose.addEventListener("click", closeModal);
  overlay.addEventListener("click", e => { if (e.target === overlay) closeModal(); });

  btnStart.addEventListener("click", async () => {
    const noTune = document.getElementById("optNoTune").checked;
    const noCV   = document.getElementById("optNoCV").checked;

    modalBody.style.display = "none";
    modalLog.style.display  = "block";
    logStatus.textContent   = "Running…";
    logOutput.textContent   = "";
    btnRefresh.style.display = "none";

    await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ no_tune: noTune, no_cv: noCV }),
    });

    pollTimer = setInterval(pollStatus, 1000);
  });

  btnRefresh.addEventListener("click", () => {
    closeModal();
    loadAll();
  });

  async function pollStatus() {
    try {
      const r = await fetch("/api/status");
      const s = await r.json();

      logOutput.textContent = s.log.join("\n");
      logOutput.scrollTop   = logOutput.scrollHeight;

      if (!s.running) {
        clearInterval(pollTimer);
        if (s.error) {
          logStatus.textContent      = "Pipeline failed.";
          logStatus.style.color      = "#ef5350";
        } else {
          logStatus.textContent      = "Pipeline complete.";
          logStatus.style.color      = "#66bb6a";
        }
        btnRefresh.style.display = "inline-block";
      }
    } catch {}
  }

  function closeModal() {
    if (pollTimer) clearInterval(pollTimer);
    overlay.classList.remove("open");
  }
}

// ── HELPERS ────────────────────────────────────────────────────────────
function fmt(v) {
  if (v === null || v === undefined) return "—";
  return (v >= 0 ? "+" : "") + v.toFixed(2);
}

function valClass(v) {
  if (v > 0.05)  return "positive";
  if (v < -0.05) return "negative";
  return "neutral";
}

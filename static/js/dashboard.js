// dashboard.js
document.addEventListener("DOMContentLoaded", () => {
  const trainBtn = document.getElementById("trainBtn");
  const trainProgress = document.getElementById("trainProgress");
  const trainMsg = document.getElementById("trainMsg");

  async function pollStatus() {
    try {
      const res = await fetch("/train_status");
      const data = await res.json();
      const progressLine = document.getElementById("trainProgressLine");
      if (progressLine) {
        progressLine.style.width = data.progress + "%";
      }
      trainProgress.innerText = data.progress + "%";
      trainMsg.innerText = data.message || "";
      return data;
    } catch (e) {
      console.error(e);
      return null;
    }
  }

  trainBtn.addEventListener("click", async () => {
    trainBtn.disabled = true;
    const start = await fetch("/train_model");
    if (!start.ok && start.status !== 202) {
      alert("Failed to start training");
      trainBtn.disabled = false;
      return;
    }
    trainMsg.innerText = "Training started...";
    // poll until progress==100 or not running
    const t = setInterval(async () => {
      const s = await pollStatus();
      if (s && s.progress >= 100) {
        clearInterval(t);
        trainBtn.disabled = false;
        alert("Training completed");
      }
    }, 1500);
  });

  // Chart initial render & update every 10s
  let chart = null;
  let chartDataCache = null;
  const chartViewFilter = document.getElementById("chartViewFilter");

  function renderChartData() {
      if (!chartDataCache || !document.getElementById("attendanceChart")) return;
      
      const filter = chartViewFilter ? chartViewFilter.value : "present";
      
      // Determine what to show
      const labelName = filter === "present" ? "Present" : "Absent";
      const dataArray = filter === "present" ? chartDataCache.counts : chartDataCache.absent_counts;
      const totalStudents = chartDataCache.total_students || 1;
      
      const borderColor = filter === "present" ? "#3b82f6" : "#ef4444";
      const bgColor = filter === "present" ? "rgba(59,130,246,0.15)" : "rgba(239,68,68,0.15)";
      
      const ctx = document.getElementById("attendanceChart").getContext("2d");
      
      if (!chart) {
          chart = new Chart(ctx, {
              type: "line",
              data: {
                  labels: chartDataCache.dates,
                  datasets: [{ 
                      label: labelName, 
                      data: dataArray, 
                      borderColor: borderColor,
                      backgroundColor: bgColor,
                      borderWidth: 2,
                      fill: true,
                      tension: 0.4,
                      pointBackgroundColor: "#ffffff",
                      pointBorderColor: borderColor
                  }]
              },
              options: { 
                  responsive: true, 
                  maintainAspectRatio: false,
                  plugins: { 
                      legend: { display: false },
                      tooltip: {
                          callbacks: {
                              label: function(context) {
                                  let val = context.parsed.y;
                                  let pct = Math.round((val / totalStudents) * 100) || 0;
                                  return `${labelName}: ${val} (${pct}%)`;
                              }
                          }
                      }
                  },
                  scales: { 
                      x: { grid: { display: false } },
                      y: { border: { dash: [4,4] }, grid: { color: "rgba(0,0,0,0.05)" }, beginAtZero: true }
                  }
              }
          });
      } else {
          chart.data.labels = chartDataCache.dates;
          chart.data.datasets[0].label = labelName;
          chart.data.datasets[0].data = dataArray;
          chart.data.datasets[0].borderColor = borderColor;
          chart.data.datasets[0].backgroundColor = bgColor;
          chart.data.datasets[0].pointBorderColor = borderColor;
          chart.options.plugins.tooltip.callbacks.label = function(context) {
                let val = context.parsed.y;
                let pct = Math.round((val / totalStudents) * 100) || 0;
                return `${labelName}: ${val} (${pct}%)`;
          };
          chart.update();
      }
  }

  async function fetchAndUpdateChart() {
    if(!document.getElementById("attendanceChart")) return;
    try {
        const res = await fetch("/attendance_stats");
        chartDataCache = await res.json();
        renderChartData();
    } catch(e) { console.error(e); }
  }

  fetchAndUpdateChart();
  setInterval(fetchAndUpdateChart, 10000);

  if(chartViewFilter) {
      chartViewFilter.addEventListener("change", renderChartData);
  }
  // Calendar Logic
  const calendarTitle = document.getElementById("calendarTitle");
  const calendarGrid = document.getElementById("calendarGrid");
  const prevMonthBtn = document.getElementById("prevMonthBtn");
  const nextMonthBtn = document.getElementById("nextMonthBtn");

  let currentDate = new Date();
  let selectedDateStr = new Date().toISOString().split('T')[0];

  const recentLogsTitle = document.getElementById("recentLogsTitle");
  const recentLogsContainer = document.getElementById("recentLogsContainer");

  async function fetchLogsForDate(dateStr) {
      if(!recentLogsContainer) return;
      try {
          const res = await fetch(`/api/logs_by_date?date=${dateStr}`);
          const data = await res.json();
          
          if(!data.logs || data.logs.length === 0) {
              recentLogsContainer.innerHTML = '<div class="text-center text-muted py-4">No scans found for this date</div>';
              return;
          }
          
          let html = "";
          data.logs.forEach((log, idx) => {
              html += `
                <div class="d-flex align-items-center justify-content-between pb-3 border-bottom">
                    <div class="d-flex align-items-center gap-3">
                        <span class="text-muted fw-bold small">${idx + 1}.</span>
                        <img src="/student_image/${log.student_id}" 
                             onerror="this.src='https://ui-avatars.com/api/?name=${encodeURIComponent(log.name)}&background=f1f5f9';" 
                             class="student-thumb rounded-circle shadow-sm" style="width: 40px; height: 40px; object-fit: cover;">
                        <span class="fw-bold">${log.name}</span>
                    </div>
                    <span class="text-muted small">${log.time_str}</span>
                </div>
              `;
          });
          recentLogsContainer.innerHTML = html;
      } catch(e) {
          console.error("Fetch logs error:", e);
      }
  }

  function renderCalendar() {
      if(!calendarGrid) return;
      calendarGrid.innerHTML = "";
      
      const year = currentDate.getFullYear();
      const month = currentDate.getMonth();
      const today = new Date();

      const monthNames = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
      calendarTitle.innerText = `${monthNames[month]} ${year}`;

      const firstDayOfMonth = new Date(year, month, 1).getDay();
      const daysInMonth = new Date(year, month + 1, 0).getDate();
      const daysInPrevMonth = new Date(year, month, 0).getDate();

      // Previous month's dates
      for (let i = firstDayOfMonth - 1; i >= 0; i--) {
          const div = document.createElement("div");
          div.className = "text-black-50 mb-2";
          div.style.width = "14%";
          div.innerText = daysInPrevMonth - i;
          calendarGrid.appendChild(div);
      }

      // Current month's dates
      for (let i = 1; i <= daysInMonth; i++) {
          const div = document.createElement("div");
          div.style.width = "14%";
          div.className = "mb-2 d-flex justify-content-center align-items-center";
          
          let innerSpan = document.createElement("span");
          innerSpan.innerText = i;
          innerSpan.style.cursor = "pointer";
          
          const loopDateStr = `${year}-${String(month+1).padStart(2,'0')}-${String(i).padStart(2,'0')}`;
          const isSelected = (loopDateStr === selectedDateStr);
          const isToday = (i === today.getDate() && month === today.getMonth() && year === today.getFullYear());
          
          if (isSelected) {
              innerSpan.className = "bg-primary text-white rounded-circle d-flex align-items-center justify-content-center fw-bold shadow-sm";
          } else if (isToday) {
              innerSpan.className = "border border-primary text-primary rounded-circle d-flex align-items-center justify-content-center fw-bold";
          } else {
              innerSpan.className = "d-flex justify-content-center align-items-center text-dark rounded-circle";
          }
          innerSpan.style.width = "28px";
          innerSpan.style.height = "28px";
          
          innerSpan.addEventListener("click", () => {
              selectedDateStr = loopDateStr;
              if (recentLogsTitle) {
                  const displayDate = new Date(year, month, i).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
                  recentLogsTitle.innerText = `Check-ins for ${displayDate}`;
              }
              renderCalendar();
              fetchLogsForDate(selectedDateStr);
          });
          
          div.appendChild(innerSpan);
          calendarGrid.appendChild(div);
      }

      // Next month's dates
      const totalCells = firstDayOfMonth + daysInMonth;
      const remainingCells = (7 - (totalCells % 7)) % 7;

      for (let i = 1; i <= remainingCells; i++) {
          const div = document.createElement("div");
          div.className = "text-black-50 mb-2";
          div.style.width = "14%";
          div.innerText = i;
          calendarGrid.appendChild(div);
      }
  }

  if (calendarGrid) {
      renderCalendar();
      fetchLogsForDate(selectedDateStr); // initial load
      prevMonthBtn.addEventListener("click", () => {
          currentDate.setMonth(currentDate.getMonth() - 1);
          renderCalendar();
      });
      nextMonthBtn.addEventListener("click", () => {
          currentDate.setMonth(currentDate.getMonth() + 1);
          renderCalendar();
      });
  }
});

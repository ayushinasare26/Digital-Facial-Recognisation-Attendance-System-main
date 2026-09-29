// static/js/admin/dashboard.js - Admin Portal Chart.js Visualizations

document.addEventListener("DOMContentLoaded", async () => {
    const trendCtx = document.getElementById("trendChart");
    const confCtx = document.getElementById("confidenceChart");

    if (!trendCtx || !confCtx) return;

    try {
        const resp = await fetch("/admin/dashboard/summary");
        if (!resp.ok) throw new Error("Failed to fetch dashboard summary");
        const data = await resp.json();

        // 1. Trend Chart (Verified vs Flagged over 14 days)
        new Chart(trendCtx.getContext("2d"), {
            type: "line",
            data: {
                labels: data.trend.labels,
                datasets: [
                    {
                        label: "Verified Attendances",
                        data: data.trend.success,
                        borderColor: "#4f46e5",
                        backgroundColor: "rgba(79, 70, 229, 0.12)",
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: "#ffffff",
                        pointBorderColor: "#4f46e5",
                        pointRadius: 4,
                        pointHoverRadius: 6
                    },
                    {
                        label: "Flagged Anomalies",
                        data: data.trend.flagged,
                        borderColor: "#ef4444",
                        backgroundColor: "rgba(239, 68, 68, 0.08)",
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: "#ffffff",
                        pointBorderColor: "#ef4444",
                        pointRadius: 3,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: "top",
                        labels: {
                            boxWidth: 12,
                            font: { family: "Inter", size: 12 }
                        }
                    },
                    tooltip: {
                        padding: 10,
                        backgroundColor: "#0f172a",
                        titleFont: { family: "Inter", size: 12 },
                        bodyFont: { family: "Inter", size: 12 }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: "Inter", size: 11 } }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: "rgba(0, 0, 0, 0.05)" },
                        ticks: { precision: 0, font: { family: "Inter", size: 11 } }
                    }
                }
            }
        });

        // 2. Confidence Distribution Histogram
        const binLabels = Object.keys(data.confidence_distribution);
        const binValues = Object.values(data.confidence_distribution);

        new Chart(confCtx.getContext("2d"), {
            type: "bar",
            data: {
                labels: binLabels,
                datasets: [{
                    label: "Events",
                    data: binValues,
                    backgroundColor: [
                        "rgba(239, 68, 68, 0.8)",
                        "rgba(245, 158, 11, 0.8)",
                        "rgba(234, 179, 8, 0.8)",
                        "rgba(59, 130, 246, 0.8)",
                        "rgba(99, 102, 241, 0.8)",
                        "rgba(16, 185, 129, 0.8)"
                    ],
                    borderRadius: 6,
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.parsed.y} check-in events`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { family: "Inter", size: 10 } }
                    },
                    y: {
                        beginAtZero: true,
                        ticks: { precision: 0, font: { family: "Inter", size: 10 } },
                        grid: { color: "rgba(0, 0, 0, 0.05)" }
                    }
                }
            }
        });

    } catch (err) {
        console.error("Dashboard chart error:", err);
    }
});

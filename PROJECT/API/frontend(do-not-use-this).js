class AnomalyDashboard {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.updateInterval = 5000; // 5 seconds
        this.ws = null;
    }

    async fetchLatestAnomalies() {
        try {
            const response = await fetch(`${this.apiUrl}/anomalies/latest`);
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            return data.data;
        } catch (error) {
            console.error('Error fetching anomalies:', error);
            return [];
        }
    }

    async fetchThreats() {
        try {
            const response = await fetch(`${this.apiUrl}/anomalies/threats`);
            if (!response.ok) throw new Error('Network response was not ok');
            const data = await response.json();
            return data.data;
        } catch (error) {
            console.error('Error fetching threats:', error);
            return [];
        }
    }

    startPolling() {
        // Poll for all anomalies
        setInterval(async () => {
            const anomalies = await this.fetchLatestAnomalies();
            this.updateDashboard(anomalies);
        }, this.updateInterval);

        // Poll for threats separately
        setInterval(async () => {
            const threats = await this.fetchThreats();
            this.updateThreatsDashboard(threats);
        }, this.updateInterval);
    }

    updateDashboard(anomalies) {
        const container = document.getElementById('anomalies-container');
        container.innerHTML = anomalies
            .map(a => `
                <div class="anomaly-card ${a.is_anomaly ? 'danger' : 'normal'}">
                    <p>Time: ${new Date(a.timestamp).toLocaleString()}</p>
                    <p>Status: ${a.is_anomaly ? '⚠️ THREAT' : 'Normal'}</p>
                    <p>MSE: ${a.mse.toFixed(2)}</p>
                    <details>
                        <summary>Original Data</summary>
                        <pre>${JSON.stringify(a.original_data, null, 2)}</pre>
                    </details>
                </div>
            `)
            .join('');
    }

    updateThreatsDashboard(threats) {
        const container = document.getElementById('threats-container');
        if (!container) return;
        
        container.innerHTML = threats
            .map(t => `
                <div class="threat-card">
                    <h3>⚠️ Threat Detected</h3>
                    <p>Time: ${new Date(t.timestamp).toLocaleString()}</p>
                    <p>MSE: ${t.mse.toFixed(2)}</p>
                    <details>
                        <summary>Threat Details</summary>
                        <pre>${JSON.stringify(t.original_data, null, 2)}</pre>
                    </details>
                </div>
            `)
            .join('');
    }
}

// Initialize dashboard
const dashboard = new AnomalyDashboard();
dashboard.startPolling();
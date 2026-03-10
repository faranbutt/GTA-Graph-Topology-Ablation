const LEADERBOARD_CONFIG = {
    csvPath: './leaderboard.csv',
    sortBy: 'validation_f1_score',
    primaryScoreField: 'validation_f1_score',
    fieldNames: {
        team_name: 'Team',
        validation_accuracy: 'Validation Accuracy',
        validation_f1_score: 'Validation F1 Score',
        timestamp: 'Submission Time',
    },
    fieldFormatters: {
        timestamp: (value) => {
            try {
                const date = new Date(value.replace(' ', 'T'));
                return date.toLocaleString('en-US', {
                    month: 'long',
                    day: 'numeric',
                    year: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false,
                }).replace(',', ' at');
            } catch (e) {
                return value;
            }
        },
        validation_accuracy: (value) => (Number.isFinite(value) ? value.toFixed(6) : value),
        validation_f1_score: (value) => (Number.isFinite(value) ? value.toFixed(6) : value),
    },
};

function parseCsv(text) {
    const lines = text.trim().split(/\r?\n/);
    if (!lines.length) return [];
    const headers = lines[0].split(',').map((h) => h.trim());
    return lines.slice(1).map((line) => {
        const values = line.split(',').map((v) => v.trim());
        const row = {};
        headers.forEach((header, idx) => {
            const raw = values[idx] ?? '';
            const num = Number(raw);
            row[header] = Number.isFinite(num) && raw !== '' ? num : raw;
        });
        return row;
    });
}

class LeaderboardManager {
    constructor(config) {
        this.config = config;
        this.data = [];
    }

    async loadData() {
        const response = await fetch(this.config.csvPath, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`Failed to load leaderboard data: ${response.statusText}`);
        }
        const text = await response.text();
        this.data = parseCsv(text);
        return this.data;
    }

    sortData() {
        if (!this.config.sortBy || !this.data.length) return;
        this.data.sort((a, b) => {
            const aVal = a[this.config.sortBy];
            const bVal = b[this.config.sortBy];
            if (aVal == null && bVal == null) return 0;
            if (aVal == null) return 1;
            if (bVal == null) return -1;
            if (typeof aVal === 'number' && typeof bVal === 'number') {
                return bVal - aVal;
            }
            return String(bVal).localeCompare(String(aVal));
        });
    }

    getColumns() {
        if (!this.data.length) return [];
        const columns = [];
        const firstItem = this.data[0];
        if ('team_name' in firstItem) columns.push('team_name');
        for (const key in firstItem) {
            if (key !== 'team_name') columns.push(key);
        }
        return columns;
    }

    formatValue(fieldName, value) {
        if (this.config.fieldFormatters[fieldName]) {
            return this.config.fieldFormatters[fieldName](value);
        }
        if (typeof value === 'number') return value.toFixed(6);
        return value;
    }

    getDisplayName(fieldName) {
        return this.config.fieldNames[fieldName] ||
            fieldName.split('_').map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
    }

    getFieldClass(fieldName) {
        const classes = [];
        if (fieldName === this.config.primaryScoreField) classes.push('score', 'primary-score');
        else if (typeof this.data[0]?.[fieldName] === 'number') classes.push('score');
        if (fieldName === 'team_name') classes.push('team-name');
        return classes.join(' ');
    }

    getMedal(rank) {
        if (rank === 1) return '🥇';
        if (rank === 2) return '🥈';
        if (rank === 3) return '🥉';
        return '';
    }

    getRankClass(rank) {
        if (rank <= 3) return `rank-${rank}`;
        return '';
    }

    render() {
        const tableHeader = document.getElementById('table-header');
        const tableBody = document.getElementById('table-body');
        const lastUpdated = document.getElementById('last-updated');

        if (!tableHeader || !tableBody) return;
        tableHeader.innerHTML = '';
        tableBody.innerHTML = '';

        if (!this.data || this.data.length === 0) {
            tableBody.innerHTML = '<tr><td colspan="100%" class="empty">No leaderboard data available</td></tr>';
            if (lastUpdated) lastUpdated.textContent = 'No data available';
            return;
        }

        this.sortData();
        const columns = this.getColumns();

        const headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th class="rank">Rank</th>';
        columns.forEach((column) => {
            const th = document.createElement('th');
            const displayName = this.getDisplayName(column);
            const fieldClass = this.getFieldClass(column);
            th.textContent = displayName;
            if (fieldClass) th.className = fieldClass;
            headerRow.appendChild(th);
        });
        tableHeader.appendChild(headerRow);

        this.data.forEach((entry, index) => {
            const rank = index + 1;
            const row = document.createElement('tr');
            row.style.animationDelay = `${(index + 1) * 0.1}s`;

            const rankCell = document.createElement('td');
            rankCell.className = `rank ${this.getRankClass(rank)}`;
            const medal = this.getMedal(rank);
            rankCell.innerHTML = `${medal ? `<span class="medal">${medal} </span>` : ''}${rank}`;
            row.appendChild(rankCell);

            columns.forEach((column) => {
                const cell = document.createElement('td');
                const value = entry[column];
                const fieldClass = this.getFieldClass(column);
                if (fieldClass) cell.className = fieldClass;
                cell.textContent = value != null ? this.formatValue(column, value) : '-';
                row.appendChild(cell);
            });

            tableBody.appendChild(row);
        });

        if (lastUpdated) {
            const now = new Date();
            const formattedDate = now.toLocaleString('en-US', {
                month: 'long',
                day: 'numeric',
                year: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
                hour12: false,
            }).replace(',', ' at');
            lastUpdated.textContent = `Last updated: ${formattedDate}`;
        }
    }

    async init() {
        try {
            await this.loadData();
            this.render();
        } catch (error) {
            const tableBody = document.getElementById('table-body');
            if (tableBody) {
                tableBody.innerHTML = `<tr><td colspan="100%" class="empty">Error loading leaderboard: ${error.message}</td></tr>`;
            }
        }
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const leaderboard = new LeaderboardManager(LEADERBOARD_CONFIG);
    leaderboard.init();
    initGraphBackground();
});

function initGraphBackground() {
    const canvas = document.getElementById('graph-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    function resizeCanvas() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    const nodes = [];
    const nodeCount = 80;
    const connectionDistance = 280;
    const minDistance = 120;
    const springForce = 0.0001;
    const repulsionForce = 0.01;
    const damping = 0.98;

    class Node {
        constructor() {
            this.initialX = Math.random() * canvas.width;
            this.initialY = Math.random() * canvas.height;
            this.x = this.initialX;
            this.y = this.initialY;
            this.vx = 0;
            this.vy = 0;
            this.radius = Math.random() * 2.5 + 1.5;
            this.baseHue = Math.random() * 60 + 240;
            this.hue = this.baseHue;
            this.brightness = 50 + Math.random() * 20;
            this.pulsePhase = Math.random() * Math.PI * 2;
            this.pulseSpeed = 0.01 + Math.random() * 0.01;
            this.oscillationPhase = Math.random() * Math.PI * 2;
            this.oscillationSpeed = 0.005 + Math.random() * 0.005;
            this.oscillationRadius = 2 + Math.random() * 3;
        }

        update(others) {
            const dx = this.initialX - this.x;
            const dy = this.initialY - this.y;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance > 0.1) {
                const spring = springForce * distance;
                this.vx += (dx / distance) * spring;
                this.vy += (dy / distance) * spring;
            }

            for (const other of others) {
                if (other === this) continue;
                const dx2 = other.x - this.x;
                const dy2 = other.y - this.y;
                const dist = Math.sqrt(dx2 * dx2 + dy2 * dy2);
                if (dist < minDistance && dist > 0.1) {
                    const force = repulsionForce / (dist * dist + 1);
                    this.vx -= (dx2 / dist) * force;
                    this.vy -= (dy2 / dist) * force;
                }
            }

            this.oscillationPhase += this.oscillationSpeed;
            const oscX = Math.cos(this.oscillationPhase) * this.oscillationRadius;
            const oscY = Math.sin(this.oscillationPhase * 1.3) * this.oscillationRadius;

            this.vx *= damping;
            this.vy *= damping;
            this.x = this.initialX + oscX + this.vx;
            this.y = this.initialY + oscY + this.vy;

            this.pulsePhase += this.pulseSpeed;
            this.hue = this.baseHue + Math.sin(this.pulsePhase * 0.5) * 10;
        }

        draw(ctx) {
            const glowRadius = this.radius * 4;
            const gradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, glowRadius);
            const pulse = Math.sin(this.pulsePhase) * 0.2 + 0.3;
            gradient.addColorStop(0, `hsla(${this.hue}, 80%, ${this.brightness}%, ${pulse * 0.6})`);
            gradient.addColorStop(0.5, `hsla(${this.hue}, 70%, ${this.brightness}%, ${pulse * 0.3})`);
            gradient.addColorStop(1, `hsla(${this.hue}, 60%, ${this.brightness}%, 0)`);
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(this.x, this.y, glowRadius, 0, Math.PI * 2);
            ctx.fill();

            const coreGradient = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius * 1.5);
            coreGradient.addColorStop(0, `hsl(${this.hue}, 90%, ${this.brightness + 10}%)`);
            coreGradient.addColorStop(1, `hsl(${this.hue}, 80%, ${this.brightness}%)`);
            ctx.fillStyle = coreGradient;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = `hsla(${this.hue}, 100%, 90%, 0.6)`;
            ctx.beginPath();
            ctx.arc(this.x - this.radius * 0.3, this.y - this.radius * 0.3, this.radius * 0.4, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    for (let i = 0; i < nodeCount; i++) nodes.push(new Node());

    function drawConnections(ctx) {
        const connections = [];
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);
                if (distance < connectionDistance) {
                    connections.push({
                        from: nodes[i],
                        to: nodes[j],
                        distance: distance,
                        strength: 1 - (distance / connectionDistance),
                    });
                }
            }
        }
        connections.sort((a, b) => b.distance - a.distance);
        connections.forEach((conn) => {
            const opacity = conn.strength * 0.4;
            const lineWidth = 0.5 + conn.strength * 1.5;
            const gradient = ctx.createLinearGradient(conn.from.x, conn.from.y, conn.to.x, conn.to.y);
            gradient.addColorStop(0, `hsla(${conn.from.hue}, 70%, 60%, ${opacity})`);
            gradient.addColorStop(0.5, `hsla(${(conn.from.hue + conn.to.hue) / 2}, 80%, 65%, ${opacity * 1.2})`);
            gradient.addColorStop(1, `hsla(${conn.to.hue}, 70%, 60%, ${opacity})`);
            ctx.strokeStyle = gradient;
            ctx.lineWidth = lineWidth;
            ctx.shadowBlur = 3;
            ctx.shadowColor = `hsla(${(conn.from.hue + conn.to.hue) / 2}, 70%, 60%, ${opacity})`;
            ctx.beginPath();
            ctx.moveTo(conn.from.x, conn.from.y);
            ctx.lineTo(conn.to.x, conn.to.y);
            ctx.stroke();
            ctx.shadowBlur = 0;
        });
    }

    let mouseX = canvas.width / 2;
    let mouseY = canvas.height / 2;
    let targetMouseX = mouseX;
    let targetMouseY = mouseY;

    canvas.addEventListener('mousemove', (e) => {
        targetMouseX = e.clientX;
        targetMouseY = e.clientY;
    });

    function updateMouse() {
        mouseX += (targetMouseX - mouseX) * 0.1;
        mouseY += (targetMouseY - mouseY) * 0.1;
    }

    function animate() {
        updateMouse();
        ctx.fillStyle = 'rgba(10, 14, 39, 0.15)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        nodes.forEach((node) => {
            const dx = node.x - mouseX;
            const dy = node.y - mouseY;
            const distance = Math.sqrt(dx * dx + dy * dy);
            if (distance < 150 && distance > 0) {
                const force = (150 - distance) / 150 * 0.01;
                node.vx += (dx / distance) * force;
                node.vy += (dy / distance) * force;
            }
            node.update(nodes);
        });

        drawConnections(ctx);
        nodes.forEach((node) => node.draw(ctx));
        requestAnimationFrame(animate);
    }

    animate();
}

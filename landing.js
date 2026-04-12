/**
 * ANI Landing Page JS
 * Handles loader, custom cursor, and brutalist particle canvas.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ─── Custom Cursor ──────────────────────────────────────────
    const cursorDot = document.getElementById('cursor-dot');
    const cursorRing = document.getElementById('cursor-ring');
    
    let mouseX = window.innerWidth / 2;
    let mouseY = window.innerHeight / 2;
    let ringX = mouseX;
    let ringY = mouseY;

    window.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
        
        // Dot snaps instantly
        cursorDot.style.left = `${mouseX}px`;
        cursorDot.style.top = `${mouseY}px`;
    });

    function animateCursor() {
        // Ring lags behind (lerp)
        ringX += (mouseX - ringX) / 10;
        ringY += (mouseY - ringY) / 10;
        
        cursorRing.style.left = `${ringX}px`;
        cursorRing.style.top = `${ringY}px`;
        
        requestAnimationFrame(animateCursor);
    }
    animateCursor();

    // Hover effects for links
    document.querySelectorAll('a').forEach(link => {
        link.addEventListener('mouseenter', () => {
            cursorRing.style.width = '50px';
            cursorRing.style.height = '50px';
            cursorRing.style.backgroundColor = 'rgba(184, 255, 61, 0.1)';
        });
        link.addEventListener('mouseleave', () => {
            cursorRing.style.width = '32px';
            cursorRing.style.height = '32px';
            cursorRing.style.backgroundColor = 'transparent';
        });
    });

    // ─── Phase 1: Loading Screen ──────────────────────────────
    const percentageEl = document.getElementById('loader-percentage');
    const progressFill = document.getElementById('loader-progress');
    const loadingScreen = document.getElementById('loading-screen');
    const homepage = document.getElementById('homepage');
    
    let percentage = 0;
    
    function updateLoader() {
        percentage += 1;
        percentageEl.textContent = percentage;
        progressFill.style.width = `${percentage}%`;
        
        if (percentage < 100) {
            // Two speeds: fast below 60%, slow above
            let delay = percentage < 60 ? Math.random() * 20 + 10 : Math.random() * 50 + 30;
            setTimeout(updateLoader, delay);
        } else {
            // Done! Switch phases
            setTimeout(() => {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.classList.add('hidden');
                    homepage.classList.remove('hidden');
                    document.body.style.overflow = 'auto'; // allow scrolling if needed
                    initParticles();
                }, 650);
            }, 650);
        }
    }
    
    // Start loader
    setTimeout(updateLoader, 100);

    // ─── Phase 2: Particle Canvas ─────────────────────────────
    function initParticles() {
        const canvas = document.getElementById('particle-canvas');
        const ctx = canvas.getContext('2d');
        
        let width, height;
        
        function resize() {
            width = window.innerWidth;
            height = window.innerHeight;
            canvas.width = width;
            canvas.height = height;
        }
        window.addEventListener('resize', resize);
        resize();

        const NUM_PARTICLES = 140;
        const MOUSE_Radius = 170;
        const LIME_PERCENT = 0.14;
        
        const LIME_COLOR = '#b8ff3d';
        const WHITE_COLOR = '#e4e0d8';
        const BG_COLOR = 'rgba(7, 7, 7, 0.13)'; // Persistent trail

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 1;
                this.vy = (Math.random() - 0.5) * 1;
                this.radius = Math.random() * 1.5 + 0.5;
                this.isLime = Math.random() < LIME_PERCENT;
                this.color = this.isLime ? LIME_COLOR : WHITE_COLOR;
                // For curl flow
                this.angle = Math.random() * Math.PI * 2;
            }

            update() {
                // Curl flow motion
                this.angle += (Math.random() - 0.5) * 0.2;
                this.vx += Math.cos(this.angle) * 0.05;
                this.vy += Math.sin(this.angle) * 0.05;

                // Friction
                this.vx *= 0.98;
                this.vy *= 0.98;

                // Mouse attraction
                let dx = mouseX - this.x;
                let dy = mouseY - this.y;
                let dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < MOUSE_Radius) {
                    let force = (MOUSE_Radius - dist) / MOUSE_Radius;
                    this.vx += (dx / dist) * force * 0.4;
                    this.vy += (dy / dist) * force * 0.4;
                }

                // Minimum speed
                if (Math.abs(this.vx) < 0.1) this.vx += (Math.random() - 0.5) * 0.1;
                if (Math.abs(this.vy) < 0.1) this.vy += (Math.random() - 0.5) * 0.1;

                this.x += this.vx;
                this.y += this.vy;

                // Wrap around
                if (this.x < 0) this.x = width;
                if (this.x > width) this.x = 0;
                if (this.y < 0) this.y = height;
                if (this.y > height) this.y = 0;
            }

            draw() {
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
                ctx.fillStyle = this.color;
                ctx.globalAlpha = this.isLime ? 1.0 : 0.4;
                ctx.fill();
                ctx.globalAlpha = 1.0;
            }
        }

        const particles = [];
        for (let i = 0; i < NUM_PARTICLES; i++) {
            particles.push(new Particle());
        }

        function drawConnections() {
            for (let i = 0; i < NUM_PARTICLES; i++) {
                for (let j = i + 1; j < NUM_PARTICLES; j++) {
                    const p1 = particles[i];
                    const p2 = particles[j];
                    const dx = p1.x - p2.x;
                    const dy = p1.y - p2.y;
                    const dist = dx * dx + dy * dy;

                    if (dist < 10000) { // 100x100
                        const opacity = 1 - (dist / 10000);
                        ctx.beginPath();
                        ctx.moveTo(p1.x, p1.y);
                        ctx.lineTo(p2.x, p2.y);
                        // If one is lime, tint the line slightly
                        if (p1.isLime || p2.isLime) {
                            ctx.strokeStyle = `rgba(184, 255, 61, ${opacity * 0.15})`;
                        } else {
                            ctx.strokeStyle = `rgba(228, 224, 216, ${opacity * 0.05})`;
                        }
                        ctx.stroke();
                    }
                }
            }
        }

        function loop() {
            // Persistent trail effect
            ctx.fillStyle = BG_COLOR;
            ctx.fillRect(0, 0, width, height);

            particles.forEach(p => {
                p.update();
                p.draw();
            });

            drawConnections();

            requestAnimationFrame(loop);
        }

        loop();
    }
});

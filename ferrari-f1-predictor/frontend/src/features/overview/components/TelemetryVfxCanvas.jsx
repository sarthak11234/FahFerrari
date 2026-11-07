import React, { useEffect, useRef } from 'react';

export default function TelemetryVfxCanvas({ active = true, density = 60 }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !active) return;

    const ctx = canvas.getContext('2d');
    let animationFrameId;
    let width = (canvas.width = canvas.parentElement.offsetWidth);
    let height = (canvas.height = canvas.parentElement.offsetHeight);

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return;
      width = canvas.width = canvas.parentElement.offsetWidth;
      height = canvas.height = canvas.parentElement.offsetHeight;
    };

    window.addEventListener('resize', handleResize);

    // Particle streams (simulating wind tunnel aerodynamics)
    const particles = [];
    const count = Math.min(density, Math.floor(width / 20));

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        length: 20 + Math.random() * 80,
        speed: 4 + Math.random() * 8,
        opacity: 0.15 + Math.random() * 0.45,
        color: Math.random() > 0.85 ? '#55C58A' : Math.random() > 0.3 ? '#E10600' : '#FF3028',
        width: 1 + Math.random() * 1.5,
      });
    }

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Draw subtle telemetry grid crosshairs
      ctx.strokeStyle = 'rgba(41, 45, 51, 0.25)';
      ctx.lineWidth = 1;

      // Draw wind tunnel speed lines
      for (let i = 0; i < particles.length; i++) {
        const p = particles[i];
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + p.length, p.y - p.length * 0.08); // slight downforce angle
        ctx.strokeStyle = p.color;
        ctx.globalAlpha = p.opacity;
        ctx.lineWidth = p.width;
        ctx.stroke();

        // Advance particle
        p.x -= p.speed;
        p.y += p.speed * 0.08;

        // Reset if offscreen
        if (p.x + p.length < 0 || p.y > height) {
          p.x = width + Math.random() * 100;
          p.y = Math.random() * height;
          p.speed = 4 + Math.random() * 8;
        }
      }

      ctx.globalAlpha = 1.0;
      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
    };
  }, [active, density]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 pointer-events-none z-10 w-full h-full mix-blend-screen"
    />
  );
}


document.addEventListener('DOMContentLoaded', function() {
  const canvas = document.createElement('canvas');
  canvas.id = 'healthcare-bg';
  canvas.style.position = 'fixed';
  canvas.style.top = '0';
  canvas.style.left = '0';
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.zIndex = '-1';

  // Replace the particles-js div with our canvas
  const particlesDiv = document.getElementById('particles-js');
  if (particlesDiv) {
    particlesDiv.parentNode.replaceChild(canvas, particlesDiv);
  } else {
    document.body.prepend(canvas);
  }

  // Set canvas dimensions
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;

  // Get canvas context
  const ctx = canvas.getContext('2d');

  // Resize handler
  window.addEventListener('resize', function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    nodes.forEach(node => {
      if (node.x > canvas.width) node.x = Math.random() * canvas.width;
      if (node.y > canvas.height) node.y = Math.random() * canvas.height;
    });
  });

  // Node class for data points
  class Node {
    constructor(x, y, size, type) {
      this.x = x;
      this.y = y;
      this.size = size;
      this.baseSize = size;
      this.type = type; // 'data', 'pulse', or 'connection'
      this.color = this.getColor();
      this.speed = {
        x: (Math.random() - 0.5) * 0.5,
        y: (Math.random() - 0.5) * 0.5
      };
      this.opacity = Math.random() * 0.5 + 0.2;
      this.growing = Math.random() > 0.5;
      this.pulseSpeed = Math.random() * 0.02 + 0.01;
      this.connections = [];
      this.connectionStrength = Math.random();
    }

    getColor() {
      switch(this.type) {
        case 'data':
          return '#3498db'; // Blue for data points
        case 'pulse':
          return '#2ecc71'; // Green for health pulses
        case 'connection':
          return '#9b59b6'; // Purple for AI connections
        default:
          return '#ffffff';
      }
    }

    update() {
      // Movement
      this.x += this.speed.x;
      this.y += this.speed.y;

      // Bounce off edges
      if (this.x > canvas.width || this.x < 0) {
        this.speed.x *= -1;
      }

      if (this.y > canvas.height || this.y < 0) {
        this.speed.y *= -1;
      }

      // Pulsing effect
      if (this.growing) {
        this.size += this.pulseSpeed;
        if (this.size > this.baseSize * 1.5) this.growing = false;
      } else {
        this.size -= this.pulseSpeed;
        if (this.size < this.baseSize * 0.5) this.growing = true;
      }
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.globalAlpha = this.opacity;
      ctx.fill();

      // Draw connection lines
      this.connections.forEach(connection => {
        ctx.beginPath();
        ctx.moveTo(this.x, this.y);
        ctx.lineTo(connection.x, connection.y);
        ctx.strokeStyle = 'rgba(255, 255, 255, ' + (this.connectionStrength * 0.2) + ')';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      });
    }
  }

  // Create nodes
  const nodeCount = Math.floor((canvas.width * canvas.height) / 15000); // Responsive node count
  const nodes = [];

  for (let i = 0; i < nodeCount; i++) {
    const size = Math.random() * 3 + 1;
    const types = ['data', 'pulse', 'connection'];
    const type = types[Math.floor(Math.random() * types.length)];

    const node = new Node(
      Math.random() * canvas.width,
      Math.random() * canvas.height,
      size,
      type
    );

    nodes.push(node);
  }

  // Create connections between nodes
  nodes.forEach(node => {
    nodes.forEach(otherNode => {
      if (node !== otherNode && Math.random() > 0.92) {
        node.connections.push(otherNode);
      }
    });
  });

  // DNA helix effect parameters
  const dnaStrands = 2;
  const dnaRadius = 50;
  const dnaLength = canvas.height * 1.5;
  const dnaGap = 15;
  const dnaSpeed = 0.002;
  let dnaOffset = 0;

  // Heartbeat effect parameters
  const heartbeat = {
    x: canvas.width * 0.15,
    y: canvas.height * 0.85,
    amplitude: 30,
    frequency: 0.02,
    speed: 0.1,
    offset: 0,
    lineLength: 150
  };

  // Brain wave effect parameters
  const brainwave = {
    x: canvas.width * 0.85,
    y: canvas.height * 0.15,
    amplitude: 20,
    frequency: 0.05,
    speed: 0.08,
    offset: 0,
    lineLength: 150
  };

  // Draw DNA helix
  function drawDNA() {
    dnaOffset += dnaSpeed;

    // Position DNA in a visible part of the screen
    const dnaX = canvas.width * 0.75;
    const dnaY = canvas.height * 0.5 - dnaLength / 2;

    for (let i = 0; i < dnaLength; i += dnaGap) {
      const y = dnaY + i;

      // Skip if out of screen
      if (y < 0 || y > canvas.height) continue;

      for (let strand = 0; strand < dnaStrands; strand++) {
        const phase = strand * Math.PI + dnaOffset;
        const x = dnaX + Math.sin(i * 0.03 + phase) * dnaRadius;

        // DNA node
        ctx.beginPath();
        ctx.arc(x, y, 2, 0, Math.PI * 2);
        ctx.fillStyle = strand === 0 ? '#3498db' : '#9b59b6';
        ctx.globalAlpha = 0.6;
        ctx.fill();

        // Connect strands periodically
        if (i % (dnaGap * 4) === 0) {
          ctx.beginPath();
          const otherX = dnaX + Math.sin(i * 0.03 + (strand === 0 ? Math.PI : 0) + dnaOffset) * dnaRadius;
          ctx.moveTo(x, y);
          ctx.lineTo(otherX, y);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
  }

  // Draw heartbeat line
  function drawHeartbeat() {
    heartbeat.offset += heartbeat.speed;

    ctx.beginPath();
    ctx.moveTo(heartbeat.x, heartbeat.y);

    for (let i = 0; i < heartbeat.lineLength; i++) {
      const x = heartbeat.x + i;

      // Create the ECG-like pattern
      let y = heartbeat.y;
      const phase = i * heartbeat.frequency + heartbeat.offset;

      // Create the characteristic ECG spike
      if (phase % (2 * Math.PI) > 1.5 * Math.PI && phase % (2 * Math.PI) < 1.7 * Math.PI) {
        y -= heartbeat.amplitude * 2;
      } else if (phase % (2 * Math.PI) > 1.7 * Math.PI && phase % (2 * Math.PI) < 1.9 * Math.PI) {
        y += heartbeat.amplitude * 3;
      } else {
        y -= Math.sin(phase) * heartbeat.amplitude * 0.2;
      }

      ctx.lineTo(x, y);
    }

    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;
    ctx.stroke();
  }

  // Draw brain wave
  function drawBrainwave() {
    brainwave.offset += brainwave.speed;

    ctx.beginPath();
    ctx.moveTo(brainwave.x, brainwave.y);

    for (let i = 0; i < brainwave.lineLength; i++) {
      const x = brainwave.x - i;

      // Create the EEG-like pattern
      let y = brainwave.y;
      const phase = i * brainwave.frequency + brainwave.offset;

      // More random looking waves for brain activity
      y += Math.sin(phase) * brainwave.amplitude * 0.5;
      y += Math.sin(phase * 2.5) * brainwave.amplitude * 0.3;
      y += Math.sin(phase * 0.6) * brainwave.amplitude * 0.2;

      ctx.lineTo(x, y);
    }

    ctx.strokeStyle = '#9b59b6';
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.7;
    ctx.stroke();
  }

  // Animation loop
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw radial gradient background
    const gradient = ctx.createRadialGradient(
      canvas.width / 2, canvas.height / 2, 0,
      canvas.width / 2, canvas.height / 2, canvas.width * 0.8
    );
    gradient.addColorStop(0, '#1f1b4e');
    gradient.addColorStop(1, '#121224');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Update and draw nodes
    nodes.forEach(node => {
      node.update();
      node.draw();
    });

    // Draw healthcare-specific elements
    drawDNA();
    drawHeartbeat();
    drawBrainwave();

    requestAnimationFrame(animate);
  }

  // Start animation
  animate();
});
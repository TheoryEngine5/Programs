import React, { useState, useEffect } from 'react';

const SierpinskiFractals = () => {
  const [iterations, setIterations] = useState(3);
  const [displayMode, setDisplayMode] = useState('text');
  const [animationSpeed, setAnimationSpeed] = useState(1000);
  const [currentIteration, setCurrentIteration] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // Original text-based Sierpinski function
  const sierpinski = (n) => {
    if (n === 0) {
      return ["*"];
    }
    const prev = sierpinski(n - 1);
    const result = [];
    const space = " ".repeat(prev[0].length);
    
    // Top part: each line duplicated side by side
    for (const line of prev) {
      result.push(line + " " + line);
    }
    
    // Bottom part: each line centered below
    for (const line of prev) {
      result.push(space + line + space);
    }
    
    return result;
  };

  // Generate coordinates for graphical display
  const generateSierpinskiPoints = (n, size = 400) => {
    const points = [];
    const height = size * Math.sqrt(3) / 2;
    
    // Base triangle vertices
    const vertices = [
      { x: size / 2, y: 0 },           // Top
      { x: 0, y: height },             // Bottom left  
      { x: size, y: height }           // Bottom right
    ];

    const generatePoints = (level, v1, v2, v3) => {
      if (level === 0) {
        // Add the triangle
        points.push([v1, v2, v3]);
        return;
      }
      
      // Calculate midpoints
      const m1 = { x: (v1.x + v2.x) / 2, y: (v1.y + v2.y) / 2 };
      const m2 = { x: (v2.x + v3.x) / 2, y: (v2.y + v3.y) / 2 };
      const m3 = { x: (v3.x + v1.x) / 2, y: (v3.y + v1.y) / 2 };
      
      // Recursively generate three smaller triangles
      generatePoints(level - 1, v1, m1, m3);
      generatePoints(level - 1, m1, v2, m2);
      generatePoints(level - 1, m3, m2, v3);
    };

    generatePoints(n, vertices[0], vertices[1], vertices[2]);
    return points;
  };

  // Chaos game method
  const generateChaosGamePoints = (numPoints = 10000) => {
    const points = [];
    const vertices = [
      { x: 200, y: 50 },
      { x: 50, y: 350 },
      { x: 350, y: 350 }
    ];
    
    let current = { x: 200, y: 200 }; // Starting point
    
    for (let i = 0; i < numPoints; i++) {
      const randomVertex = vertices[Math.floor(Math.random() * 3)];
      current = {
        x: (current.x + randomVertex.x) / 2,
        y: (current.y + randomVertex.y) / 2
      };
      
      if (i > 100) { // Skip first few points to let it settle
        points.push({ ...current });
      }
    }
    
    return points;
  };

  const [chaosPoints, setChaosPoints] = useState([]);

  useEffect(() => {
    if (displayMode === 'chaos') {
      setChaosPoints(generateChaosGamePoints());
    }
  }, [displayMode]);

  // Animation effect
  useEffect(() => {
    let interval;
    if (isAnimating) {
      interval = setInterval(() => {
        setCurrentIteration(prev => {
          if (prev >= iterations) {
            setIsAnimating(false);
            return iterations;
          }
          return prev + 1;
        });
      }, animationSpeed);
    }
    return () => clearInterval(interval);
  }, [isAnimating, animationSpeed, iterations]);

  const startAnimation = () => {
    setCurrentIteration(0);
    setIsAnimating(true);
  };

  const textLines = sierpinski(displayMode === 'animated' ? currentIteration : iterations);
  const graphicalPoints = generateSierpinskiPoints(displayMode === 'animated' ? currentIteration : iterations);

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg">
      <h1 className="text-3xl font-bold text-center mb-6 bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
        🔺 Sierpinski Triangle Explorer
      </h1>
      
      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6 p-4 bg-white rounded-lg shadow-sm">
        <div className="flex items-center gap-2">
          <label className="font-medium">Iterations:</label>
          <input
            type="range"
            min="0"
            max="6"
            value={iterations}
            onChange={(e) => setIterations(parseInt(e.target.value))}
            className="w-20"
          />
          <span className="w-8 text-center">{iterations}</span>
        </div>
        
        <div className="flex items-center gap-2">
          <label className="font-medium">Display:</label>
          <select
            value={displayMode}
            onChange={(e) => setDisplayMode(e.target.value)}
            className="border rounded px-2 py-1"
          >
            <option value="text">Text Pattern</option>
            <option value="graphical">Geometric</option>
            <option value="chaos">Chaos Game</option>
            <option value="animated">Animation</option>
          </select>
        </div>

        {displayMode === 'animated' && (
          <>
            <div className="flex items-center gap-2">
              <label className="font-medium">Speed (ms):</label>
              <input
                type="range"
                min="200"
                max="2000"
                value={animationSpeed}
                onChange={(e) => setAnimationSpeed(parseInt(e.target.value))}
                className="w-20"
              />
              <span className="w-12 text-center text-sm">{animationSpeed}</span>
            </div>
            <button
              onClick={startAnimation}
              disabled={isAnimating}
              className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600 disabled:opacity-50"
            >
              {isAnimating ? `Gen ${currentIteration}` : 'Start Animation'}
            </button>
          </>
        )}
      </div>

      {/* Display Area */}
      <div className="bg-white rounded-lg shadow-lg p-6 min-h-96">
        {displayMode === 'text' || displayMode === 'animated' ? (
          <div className="font-mono text-sm leading-tight overflow-x-auto">
            <div className="bg-gray-900 text-green-400 p-4 rounded">
              {textLines.map((line, index) => (
                <div key={index} className="whitespace-pre">
                  {line}
                </div>
              ))}
            </div>
          </div>
        ) : displayMode === 'graphical' ? (
          <div className="flex justify-center">
            <svg width="400" height="350" className="border rounded">
              {graphicalPoints.map((triangle, index) => (
                <polygon
                  key={index}
                  points={triangle.map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="#8b5cf6"
                  strokeWidth="1"
                  opacity="0.8"
                />
              ))}
            </svg>
          </div>
        ) : displayMode === 'chaos' && (
          <div className="flex justify-center">
            <svg width="400" height="400" className="border rounded bg-black">
              {/* Vertices */}
              <circle cx="200" cy="50" r="3" fill="#ff6b6b" />
              <circle cx="50" cy="350" r="3" fill="#4ecdc4" />
              <circle cx="350" cy="350" r="3" fill="#45b7d1" />
              
              {/* Generated points */}
              {chaosPoints.map((point, index) => (
                <circle
                  key={index}
                  cx={point.x}
                  cy={point.y}
                  r="0.5"
                  fill="#ffffff"
                  opacity="0.8"
                />
              ))}
            </svg>
          </div>
        )}
      </div>

      {/* Information Panel */}
      <div className="mt-6 grid md:grid-cols-2 gap-4">
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <h3 className="font-bold text-lg mb-2">🧮 Math Facts</h3>
          <ul className="text-sm space-y-1">
            <li>• Dimension: ~1.585 (fractal dimension)</li>
            <li>• Self-similar at all scales</li>
            <li>• Area approaches 0 as iterations increase</li>
            <li>• Perimeter grows without bound</li>
            <li>• Connected to Pascal's triangle mod 2</li>
          </ul>
        </div>
        
        <div className="bg-white p-4 rounded-lg shadow-sm">
          <h3 className="font-bold text-lg mb-2">💡 Generation Methods</h3>
          <ul className="text-sm space-y-1">
            <li>• <strong>Text:</strong> Recursive string building</li>
            <li>• <strong>Geometric:</strong> Triangle subdivision</li>
            <li>• <strong>Chaos Game:</strong> Random vertex jumping</li>
            <li>• <strong>Animation:</strong> Progressive generation</li>
          </ul>
        </div>
      </div>

      {/* Code Display */}
      <div className="mt-6 bg-gray-800 text-gray-100 p-4 rounded-lg">
        <h3 className="text-lg font-bold mb-2 text-green-400">📝 Original Algorithm:</h3>
        <pre className="text-sm overflow-x-auto">
{`def sierpinski(n):
    if n == 0:
        return ["*"]
    prev = sierpinski(n-1)
    result = []
    space = " " * len(prev[0])
    
    # Top: duplicate each line side by side  
    for line in prev:
        result.append(line + " " + line)
    
    # Bottom: center each line below
    for line in prev:
        result.append(space + line + space)
    
    return result`}
        </pre>
      </div>
    </div>
  );
};

export default SierpinskiFractals;
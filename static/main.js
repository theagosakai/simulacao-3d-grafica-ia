import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.164/build/three.module.js";
import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.164/examples/jsm/controls/OrbitControls.js";

const sceneElement = document.querySelector("#scene");
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x080b14, 0.012);

const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 1000);
camera.position.set(0, 11, 27);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
sceneElement.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xb9c4e6, 1.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
keyLight.position.set(8, 12, 12);
scene.add(keyLight);

const grid = new THREE.GridHelper(26, 26, 0x29365f, 0x171d31);
grid.rotation.x = 0;
grid.position.y = -7;
scene.add(grid);

const nodeObjects = new Map();
const edgeObjects = [];

function nodePosition(layer, index, count) {
  const x = (layer - 1) * 11;
  const spacing = count > 1 ? 8 / (count - 1) : 0;
  const y = (index - (count - 1) / 2) * spacing;
  return new THREE.Vector3(x, y, 0);
}

function addNode(id, label, position, color) {
  const geometry = new THREE.SphereGeometry(0.72, 32, 32);
  const material = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.25,
    metalness: 0.15,
    emissive: color,
    emissiveIntensity: 0.12,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(position);
  mesh.userData = { id, label, baseColor: color };
  scene.add(mesh);
  nodeObjects.set(id, mesh);

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.88, 24, 24),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08 })
  );
  halo.position.copy(position);
  scene.add(halo);
}

function addEdge(sourceId, targetId, weight) {
  const source = nodeObjects.get(sourceId).position;
  const target = nodeObjects.get(targetId).position;
  const direction = new THREE.Vector3().subVectors(target, source);
  const length = direction.length();
  const geometry = new THREE.CylinderGeometry(
    0.018 + Math.min(Math.abs(weight) * 0.025, 0.055),
    0.018 + Math.min(Math.abs(weight) * 0.025, 0.055),
    length,
    8
  );
  const color = weight >= 0 ? 0x78e2b4 : 0xff718b;
  const material = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.34 + Math.min(Math.abs(weight), 1) * 0.25,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.copy(source).add(direction.multiplyScalar(0.5));
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    direction.clone().normalize()
  );
  scene.add(mesh);
  edgeObjects.push(mesh);
}

function buildGraph(model) {
  const inputs = model.inputNodes;
  const hidden = model.hiddenNodes;
  const outputs = model.outputNodes;

  inputs.forEach((node, index) =>
    addNode(node.id, node.label, nodePosition(0, index, inputs.length), 0x7188ff)
  );
  hidden.forEach((node, index) =>
    addNode(node.id, node.label, nodePosition(1, index, hidden.length), 0x8c72d9)
  );
  outputs.forEach((node, index) =>
    addNode(node.id, node.label, nodePosition(2, index, outputs.length), 0xf1b76e)
  );
  model.inputHiddenEdges.forEach((edge) =>
    addEdge(edge.source, edge.target, edge.weight)
  );
  model.hiddenOutputEdges.forEach((edge) =>
    addEdge(edge.source, edge.target, edge.weight)
  );
}

async function updatePrediction() {
  const x = document.querySelector("#xInput").value;
  const y = document.querySelector("#yInput").value;
  document.querySelector("#xValue").value = Number(x).toFixed(2);
  document.querySelector("#yValue").value = Number(y).toFixed(2);

  const response = await fetch(`/api/predict?x=${x}&y=${y}`);
  const result = await response.json();
  document.querySelector("#probability").textContent =
    `${(result.probability * 100).toFixed(1)}%`;
  document.querySelector("#predictionClass").textContent =
    `Classe ${result.class}`;

  const output = nodeObjects.get("output");
  if (output) {
    output.material.emissive.set(result.class === 1 ? 0x78e2b4 : 0x7188ff);
    output.scale.setScalar(0.9 + result.probability * 0.2);
  }
}

function drawLossChart(history) {
  const canvas = document.querySelector("#lossChart");
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "rgba(143,160,208,.18)";
  context.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (height / 4) * i;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(width, y);
    context.stroke();
  }
  if (!history.length) return;
  const max = Math.max(...history, 0.01);
  context.strokeStyle = "#78e2b4";
  context.lineWidth = 2;
  context.beginPath();
  history.forEach((value, index) => {
    const x = (index / Math.max(history.length - 1, 1)) * width;
    const y = height - (value / max) * (height - 18) - 9;
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.stroke();
}

function resize() {
  const width = sceneElement.clientWidth;
  const height = sceneElement.clientHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

const modelResponse = await fetch("/api/model");
const modelData = await modelResponse.json();
buildGraph(modelData.graph);
drawLossChart(modelData.lossHistory);
document.querySelector("#finalLoss").textContent =
  modelData.lossHistory.at(-1)?.toFixed(5) ?? "--";

document.querySelector("#trainingSamples").innerHTML =
  modelData.trainingSamples.map((sample) =>
    `<div class="sample">[${sample.x[0]}, ${sample.x[1]}] <strong>${sample.target}</strong></div>`
  ).join("");

document.querySelector("#xInput").addEventListener("input", updatePrediction);
document.querySelector("#yInput").addEventListener("input", updatePrediction);
window.addEventListener("resize", resize);
resize();
updatePrediction();

const clock = new THREE.Clock();
function animate() {
  const elapsed = clock.getElapsedTime();
  edgeObjects.forEach((edge, index) => {
    edge.material.opacity = 0.28 + Math.sin(elapsed * 1.5 + index * 0.3) * 0.08;
  });
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

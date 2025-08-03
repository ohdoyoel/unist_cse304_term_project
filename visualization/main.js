// 지도 초기화 (한국 중심으로 설정)
const map = L.map("map").setView([37.5665, 126.978], 6);

// ESRI World Imagery (위성 사진) 타일 레이어 추가
L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
  {
    attribution:
      "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community",
    maxZoom: 19,
  }
).addTo(map);

// 지도 컨트롤 추가
L.control.scale().addTo(map);
map.zoomControl.setPosition("topright");

// 요소 초기화
const datasetSelect = document.getElementById("dataset-select");
const algorithmSelect = document.getElementById("algorithm-select");
const loadingElement = document.getElementById("loading");
const showEdgesCheckbox = document.getElementById("show-edges");
const boundaryEdgesCheckbox = document.getElementById("boundary-edges");
let selectedDataset = datasetSelect.value;
let selectedAlgorithm = algorithmSelect.value;

// 전역 변수
const clusterColors = new Map();
let edgeLayerGroup = L.layerGroup().addTo(map);
let nodeLayerGroup = L.layerGroup().addTo(map);
let currentEdges = []; // 현재 로드된 모든 엣지 데이터
let currentNodeIndex = new Map(); // 현재 로드된 노드 인덱스

// 로딩 상태 제어 함수
function setLoading(isLoading) {
  loadingElement.style.display = isLoading ? "block" : "none";
  datasetSelect.disabled = isLoading;
  algorithmSelect.disabled = isLoading;
}

// cluster_id를 기반으로 색상 생성 함수
function getClusterColor(clusterId) {
  if (!clusterColors.has(clusterId)) {
    const id = parseInt(clusterId);
    const goldenRatio = 0.618033988749895;
    const hue = (id * goldenRatio) % 1;
    const color = `hsl(${Math.floor(hue * 360)}, 70%, 60%)`;
    clusterColors.set(clusterId, color);
  }
  return clusterColors.get(clusterId);
}

// 선택 변경 처리 함수
function handleChange(event) {
  if (event.target.id === "dataset-select") {
    selectedDataset = event.target.value;
  } else {
    selectedAlgorithm = event.target.value;
  }
  console.log("데이터셋:", selectedDataset, "알고리즘:", selectedAlgorithm);
  clearMapData();
  loadData();
}

// 지도에서 기존 데이터 제거하는 함수
function clearMapData() {
  edgeLayerGroup.clearLayers();
  nodeLayerGroup.clearLayers();
  clusterColors.clear();
  currentEdges = [];
  currentNodeIndex.clear();
}

// CSV 파일 로드 함수
async function loadCSV(url) {
  return new Promise((resolve, reject) => {
    Papa.parse(url, {
      download: true,
      header: true,
      complete: (results) => resolve(results.data),
      error: (error) => reject(error),
    });
  });
}

// 노드 인덱스 생성 함수
function createNodeIndex(nodes) {
  const index = new Map();
  nodes.forEach((node) => {
    index.set(node.node_id, node);
  });
  return index;
}

// 좌표가 현재 지도 경계 내에 있는지 확인하는 함수
function isCoordInBounds(coord) {
  const bounds = map.getBounds();
  return bounds.contains(L.latLng(coord[0], coord[1]));
}

// 엣지가 현재 지도 경계 내에 있는지 확인하는 함수
function isEdgeInBounds(coords) {
  if (boundaryEdgesCheckbox.checked) {
    return isCoordInBounds(coords[0]) && isCoordInBounds(coords[1]);
  }
  return true;
}

// 엣지 시각화 함수
function visualizeEdges() {
  edgeLayerGroup.clearLayers();

  if (!showEdgesCheckbox.checked) return;

  // 엣지 데이터 준비
  const edgesByCluster = new Map();
  const interClusterEdges = [];

  currentEdges.forEach((edge) => {
    const sourceNode = currentNodeIndex.get(edge.source);
    const targetNode = currentNodeIndex.get(edge.target);

    if (sourceNode && targetNode) {
      const coords = [
        [parseFloat(sourceNode.latitude), parseFloat(sourceNode.longitude)],
        [parseFloat(targetNode.latitude), parseFloat(targetNode.longitude)],
      ];

      // 경계 체크
      if (!isEdgeInBounds(coords)) return;

      if (sourceNode.cluster_label === targetNode.cluster_label) {
        const clusterId = sourceNode.cluster_label;
        if (!edgesByCluster.has(clusterId)) {
          edgesByCluster.set(clusterId, []);
        }
        edgesByCluster.get(clusterId).push(coords);
      } else {
        interClusterEdges.push(coords);
      }
    }
  });

  // 클러스터 내부 엣지 그리기
  edgesByCluster.forEach((clusterEdges, clusterId) => {
    L.polyline(clusterEdges, {
      color: getClusterColor(clusterId),
      weight: boundaryEdgesCheckbox.checked ? 2 : 0.5,
      opacity: 1,
    }).addTo(edgeLayerGroup);
  });

  // 클러스터 간 엣지 그리기
  if (interClusterEdges.length > 0) {
    L.polyline(interClusterEdges, {
      color: "#ffffff",
      weight: 0.5,
      opacity: 0.5,
    }).addTo(edgeLayerGroup);
  }
}

// 데이터 로드 및 시각화 함수
async function loadData() {
  try {
    setLoading(true);

    // 데이터 파일 경로 설정 (상대 경로 사용)
    const nodesFile = `data/${selectedDataset}_${selectedAlgorithm}_nodes.csv`;
    const edgesFile = `data/${selectedDataset}_${selectedAlgorithm}_edges.csv`;

    const [nodes, edges] = await Promise.all([
      loadCSV(nodesFile),
      loadCSV(edgesFile),
    ]);

    // 전역 변수 업데이트
    currentEdges = edges;
    currentNodeIndex = createNodeIndex(nodes);

    // 노드 시각화
    nodes.forEach((node) => {
      const lat = parseFloat(node.latitude);
      const lng = parseFloat(node.longitude);
      const clusterId = node.cluster_label;

      if (!isNaN(lat) && !isNaN(lng)) {
        const color = getClusterColor(clusterId);
        L.circle([lat, lng], {
          color: color,
          fillColor: color,
          fillOpacity: 1,
          radius: 40,
        }).addTo(nodeLayerGroup);
      }
    });

    // 엣지 시각화
    visualizeEdges();

    // 지도 뷰 조정
    const validCoords = nodes
      .map((node) => {
        const lat = parseFloat(node.latitude);
        const lng = parseFloat(node.longitude);
        return !isNaN(lat) && !isNaN(lng) ? [lat, lng] : null;
      })
      .filter((coord) => coord !== null);

    if (validCoords.length > 0) {
      const bounds = L.latLngBounds(validCoords);
      map.fitBounds(bounds);
    }
    map.fitBounds(bounds);
  } catch (error) {
    // alert("데이터 로드 중 오류 발생: " + error);
    console.error("데이터 로드 중 오류 발생:", error);
  } finally {
    setLoading(false);
  }
}

// 이벤트 리스너 추가
algorithmSelect.addEventListener("change", handleChange);
datasetSelect.addEventListener("change", handleChange);
showEdgesCheckbox.addEventListener("change", visualizeEdges);
boundaryEdgesCheckbox.addEventListener("change", visualizeEdges);
map.on("moveend", visualizeEdges); // 지도 이동/줌 완료 시 엣지 다시 그리기

// 초기 데이터 로드
loadData();

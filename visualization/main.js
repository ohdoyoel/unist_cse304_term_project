// 지도 초기화 (전 세계 지도)
const map = L.map("map").setView([20, 0], 2);

// ESRI World Imagery (위성 사진) 타일 레이어 추가
L.tileLayer(
  "https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}",
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
let isUIVisible = true; // UI 표시 상태 관리
let overlayLayer = null; // 어두운 배경 레이어
let mapInteractionDisabled = false; // 지도 상호작용 비활성화 상태

// 화면 내 노드만 다시 그리는 함수
function redrawVisibleNodes() {
  if (currentNodeIndex.size === 0) return;

  nodeLayerGroup.clearLayers();

  const nodes = Array.from(currentNodeIndex.values());
  const clusterCounts = !isUIVisible
    ? getVisibleClusterNodeCounts()
    : getClusterNodeCounts();
  const connectedNodes = !isUIVisible ? getConnectedNodesInBounds() : null;
  const bounds = map.getBounds();

  nodes.forEach((node) => {
    const lat = parseFloat(node.latitude);
    const lng = parseFloat(node.longitude);
    const clusterId = node.cluster_label;

    if (!isNaN(lat) && !isNaN(lng)) {
      const nodeLatLng = L.latLng(lat, lng);

      // 화면 내에 있는 노드만 그리기
      if (!bounds.contains(nodeLatLng)) {
        return;
      }

      // UI 숨김 모드에서 필터링 조건들
      if (!isUIVisible) {
        // 1개 노드 클러스터는 제외
        if (clusterCounts.get(clusterId) <= 1) {
          console.log(
            `노드 ${node.node_id} 제외: 클러스터 크기 ${clusterCounts.get(
              clusterId
            )}`
          );
          return;
        }
        // 엣지로 연결되지 않은 노드는 제외
        if (!connectedNodes.has(node.node_id)) {
          console.log(`노드 ${node.node_id} 제외: 연결된 엣지 없음`);
          return;
        }
      }

      const color = getClusterColor(clusterId);
      L.circleMarker([lat, lng], {
        color: color,
        fillColor: color,
        fillOpacity: 0.7,
        radius: 2,
        weight: 0,
      }).addTo(nodeLayerGroup);
    }
  });
}

// 이벤트 핸들러 함수들을 전역으로 선언하여 제거할 수 있도록 함
const moveEndHandler = function () {
  redrawVisibleNodes(); // 화면 내 노드 다시 그리기
  visualizeEdges();
  updateMapInfo();
};

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

// 클러스터별 노드 개수 계산 함수 (전체 데이터 기준)
function getClusterNodeCounts() {
  const clusterCounts = new Map();
  currentNodeIndex.forEach((node) => {
    const clusterId = node.cluster_label;
    clusterCounts.set(clusterId, (clusterCounts.get(clusterId) || 0) + 1);
  });
  return clusterCounts;
}

// 현재 화면 내 클러스터별 노드 개수 계산 함수
function getVisibleClusterNodeCounts() {
  const clusterCounts = new Map();
  const bounds = map.getBounds();

  currentNodeIndex.forEach((node) => {
    const lat = parseFloat(node.latitude);
    const lng = parseFloat(node.longitude);

    if (!isNaN(lat) && !isNaN(lng)) {
      const nodeLatLng = L.latLng(lat, lng);
      if (bounds.contains(nodeLatLng)) {
        const clusterId = node.cluster_label;
        clusterCounts.set(clusterId, (clusterCounts.get(clusterId) || 0) + 1);
      }
    }
  });

  return clusterCounts;
}

// 현재 화면 내에서 엣지로 연결된 노드들을 찾는 함수
function getConnectedNodesInBounds() {
  const connectedNodes = new Set();
  const bounds = map.getBounds();

  currentEdges.forEach((edge) => {
    const sourceNode = currentNodeIndex.get(edge.source);
    const targetNode = currentNodeIndex.get(edge.target);

    if (sourceNode && targetNode) {
      const sourceLat = parseFloat(sourceNode.latitude);
      const sourceLng = parseFloat(sourceNode.longitude);
      const targetLat = parseFloat(targetNode.latitude);
      const targetLng = parseFloat(targetNode.longitude);

      const sourceInBounds =
        !isNaN(sourceLat) &&
        !isNaN(sourceLng) &&
        bounds.contains(L.latLng(sourceLat, sourceLng));
      const targetInBounds =
        !isNaN(targetLat) &&
        !isNaN(targetLng) &&
        bounds.contains(L.latLng(targetLat, targetLng));

      // 양쪽 노드가 모두 화면 내에 있는 엣지만 고려 (더 엄격한 조건)
      if (sourceInBounds && targetInBounds) {
        connectedNodes.add(edge.source);
        connectedNodes.add(edge.target);
      }
    }
  });

  console.log(`화면 내 연결된 노드 개수: ${connectedNodes.size}`);
  return connectedNodes;
}

// 엣지 시각화 함수
function visualizeEdges() {
  edgeLayerGroup.clearLayers();

  if (!showEdgesCheckbox.checked) return;

  // 클러스터별 노드 개수 계산 (UI 숨김 모드에서 사용 - 화면 내 기준)
  const clusterCounts = !isUIVisible ? getVisibleClusterNodeCounts() : null;

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

        // UI 숨김 모드에서 1개 노드 클러스터는 제외
        if (
          !isUIVisible &&
          clusterCounts &&
          clusterCounts.get(clusterId) <= 1
        ) {
          return;
        }

        if (!edgesByCluster.has(clusterId)) {
          edgesByCluster.set(clusterId, []);
        }
        edgesByCluster.get(clusterId).push(coords);
      } else {
        // UI 숨김 모드에서는 클러스터 간 엣지 제외
        if (!isUIVisible) return;

        interClusterEdges.push(coords);
      }
    }
  });

  // 클러스터 내부 엣지 그리기
  edgesByCluster.forEach((clusterEdges, clusterId) => {
    // UI 숨김 모드에서 1개 노드 클러스터는 제외
    if (!isUIVisible && clusterCounts && clusterCounts.get(clusterId) <= 1) {
      return;
    }

    L.polyline(clusterEdges, {
      color: getClusterColor(clusterId),
      weight: boundaryEdgesCheckbox.checked ? 2 : 0.5,
      opacity: 0.7,
    }).addTo(edgeLayerGroup);
  });

  // 클러스터 간 엣지 그리기 (UI 표시 모드에서만)
  if (isUIVisible && interClusterEdges.length > 0) {
    L.polyline(interClusterEdges, {
      color: "#ffffff",
      weight: 0.5,
      opacity: 0.7,
    }).addTo(edgeLayerGroup);
  }
}

// 데이터 로드 및 시각화 함수
async function loadData() {
  try {
    setLoading(true);

    // 데이터 파일 경로 설정 (상대 경로 사용)
    const nodesFile = `../result/${selectedDataset}_${selectedAlgorithm}_nodes.csv`;
    const edgesFile = `../result/${selectedDataset}_${selectedAlgorithm}_edges.csv`;

    const [nodes, edges] = await Promise.all([
      loadCSV(nodesFile),
      loadCSV(edgesFile),
    ]);

    // 전역 변수 업데이트
    currentEdges = edges;
    currentNodeIndex = createNodeIndex(nodes);

    // UI 숨김 모드에서 필터링에 필요한 데이터 계산
    const clusterCounts = !isUIVisible
      ? getVisibleClusterNodeCounts()
      : getClusterNodeCounts();
    const connectedNodes = !isUIVisible ? getConnectedNodesInBounds() : null;

    // 노드 시각화
    nodes.forEach((node) => {
      const lat = parseFloat(node.latitude);
      const lng = parseFloat(node.longitude);
      const clusterId = node.cluster_label;

      if (!isNaN(lat) && !isNaN(lng)) {
        // UI 숨김 모드에서 필터링 조건들
        if (!isUIVisible) {
          // 1개 노드 클러스터는 제외
          if (clusterCounts.get(clusterId) <= 1) {
            console.log(
              `노드 ${node.node_id} 제외: 클러스터 크기 ${clusterCounts.get(
                clusterId
              )}`
            );
            return;
          }
          // 엣지로 연결되지 않은 노드는 제외
          if (!connectedNodes.has(node.node_id)) {
            console.log(`노드 ${node.node_id} 제외: 연결된 엣지 없음`);
            return;
          }
        }

        const color = getClusterColor(clusterId);
        L.circleMarker([lat, lng], {
          color: color,
          fillColor: color,
          fillOpacity: 0.7,
          radius: 2,
          weight: 0,
        }).addTo(nodeLayerGroup);
      }
    });

    // 엣지 시각화
    visualizeEdges();

    // // 지도 뷰 조정
    // const validCoords = nodes
    //   .map((node) => {
    //     const lat = parseFloat(node.latitude);
    //     const lng = parseFloat(node.longitude);
    //     return !isNaN(lat) && !isNaN(lng) ? [lat, lng] : null;
    //   })
    //   .filter((coord) => coord !== null);
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
// 초기 이벤트 리스너 등록
map.on("moveend", moveEndHandler);
map.on("zoom", moveEndHandler);

// 초기 데이터 로드
loadData();

// 지도 정보(위도, 경도, 줌레벨) 표시 함수 추가
function updateMapInfo() {
  // 지도 중심 좌표와 줌레벨을 가져옴
  const center = map.getCenter();
  const zoom = map.getZoom();
  // 소수점 4자리로 포맷
  const lat = center.lat.toFixed(4);
  const lng = center.lng.toFixed(4);
  // 표시할 문자열 생성 (줄바꿈을 위해 <br> 사용)
  const infoText = `latitude: ${lat}<br>longitude: ${lng}<br>zoom: ${zoom}`;
  // 우측하단 div에 표시 (innerHTML로 줄바꿈 적용)
  document.getElementById("map-info").innerHTML = infoText;
}

// 최초 로드 시에도 정보 표시
updateMapInfo();

// 어두운 배경 레이어 토글 함수
function toggleOverlayLayer() {
  if (!isUIVisible) {
    // UI 숨김 모드: 어두운 배경 레이어 추가
    if (!overlayLayer) {
      overlayLayer = L.rectangle(
        [
          [-90, -180],
          [90, 180],
        ], // 전 세계를 덮는 사각형
        {
          color: "transparent",
          fillColor: "black",
          fillOpacity: 0.5,
          weight: 0,
          interactive: false, // 마우스 이벤트 무시
        }
      );
    }
    overlayLayer.addTo(map);
    // 노드와 엣지 레이어를 맨 위로 가져오기 (LayerGroup은 bringToFront 메서드가 없으므로 제거 후 재추가)
    map.removeLayer(nodeLayerGroup);
    map.removeLayer(edgeLayerGroup);
    nodeLayerGroup.addTo(map);
    edgeLayerGroup.addTo(map);
  } else {
    // UI 표시 모드: 어두운 배경 레이어 제거
    if (overlayLayer && map.hasLayer(overlayLayer)) {
      map.removeLayer(overlayLayer);
    }
  }
}

// 지도 상호작용 제어 함수
function toggleMapInteraction() {
  if (!isUIVisible) {
    // UI 숨김 모드: 지도 상호작용 및 이벤트 리스너 완전 비활성화
    map.dragging.disable();
    map.touchZoom.disable();
    map.doubleClickZoom.disable();
    map.scrollWheelZoom.disable();
    map.boxZoom.disable();
    map.keyboard.disable();

    // 추가적인 상호작용 비활성화
    map.zoomControl.disable();
    if (map.tap) map.tap.disable();

    // 지도 컨테이너에 pointer-events none 추가
    map.getContainer().style.pointerEvents = "none";
    map.getContainer().style.cursor = "default";

    // 이벤트 리스너 제거
    map.off("moveend", moveEndHandler);
    map.off("zoom", moveEndHandler);

    mapInteractionDisabled = true;
  } else {
    // UI 표시 모드: 지도 상호작용 및 이벤트 리스너 활성화
    map.dragging.enable();
    map.touchZoom.enable();
    map.doubleClickZoom.enable();
    map.scrollWheelZoom.enable();
    map.boxZoom.enable();
    map.keyboard.enable();

    // 추가적인 상호작용 활성화
    map.zoomControl.enable();
    if (map.tap) map.tap.enable();

    // 지도 컨테이너의 pointer-events 복원
    map.getContainer().style.pointerEvents = "";
    map.getContainer().style.cursor = "";

    // 이벤트 리스너 추가
    map.on("moveend", moveEndHandler);
    map.on("zoom", moveEndHandler);

    mapInteractionDisabled = false;
  }
}

// UI 투명도 토글 함수
function toggleUIVisibility() {
  isUIVisible = !isUIVisible;
  const opacity = isUIVisible ? 0.8 : 0;

  // UI 컨트롤 요소들의 투명도 변경
  const selectionControl = document.querySelector(".selection-control");
  const edgeControl = document.querySelector(".edge-control");
  const mapInfoControl = document.querySelector(".map-info-control");

  if (selectionControl) selectionControl.style.opacity = opacity;
  if (edgeControl) edgeControl.style.opacity = opacity;
  if (mapInfoControl) mapInfoControl.style.opacity = opacity;

  // 어두운 배경 레이어 토글
  toggleOverlayLayer();

  // 지도 상호작용 토글
  toggleMapInteraction();

  // 노드와 엣지 다시 그리기 (1개 노드 클러스터 및 클러스터 간 엣지 필터링 적용)
  if (currentNodeIndex.size > 0) {
    // 기존 레이어만 클리어하고 데이터는 유지
    nodeLayerGroup.clearLayers();
    edgeLayerGroup.clearLayers();

    // 현재 데이터로 다시 로드
    const nodes = Array.from(currentNodeIndex.values());
    const clusterCounts = !isUIVisible
      ? getVisibleClusterNodeCounts()
      : getClusterNodeCounts();
    const connectedNodes = !isUIVisible ? getConnectedNodesInBounds() : null;

    // 노드 다시 그리기
    nodes.forEach((node) => {
      const lat = parseFloat(node.latitude);
      const lng = parseFloat(node.longitude);
      const clusterId = node.cluster_label;

      if (!isNaN(lat) && !isNaN(lng)) {
        // UI 숨김 모드에서 필터링 조건들
        if (!isUIVisible) {
          // 1개 노드 클러스터는 제외
          if (clusterCounts.get(clusterId) <= 1) {
            console.log(
              `노드 ${node.node_id} 제외: 클러스터 크기 ${clusterCounts.get(
                clusterId
              )}`
            );
            return;
          }
          // 엣지로 연결되지 않은 노드는 제외
          if (!connectedNodes.has(node.node_id)) {
            console.log(`노드 ${node.node_id} 제외: 연결된 엣지 없음`);
            return;
          }
        }

        const color = getClusterColor(clusterId);
        L.circleMarker([lat, lng], {
          color: color,
          fillColor: color,
          fillOpacity: 0.7,
          radius: 2,
          weight: 0,
        }).addTo(nodeLayerGroup);
      }
    });

    // 엣지 다시 그리기
    visualizeEdges();
  }
}

// ESC 키 이벤트 리스너
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") {
    toggleUIVisibility();
  }
});

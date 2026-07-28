import cytoscape from "cytoscape";
import type { Core, EdgeSingular, NodeSingular } from "cytoscape";
import { useEffect, useRef } from "react";
import { REL_KO, TYPE_COLOR } from "./graphTypes";
import type { GraphEdge, GraphNode } from "./graphTypes";

/**
 * 컨텍스트 그래프 시각화 (Cytoscape.js, MIT).
 *
 * CDN을 쓰지 않고 번들에 포함한다 — 이 서비스는 네트워크 단절 상태에서도
 * 전 기능이 동작해야 하므로 외부 자원에 의존할 수 없다.
 */

function short(label: string, n = 14) {
  return label.length > n ? `${label.slice(0, n)}…` : label;
}

export default function GraphCanvas({
  nodes,
  edges,
  focusId,
  onNodeClick,
  onEdgeClick,
  height,
}: {
  nodes: GraphNode[];
  edges: GraphEdge[];
  focusId?: string | null;
  onNodeClick?: (id: string) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
  /** 지정하지 않으면 CSS(.graph-canvas)가 화면 높이에 맞춰 정한다 */
  height?: number;
}) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const cyRef = useRef<Core | null>(null);
  // 콜백을 ref로 들고 있어야 부모 리렌더마다 그래프를 다시 그리지 않는다
  const cbRef = useRef({ onNodeClick, onEdgeClick });
  cbRef.current = { onNodeClick, onEdgeClick };

  useEffect(() => {
    if (!boxRef.current) return;
    // 레이아웃이 캔버스 전체를 쓰도록 배치 영역을 넘긴다.
    // 지정하지 않으면 노드 몇 개가 한 구석에 뭉쳐 fit으로도 펴지지 않는다.
    const bb = {
      x1: 0,
      y1: 0,
      w: Math.max(boxRef.current.clientWidth - 60, 320),
      h: Math.max(boxRef.current.clientHeight - 60, 260),
    };
    const cy = cytoscape({
      container: boxRef.current,
      elements: [
        ...nodes.map((n) => ({
          data: { id: n.id, label: short(n.label), type: n.type, focus: n.id === focusId },
        })),
        ...edges
          .filter((e) => nodes.some((n) => n.id === e.src) && nodes.some((n) => n.id === e.dst))
          .map((e) => ({
            data: {
              id: e.id,
              source: e.src,
              target: e.dst,
              label: REL_KO[e.rel] ?? e.rel,
              disabled: !!e.disabled,
            },
          })),
      ],
      style: [
        {
          selector: "node",
          style: {
            "background-color": (n: NodeSingular) => TYPE_COLOR[n.data("type")] ?? "#8a8a8a",
            label: "data(label)",
            "font-size": 10,
            "font-family": "Malgun Gothic, sans-serif",
            color: "#1a1a1a",
            "text-valign": "bottom",
            "text-margin-y": 4,
            "text-background-color": "#fff",
            "text-background-opacity": 0.75,
            "text-background-padding": "2px",
            "text-max-width": "110px",
            "text-wrap": "ellipsis",
            width: 16,
            height: 16,
          },
        },
        {
          selector: "node[?focus]",
          style: {
            width: 30, height: 30,
            "border-width": 4, "border-color": "#FFCC00",
            "font-size": 12, "font-weight": "bold",
          },
        },
        {
          selector: "edge",
          style: {
            width: 1.4,
            "line-color": "#c9c9c9",
            "target-arrow-color": "#c9c9c9",
            "target-arrow-shape": "triangle",
            "arrow-scale": 0.7,
            "curve-style": "bezier",
            label: "data(label)",
            "font-size": 8,
            "font-family": "Malgun Gothic, sans-serif",
            color: "#8a8a8a",
            "text-background-color": "#fff",
            "text-background-opacity": 0.8,
            "text-background-padding": "1px",
          },
        },
        {
          // 비활성화된 엣지는 지우지 않고 흐리게 — 큐레이션 이력이 보여야 한다
          selector: "edge[?disabled]",
          style: {
            "line-style": "dashed",
            "line-color": "#e3a0a0",
            "target-arrow-color": "#e3a0a0",
            opacity: 0.5,
          },
        },
        { selector: "edge:selected", style: { width: 3, "line-color": "#FFCC00", "target-arrow-color": "#FFCC00" } },
      ],
      layout: {
        name: "cose",
        animate: false,
        boundingBox: bb,
        randomize: true,
        nodeRepulsion: 22000,
        idealEdgeLength: 130,
        nodeOverlap: 24,
        componentSpacing: 90,
        padding: 28,
      } as cytoscape.LayoutOptions,
      // 노드가 적을 때 cose가 과하게 확대해 라벨이 서로 덮는다 — 배율을 묶는다
      minZoom: 0.25,
      maxZoom: 1.3,
    });

    // 레이아웃이 끝나면 화면에 꽉 차게 맞춘다.
    // 노드가 적을 때 과확대되는 것은 maxZoom이 막으므로 배율을 따로 건드리지 않는다.
    cy.one("layoutstop", () => cy.fit(cy.elements(), 36));

    cy.on("tap", "node", (evt) => cbRef.current.onNodeClick?.((evt.target as NodeSingular).id()));
    cy.on("tap", "edge", (evt) => {
      const id = (evt.target as EdgeSingular).id();
      const e = edges.find((x) => x.id === id);
      if (e) cbRef.current.onEdgeClick?.(e);
    });

    cyRef.current = cy;

    // 캔버스 크기가 CSS(뷰포트 기준)로 정해지므로 창을 줄이거나 사이드바가 접히면
    // 캔버스만 커지고 그래프는 옛 크기에 갇힌다. 실제 박스 변화를 보고 다시 맞춘다.
    const ro = new ResizeObserver(() => {
      cy.resize();
      cy.fit(cy.elements(), 36);
    });
    ro.observe(boxRef.current);

    return () => {
      ro.disconnect();
      cy.destroy();
      cyRef.current = null;
    };
  }, [nodes, edges, focusId]);

  return <div ref={boxRef} className="graph-canvas" style={height ? { height } : undefined} />;
}

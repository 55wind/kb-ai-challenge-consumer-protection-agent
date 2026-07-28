/**
 * 컨텍스트 그래프 표시용 타입·색상.
 *
 * 시각화 라이브러리(cytoscape)를 import하지 않는다 — 관리자 화면이
 * 그래프 탭을 열 때만 라이브러리를 내려받도록 분리해 두기 위해서다.
 * 소비자 화면은 이 파일도 라이브러리도 쓰지 않는다.
 */

export type GraphNode = {
  id: string;
  type: string;
  label: string;
  meta?: Record<string, unknown>;
};

export type GraphEdge = {
  id: string;
  src: string;
  rel: string;
  dst: string;
  disabled?: boolean;
  provenance?: { method?: string; evidence?: string; source_id?: string };
};

export const REL_KO: Record<string, string> = {
  cites: "인용", categorized_as: "분류", similar_to: "유사", grounded_in: "근거",
  applies_to: "적용", remedied_by: "구제절차", escalates_to: "다음 단계", governed_by: "준거",
};

// 노드 유형별 색 — 관리자가 한눈에 종류를 구분해야 한다
export const TYPE_COLOR: Record<string, string> = {
  category: "#26241f",
  law: "#8a6a00",
  case: "#2e7d4f",
  procedure: "#2b5ca8",
  product: "#c2410c",
  rule: "#7c3aed",
  deadline: "#be123c",
};

const TYPE_KO: Record<string, string> = {
  category: "유형", law: "조문", case: "결정례", procedure: "절차",
  product: "KB상품", rule: "룰", deadline: "기한",
};

export const NODE_TYPE_LEGEND = Object.keys(TYPE_COLOR).map((t) => ({
  type: t,
  label: TYPE_KO[t] ?? t,
  color: TYPE_COLOR[t],
}));

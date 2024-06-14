export const MESSAGES = {
  ERROR: {
    HANDLE_OUTER_CHECK: "handleOuterCheck 에서 에러 발생: ",
    HANDLE_INNER_CHECK: "handleInnerCheck 에서 에러 발생: ",
    POPUP: "결과를 불러오지 못했습니다.",
    NOT_FOUND_IFRAME: "iframe을 찾지 못했습니다",
    HTTP: "HTTP 에러 발생: ",
    REQUEST: (method, url) => `[${method}] ${url} 요청 실패: `,
  },
  ANALYZING: {
    BUTTON: "분석 중...",
    SPAN: "분석 중입니다...",
    WAIT: "잠시만 기다려주세요.",
  },
  IS_PROMOTIONAL: {
    YES: "홍보성입니다",
    NO: "홍보성이 아닙니다",
    RATE: (rate) => `홍보성 확률: ${rate}%`,
  },
};

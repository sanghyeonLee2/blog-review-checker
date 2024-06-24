import { BLOG_DOM_QUERY_SELECTORS } from "../constants/domSelectors";
import { MESSAGES } from "../constants/messages";
import { parsePredictionResult } from "../utils/utils";

const displayResult = (result, url) => {
  const anchors = document.querySelectorAll(BLOG_DOM_QUERY_SELECTORS.A);
  const matchingAnchor = [...anchors].find((a) => a.href === url);
  if (!matchingAnchor) return;

  const parsed = parsePredictionResult(result);
  if (!parsed) return;

  const { isPromotional, percent } = parsed;
  const span = createMessageSpan(isPromotional, percent);
  appendToParent(span, matchingAnchor);
};

const createMessageSpan = (isPromo, percent) => {
  const span = document.createElement(BLOG_DOM_QUERY_SELECTORS.SPAN);
  span.innerHTML = `${
    isPromo ? MESSAGES.IS_PROMOTIONAL.YES : MESSAGES.IS_PROMOTIONAL.NO
  } 확률: ${percent}%`;

  Object.assign(span.style, {
    position: "absolute",
    top: "0",
    right: "1rem",
    backgroundColor: "#fff",
    color: "#000",
    padding: "0.3125rem",
    fontSize: "0.875rem",
    border: "1px solid #ccc",
    borderRadius: "4px",
    zIndex: "1000",
  });

  return span;
};

const appendToParent = (span, anchor) => {
  const parent = anchor.closest(".desc");
  if (parent) {
    parent.insertBefore(span, anchor.nextSibling);
  } else {
    anchor.parentElement?.appendChild(span);
  }
};

export default displayResult;

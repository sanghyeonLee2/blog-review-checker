import { BLOG_DOM_QUERY_SELECTORS } from "../constants/domSelectors";
import { MESSAGES } from "../constants/messages";

const displayResult = (result, url) => {
  const anchors = document.querySelectorAll(BLOG_DOM_QUERY_SELECTORS.A);
  const matchingAnchor = [...anchors].find((a) => a.href === url);
  if (!matchingAnchor) return;

  const { predictions, probabilities } = result ?? {};
  if (!Array.isArray(predictions) || !Array.isArray(probabilities)) return;

  const isPromo = predictions[0];
  const prob = probabilities[0][0].toFixed(2);

  const span = createMessageSpan(isPromo, prob);
  appendToParent(span, matchingAnchor);
};

const createMessageSpan = (isPromo, prob) => {
  const span = document.createElement(BLOG_DOM_QUERY_SELECTORS.SPAN);
  span.innerHTML = `${
    isPromo ? MESSAGES.IS_PROMOTIONAL.YES : MESSAGES.IS_PROMOTIONAL.NO
  } 확률: ${prob}%`;

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
    return;
  }

  anchor.parentElement?.appendChild(span);
};

export default displayResult;

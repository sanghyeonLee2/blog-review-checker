(() => {
  // scripts/constants/urls.js
  var API_URLS = {
    dev: "https://172.30.1.13:3000"
  };
  var END_POINTS = {
    FETCH_FROM_OUTER: "/fetch-from-outer",
    FETCH_FROM_INNER: "/fetch-from-inner"
  };

  // scripts/constants/messages.js
  var MESSAGES = {
    ERROR: {
      HANDLE_OUTER_CHECK: "handleOuterCheck \uC5D0\uC11C \uC5D0\uB7EC \uBC1C\uC0DD: ",
      HANDLE_INNER_CHECK: "handleInnerCheck \uC5D0\uC11C \uC5D0\uB7EC \uBC1C\uC0DD: ",
      POPUP: "\uACB0\uACFC\uB97C \uBD88\uB7EC\uC624\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4.",
      NOT_FOUND_IFRAME: "iframe\uC744 \uCC3E\uC9C0 \uBABB\uD588\uC2B5\uB2C8\uB2E4",
      HTTP: "HTTP \uC5D0\uB7EC \uBC1C\uC0DD: ",
      REQUEST: (method, url) => `[${method}] ${url} \uC694\uCCAD \uC2E4\uD328: `
    },
    ANALYZING: {
      BUTTON: "\uBD84\uC11D \uC911...",
      SPAN: "\uBD84\uC11D \uC911\uC785\uB2C8\uB2E4...",
      WAIT: "\uC7A0\uC2DC\uB9CC \uAE30\uB2E4\uB824\uC8FC\uC138\uC694."
    },
    IS_PROMOTIONAL: {
      YES: "\uD64D\uBCF4\uC131\uC785\uB2C8\uB2E4",
      NO: "\uD64D\uBCF4\uC131\uC774 \uC544\uB2D9\uB2C8\uB2E4",
      RATE: (rate) => `\uD64D\uBCF4\uC131 \uD655\uB960: ${rate}%`
    }
  };

  // scripts/api/request.js
  var postRequest = async (endPoint, data, timeout = 5e3) => {
    const url = `${API_URLS.dev}${endPoint}`;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(data),
        signal: controller.signal
      });
      clearTimeout(timeoutId);
      if (!response.ok) {
        throw new Error(`${MESSAGES.HTTP} ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(MESSAGES.ERROR.REQUEST("POST", url), error);
      throw error;
    }
  };
  var request_default = postRequest;

  // scripts/api/apis.js
  var fetchFromOuter = (blogUrl) => request_default(END_POINTS.FETCH_FROM_OUTER, { blogUrl });
  var fetchFromInner = (data) => request_default(END_POINTS.FETCH_FROM_INNER, data);

  // scripts/constants/domSelectors.js
  var BLOG_DOM_QUERY_SELECTORS = {
    IFRAME: "iframe",
    TEXT: 'span[class^="se-fs-"]',
    IMG: 'img[class$="egjs-visible"]',
    TITLE: "h3.se_textarea",
    A: "a",
    SPAN: "span",
    ID: {
      FETCH_BUTTON: "fetchContent",
      BUTTON_TEXT: "buttonText",
      RESULT: "result",
      RATE_RESULT: "rateResult"
    }
  };

  // scripts/handlers/handleInnerContentCheck.js
  var handleInnerContentCheck = async (_, sendResponse) => {
    const iframe = document.querySelector(BLOG_DOM_QUERY_SELECTORS.IFRAME);
    if (!iframe) {
      sendResponse(MESSAGES.ERROR.NOT_FOUND_IFRAME);
      return;
    }
    try {
      const doc = iframe.contentDocument || iframe.contentWindow.document;
      const textBlocks = doc.querySelectorAll(BLOG_DOM_QUERY_SELECTORS.TEXT);
      const contentText = [...textBlocks].map((el) => el.innerText).join("\n");
      const imgs = doc.querySelectorAll(BLOG_DOM_QUERY_SELECTORS.IMG);
      const imageUrl = imgs.length > 0 ? imgs[imgs.length - 1].src : "";
      const titleElement = doc.querySelector(BLOG_DOM_QUERY_SELECTORS.TITLE);
      const title = titleElement ? titleElement.innerText : "";
      const result = await fetchFromInner({ title, contentText, imageUrl });
      sendResponse(result);
    } catch (err) {
      console.error(MESSAGES.ERROR.HANDLE_INNER_CHECK, err);
      sendResponse({ error: err.message });
    }
  };
  var handleInnerContentCheck_default = handleInnerContentCheck;

  // scripts/ui/displayResult.js
  var displayResult = (result, url) => {
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
  var createMessageSpan = (isPromo, prob) => {
    const span = document.createElement(BLOG_DOM_QUERY_SELECTORS.SPAN);
    span.innerHTML = `${isPromo ? MESSAGES.IS_PROMOTIONAL.YES : MESSAGES.IS_PROMOTIONAL.NO} \uD655\uB960: ${prob}%`;
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
      zIndex: "1000"
    });
    return span;
  };
  var appendToParent = (span, anchor) => {
    const parent = anchor.closest(".desc");
    if (parent) {
      parent.insertBefore(span, anchor.nextSibling);
      return;
    }
    anchor.parentElement?.appendChild(span);
  };
  var displayResult_default = displayResult;

  // scripts/handlers/handleOuterContentCheck.js
  var handleOuterContentCheck = async (request, sendResponse) => {
    try {
      const result = await fetchFromOuter(request.url);
      displayResult_default(result, request.url);
      sendResponse(result);
    } catch (err) {
      console.error(MESSAGES.ERROR.HANDLE_OUTER_CHECK, err);
      sendResponse({ error: err.message });
    }
  };
  var handleOuterContentCheck_default = handleOuterContentCheck;

  // scripts/constants/eventActions.js
  var EVENT_ACTIONS = {
    FETCH_AT_OUTER: "fetchAtOuter",
    FETCH_AT_INNER: "fetchAtInner"
  };

  // content/content.js
  chrome.runtime.onMessage.addListener((request, _, sendResponse) => {
    if (request.action === EVENT_ACTIONS.FETCH_AT_OUTER) {
      handleOuterContentCheck_default(request, sendResponse);
      return true;
    }
    if (request.action === EVENT_ACTIONS.FETCH_AT_INNER) {
      handleInnerContentCheck_default(request, sendResponse);
      return true;
    }
  });
})();

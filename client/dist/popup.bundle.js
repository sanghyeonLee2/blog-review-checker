(() => {
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

  // scripts/constants/eventActions.js
  var EVENT_ACTIONS = {
    FETCH_AT_OUTER: "fetchAtOuter",
    FETCH_AT_INNER: "fetchAtInner"
  };

  // popup/utils.js
  var updateResult = (mainText, subText = "") => {
    const resultElement = document.getElementById(
      BLOG_DOM_QUERY_SELECTORS.ID.RESULT
    );
    const rateResultElement = document.getElementById(
      BLOG_DOM_QUERY_SELECTORS.ID.RATE_RESULT
    );
    if (resultElement) resultElement.textContent = mainText;
    if (rateResultElement) rateResultElement.textContent = subText;
  };

  // popup/handlers.js
  var handleMessageResponse = (response, button, buttonText) => {
    if (buttonText) buttonText.textContent = "\uAD11\uACE0\uC131 \uCCB4\uD06C";
    if (button) button.disabled = false;
    const { predictions, probabilities } = response ?? {};
    if (!Array.isArray(predictions) || !Array.isArray(probabilities)) {
      return updateResult(MESSAGES.ERROR.POPUP);
    }
    const isPromotional = predictions[0];
    const probability = probabilities[0][0];
    if (isPromotional === 1) {
      return updateResult(
        MESSAGES.IS_PROMOTIONAL.YES,
        MESSAGES.IS_PROMOTIONAL.RATE(probability.toFixed(2))
      );
    }
    return updateResult(
      MESSAGES.IS_PROMOTIONAL.NO,
      MESSAGES.IS_PROMOTIONAL.RATE(probability.toFixed(2))
    );
  };

  // popup/popup.js
  document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById(
      BLOG_DOM_QUERY_SELECTORS.ID.FETCH_BUTTON
    );
    const buttonText = document.getElementById(
      BLOG_DOM_QUERY_SELECTORS.ID.BUTTON_TEXT
    );
    button.addEventListener("click", () => {
      updateResult(MESSAGES.ANALYZING.SPAN, MESSAGES.ANALYZING.WAIT);
      buttonText.textContent = MESSAGES.ANALYZING.BUTTON;
      button.disabled = true;
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        const tabId = tabs[0].id;
        chrome.tabs.sendMessage(
          tabId,
          { action: EVENT_ACTIONS.FETCH_AT_INNER },
          (response) => handleMessageResponse(response, button, buttonText)
        );
      });
    });
  });
})();

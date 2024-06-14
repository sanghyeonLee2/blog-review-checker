import { BLOG_DOM_QUERY_SELECTORS } from "../scripts/constants/domSelectors";
import { MESSAGES } from "../scripts/constants/messages";
import { EVENT_ACTIONS } from "./../scripts/constants/eventActions";
import { updateResult } from "./utils";
import { handleMessageResponse } from "./handlers";

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

import {
  handleInnerContentCheck,
  handleOuterContentCheck,
} from "../scripts/handlers";
import { EVENT_ACTIONS } from "../scripts/constants/eventActions";

chrome.runtime.onMessage.addListener((request, _, sendResponse) => {
  if (request.action === EVENT_ACTIONS.FETCH_AT_OUTER) {
    handleOuterContentCheck(request, sendResponse);
    return true;
  }

  if (request.action === EVENT_ACTIONS.FETCH_AT_INNER) {
    handleInnerContentCheck(request, sendResponse);
    return true;
  }
});

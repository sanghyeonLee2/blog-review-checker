import { handleInnerContentCheck } from "./contentInnerChecker.js";
import { handleOuterCheck } from "./contentOuterChecker.js";

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "fetchPageTitle") {
    handleOuterCheck(request, sendResponse);
    return true;
  }

  if (request.action === "fetchBlogContent") {
    handleInnerContentCheck(request, sendResponse);
    return true;
  }
});

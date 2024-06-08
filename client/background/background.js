import { EVENT_ACTIONS } from "../scripts/constants/eventActions";
import handleLinkAnalysis from "../scripts/handlers/handleLinkAnalysis";

const OPTION_TITLE = "광고성 체크";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: EVENT_ACTIONS.FETCH_AT_OUTER,
      title: OPTION_TITLE,
      contexts: ["link"],
    });
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId !== EVENT_ACTIONS.FETCH_AT_OUTER) return;

  handleLinkAnalysis(tab.id, info.linkUrl);
});
